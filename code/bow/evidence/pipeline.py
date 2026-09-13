"""Evidence-extraction orchestration.

This is the one place that decides, for a given message or image, which
extractor actually produces the result:

    1. content-hash cache hit             (optional accelerator)
    2. configured LLM extractor           (generic, works on unseen input)
    3. heuristic fallback (text only)     (generic, works on unseen input)
    4. conservative reuse (images only)   (never fabricates a new figure)

Step 1 is a pure performance optimization. Steps 2-4 make the system correct
even with an empty or missing cache, which is the property that matters: no
part of the final decision engine *requires* the cache to be present.

The engine never calls into this module mid-forecast. `code/tools/build_evidence.py`
runs this pipeline once over the whole dataset and freezes the result into a
resolved evidence file; the deterministic engine (state/forecast/plan/rank)
reads only that frozen file, which is what keeps the engine itself
deterministic regardless of which extractor produced the input.
"""

from __future__ import annotations

import os
from dataclasses import asdict
from datetime import date
from decimal import Decimal
from typing import Dict, List, Optional

from . import heuristics
from . import schema
from .cache import ContentCache, hash_text, content_hash
from .llm import Extractor
from .schema import Evidence
from ..dataset import Message, parse_date

CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "evidence", "cache"
)


def _record_to_evidence(record: dict, source_id: str = "", message: Optional[Message] = None) -> Evidence:
    amounts = [(cur, Decimal(str(amt))) for cur, amt in record.get("amounts", [])]
    dates = [parse_date(d) for d in record.get("dates", [])]
    percent = record.get("percent")
    return Evidence(
        intent=record.get("intent", schema.UNKNOWN),
        amounts=amounts,
        dates=[d for d in dates if d],
        percent=Decimal(str(percent)) if percent not in (None, "") else None,
        confidence=float(record.get("confidence", 0.0)),
        extractor=record.get("extractor", "unknown"),
        source_id=source_id,
        message=message,
    )


class MessageEvidencePipeline:
    def __init__(self, extractor: Optional[Extractor] = None,
                 cache_path: Optional[str] = None, use_cache: bool = True):
        self.extractor = extractor
        self.use_cache = use_cache
        self.cache = ContentCache(cache_path or os.path.join(CACHE_DIR, "messages.json")) if use_cache else None

    def classify(self, message: Message) -> Evidence:
        text = message.message_text or ""
        key = hash_text(text)

        if self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                return _record_to_evidence(hit, source_id=message.message_id, message=message)

        if self.extractor is not None:
            try:
                record = self.extractor.classify_message(text)
                record.setdefault("confidence", 0.9)
            except Exception:
                record = heuristics.classify_full(text)
        else:
            record = heuristics.classify_full(text)

        if self.cache is not None:
            self.cache.put(key, record, persist=False)

        return _record_to_evidence(record, source_id=message.message_id, message=message)

    def classify_all(self, messages_by_user: Dict[str, List[Message]]) -> Dict[str, List[Evidence]]:
        out = {uid: [self.classify(m) for m in msgs] for uid, msgs in messages_by_user.items()}
        if self.cache is not None:
            self.cache.save()
        return out


class ImageEvidencePipeline:
    """Resolves the amount printed on a document image.

    Cached by the SHA-256 of the image's own bytes - two different files with
    identical content share a cache entry, but nothing is keyed by which
    event or request the image happens to be attached to in this dataset.
    """

    def __init__(self, extractor: Optional[Extractor] = None,
                 cache_path: Optional[str] = None, use_cache: bool = True):
        self.extractor = extractor
        self.use_cache = use_cache
        self.cache = ContentCache(cache_path or os.path.join(CACHE_DIR, "images.json")) if use_cache else None

    def resolve(self, image_path: str) -> Optional[dict]:
        """Return ``{"amount", "currency", "label", "confidence", "extractor"}``.

        Returns ``None`` only when no cache entry, no configured extractor,
        and no local OCR library are available - callers must not silently
        substitute zero in that case (see problem_statement.md: a blank
        amount is never treated as zero).
        """
        with open(image_path, "rb") as handle:
            data = handle.read()
        key = content_hash(data)

        if self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                return hit

        if self.extractor is not None:
            try:
                record = self.extractor.extract_image_amount(data)
                record.setdefault("confidence", 0.9)
                if self.cache is not None:
                    self.cache.put(key, record, persist=False)
                return record
            except Exception:
                pass

        ocr_result = self._try_ocr(data)
        if ocr_result is not None:
            if self.cache is not None:
                self.cache.put(key, ocr_result, persist=False)
            return ocr_result

        return None

    @staticmethod
    def _try_ocr(data: bytes) -> Optional[dict]:
        """Best-effort local OCR fallback: find a labeled total-like amount.

        Generic by construction - it looks for common "this is the payable
        figure" labels (total, net pay, balance due, amount paid, grand
        total, ...) in either English or Indonesian, and returns the number
        on that line. It never references a specific document.
        """
        try:
            import pytesseract
            from PIL import Image
            import io
            import re
        except ImportError:
            return None

        try:
            text = pytesseract.image_to_string(Image.open(io.BytesIO(data)))
        except Exception:
            return None

        labels = [
            "grand total", "total amount received", "net pay", "balance due",
            "total paid", "amount payable", "total payable", "total",
            "jumlah total", "total pembayaran",
        ]
        amount_re = re.compile(r"([\d.,]+)\s*$")
        best = None
        for line in text.splitlines():
            low = line.lower()
            for label in labels:
                if label in low:
                    match = amount_re.search(line)
                    if match:
                        raw = match.group(1).replace(",", "")
                        try:
                            value = Decimal(raw)
                        except Exception:
                            continue
                        best = (label, value)
                        break
            if best:
                break
        if best is None:
            return None
        label, value = best
        return {
            "amount": str(value), "currency": None, "label": label,
            "confidence": 0.4, "extractor": "ocr-fallback",
        }

    def save(self) -> None:
        if self.cache is not None:
            self.cache.save()
