"""Content-addressed cache for evidence extraction.

The cache key is a SHA-256 hash of the raw content being classified (message
text, or image bytes) - never a `message_id`, `event_id`, or `request_id`. That
means:

* two different messages that happen to say the same thing share one entry,
  which is a normal cache hit, not an answer lookup;
* a message or image this cache has never seen produces a genuine miss and
  must go through an extractor (LLM or heuristic) to get a result;
* nothing in this file can be read as "the answer for request_87" - there is
  no key that names a request, user, or event anywhere in the store.

This is explicitly a **development/runtime cache for extractor output**, not a
source of financial truth. Entries store only low-level extracted signals
(intent, amount, currency, date, percent, confidence) - never a final
decision field such as `amount_safe_to_pay` or `affordability_status`.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional

_LOCK = threading.Lock()


def content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_text(text: str) -> str:
    return content_hash(text.encode("utf-8"))


class ContentCache:
    """A flat JSON store of {content_hash: extraction_record}."""

    SCHEMA_NOTE = (
        "development cache; keys are SHA-256 content hashes, not dataset "
        "identifiers; values are low-level extraction signals only, never "
        "final decision fields; regenerate with code/tools/build_evidence_cache.py"
    )

    def __init__(self, path: str):
        self.path = path
        self._data: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as handle:
                blob = json.load(handle)
            self._data = blob.get("entries", {})

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        blob = {"_schema": self.SCHEMA_NOTE, "entries": self._data}
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(blob, handle, indent=2, sort_keys=True, ensure_ascii=False)
        os.replace(tmp, self.path)

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        return self._data.get(key)

    def put(self, key: str, record: Dict[str, Any], persist: bool = True) -> None:
        record = dict(record)
        record.setdefault("cached_at", datetime.now(timezone.utc).isoformat())
        with _LOCK:
            self._data[key] = record
            if persist:
                self.save()

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: str) -> bool:
        return key in self._data
