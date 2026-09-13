#!/usr/bin/env python3
"""Populate the evidence cache by running the real extraction pipeline once
over every message and image in ``dataset/``.

This is the reproducible path referenced by ``code/evidence/cache/*.json``'s
own ``_schema`` note. Running it again is always safe:

* messages go through :mod:`bow.evidence.pipeline` - a configured LLM
  extractor (``ANTHROPIC_API_KEY`` set) if available, otherwise the generic
  heuristic classifier in :mod:`bow.evidence.heuristics`;
* images go through the same pipeline's vision path when an extractor is
  configured; without one, entries already in the cache are left as-is (they
  were produced by an interactive vision read during development - see each
  entry's own ``_doc_note`` and ``extractor`` field) and any image with no
  cache entry and no configured extractor is reported, not guessed.

Usage::

    python code/tools/build_evidence_cache.py
    ANTHROPIC_API_KEY=... python code/tools/build_evidence_cache.py --force
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bow import dataset
from bow.evidence.cache import content_hash
from bow.evidence.llm import configured_extractor
from bow.evidence.pipeline import ImageEvidencePipeline, MessageEvidencePipeline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                       help="re-classify every message even if already cached")
    args = parser.parse_args()

    data = dataset.load()
    extractor = configured_extractor()
    print(f"extractor: {'configured (' + type(extractor).__name__ + ')' if extractor else 'none (heuristic fallback)'}")

    msg_pipeline = MessageEvidencePipeline(extractor=extractor)
    if args.force:
        msg_pipeline.cache._data.clear()
    before = len(msg_pipeline.cache)
    evidence = msg_pipeline.classify_all(data.messages_by_user)
    total_messages = sum(len(v) for v in data.messages_by_user.values())
    print(f"messages: {total_messages} classified, cache grew {before} -> {len(msg_pipeline.cache)}")

    unknown = [(uid, e) for uid, lst in evidence.items() for e in lst if e.intent == "unknown"]
    if unknown:
        print(f"  WARNING: {len(unknown)} message(s) classified as 'unknown' "
             "(low-confidence heuristic miss) - inspect these:")
        for uid, e in unknown[:10]:
            print(f"    {e.source_id} (user {uid}): {e.message.message_text[:80]!r}")

    img_pipeline = ImageEvidencePipeline(extractor=extractor)
    resolved, missing = 0, []
    for image in data.images:
        result = img_pipeline.resolve(image.path())
        if result is None:
            missing.append(image.image_id)
        else:
            resolved += 1
    img_pipeline.save()
    print(f"images: {resolved}/{len(data.images)} resolved, cache has {len(img_pipeline.cache)} entries")
    if missing:
        print(f"  MISSING (no cache entry, no extractor configured): {missing}")
        print("  Set ANTHROPIC_API_KEY and re-run, or review these manually - "
             "a blank event amount must never be treated as zero.")

    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
