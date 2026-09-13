"""Public evidence-layer API.

Two entry points matter to the rest of the system:

* :func:`interpret_all` turns every user's messages into :class:`Evidence`
  objects.
* :class:`ImageAmountResolver` turns a blank-amount event into a
  ``(Decimal, currency)`` pair by resolving its linked image.

Both go through :mod:`pipeline`, which prefers a configured LLM extractor,
then a generic heuristic/OCR fallback, and only then an optional
content-addressed cache - see ``pipeline.py`` for why the cache is an
accelerator, not a dependency.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from ..dataset import ImageRef, Message
from . import schema
from .llm import configured_extractor
from .pipeline import ImageEvidencePipeline, MessageEvidencePipeline
from .schema import (
    Evidence,
    INERT,
    UNCONFIRMED_INCOME,
    UNKNOWN,
    SALARY_CHANGE,
    SALARY_ONE_OFF,
    SALARY_SCHEDULE_CHANGE,
    SALARY_STOPPED,
    INCOME_UNCONFIRMED,
    INCOME_NOT_RECURRING,
    BANK_ADMIN,
    DEBIT_WILL_RETRY,
    WINDFALL_PENDING,
    WINDFALL_SETTLED,
    INVESTMENT_PAPER_MOVE,
    UPFRONT_FEE_SCAM,
    REFUND_PENDING,
    FX_PENDING,
    DOCUMENT_IS_FINAL,
    OBLIGATION_CONFIRMED,
    RECURRING_COST_CHANGE,
    PAYOUT_NOT_YET_AVAILABLE,
)

__all__ = [
    "Evidence", "interpret_all", "ImageAmountResolver",
    "INERT", "UNCONFIRMED_INCOME", "UNKNOWN",
    "SALARY_CHANGE", "SALARY_ONE_OFF", "SALARY_SCHEDULE_CHANGE", "SALARY_STOPPED",
    "INCOME_UNCONFIRMED", "INCOME_NOT_RECURRING", "BANK_ADMIN", "DEBIT_WILL_RETRY",
    "WINDFALL_PENDING", "WINDFALL_SETTLED", "INVESTMENT_PAPER_MOVE", "UPFRONT_FEE_SCAM",
    "REFUND_PENDING", "FX_PENDING", "DOCUMENT_IS_FINAL", "OBLIGATION_CONFIRMED",
    "RECURRING_COST_CHANGE", "PAYOUT_NOT_YET_AVAILABLE",
]


def interpret_all(messages_by_user: Dict[str, List[Message]],
                  pipeline: Optional[MessageEvidencePipeline] = None,
                  use_cache: bool = True) -> Dict[str, List[Evidence]]:
    pipeline = pipeline or MessageEvidencePipeline(extractor=configured_extractor(), use_cache=use_cache)
    return pipeline.classify_all(messages_by_user)


class ImageAmountResolver:
    """Resolves a blank-amount event's figure via its linked image.

    Constructed with the dataset's ``images_by_event`` map (event_id ->
    :class:`ImageRef`) so callers can keep asking "what is event X's
    amount" without knowing anything about hashing or caching.
    """

    def __init__(self, images_by_event: Dict[str, ImageRef],
                pipeline: Optional[ImageEvidencePipeline] = None,
                use_cache: bool = True):
        self._images_by_event = images_by_event
        self._pipeline = pipeline or ImageEvidencePipeline(
            extractor=configured_extractor(), use_cache=use_cache
        )

    def get(self, event_id: str):
        image = self._images_by_event.get(event_id)
        if image is None:
            return None
        result = self._pipeline.resolve(image.path())
        if result is None or result.get("amount") is None:
            return None
        from decimal import Decimal
        return Decimal(str(result["amount"])), result.get("currency")

    def save(self) -> None:
        self._pipeline.save()
