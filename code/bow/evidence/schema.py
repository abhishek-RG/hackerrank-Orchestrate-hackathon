"""The evidence taxonomy: what a financial message or document can mean.

This module only defines *categories of meaning* that occur in ordinary
payroll, banking, merchant, and financial-service correspondence. Nothing
here is tied to a specific request, user, or dataset row - the same intents
apply to a message this system has never seen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from ..dataset import Message

# --------------------------------------------------------------------------
# intents (generic financial-correspondence categories)
# --------------------------------------------------------------------------

SALARY_CHANGE = "salary_change"                 # pay raised, cut, or made temporary
SALARY_ONE_OFF = "salary_one_off"               # arrears / one-time adjustment alongside regular pay
SALARY_SCHEDULE_CHANGE = "salary_schedule_change"  # pay date moved or confirmed
SALARY_STOPPED = "salary_stopped"               # employment or contract ended, no more pay
INCOME_UNCONFIRMED = "income_unconfirmed"       # bonus/commission/gig payout not yet earned or payable
INCOME_NOT_RECURRING = "income_not_recurring"   # a credit that is a one-off, not a paycheck (reimbursement)

BANK_ADMIN = "bank_admin"                       # internal transfer, dispute status, multi-account note
DEBIT_WILL_RETRY = "debit_will_retry"           # a failed charge will be attempted again

WINDFALL_PENDING = "windfall_pending"           # prize/claim not yet paid out
WINDFALL_SETTLED = "windfall_settled"           # prize/sale proceeds landed, closed
INVESTMENT_PAPER_MOVE = "investment_paper_move"  # unrealized value change, no cash moved
UPFRONT_FEE_SCAM = "upfront_fee_scam"           # asks the user to pay first to "release" funds

REFUND_PENDING = "refund_pending"               # money coming back has not settled
FX_PENDING = "fx_pending"                       # a foreign-currency amount awaits settlement-date conversion
DOCUMENT_IS_FINAL = "document_is_final"         # this document states the settled figure

OBLIGATION_CONFIRMED = "obligation_confirmed"   # a payer confirmed an invoice/payment amount
RECURRING_COST_CHANGE = "recurring_cost_change"  # a bill/rent amount is changing going forward
PAYOUT_NOT_YET_AVAILABLE = "payout_not_yet_available"  # gig/freelance earnings not withdrawable yet

UNKNOWN = "unknown"

#: Intents that describe money not to be counted until it actually settles.
UNCONFIRMED_INCOME = {
    INCOME_UNCONFIRMED, WINDFALL_PENDING, REFUND_PENDING, PAYOUT_NOT_YET_AVAILABLE,
}

#: Intents that carry no cash-flow adjustment by themselves.
INERT = {
    BANK_ADMIN, WINDFALL_SETTLED, INVESTMENT_PAPER_MOVE, DOCUMENT_IS_FINAL,
    FX_PENDING, UPFRONT_FEE_SCAM, INCOME_NOT_RECURRING,
}


@dataclass
class Evidence:
    """One interpreted piece of evidence (a message or, later, a document)."""

    intent: str
    amounts: List[tuple] = field(default_factory=list)   # [(currency, Decimal), ...]
    dates: List[date] = field(default_factory=list)
    percent: Optional[Decimal] = None
    confidence: float = 0.0
    extractor: str = "unknown"
    source_id: str = ""            # message_id / image_id, for logging only
    message: Optional["Message"] = None   # provenance only; never read for its text at decision time

    @property
    def amount(self) -> Optional[Decimal]:
        return self.amounts[0][1] if self.amounts else None

    @property
    def currency(self) -> Optional[str]:
        return self.amounts[0][0].upper() if self.amounts else None

    @property
    def when(self) -> Optional[date]:
        return self.dates[0] if self.dates else None
