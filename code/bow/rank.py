"""Pick the winning candidate using the exact ranking rules from
problem_statement.md, "Choosing Between Safe Plans":

    1. Complete the full request by desired_completion_date.
    2. Require no spending changes.
    3. Minimize the total amount paid.
    4. Start payment earlier.
    5. Use fewer payments.
    6. Use the lowest payment_option_id as the final tie-breaker.

Every :class:`~bow.candidates.Candidate` reaching this module has already
been proven safe and rule-eligible (:mod:`candidates` only builds candidates
whose schedule finishes by ``desired_completion_date`` and whose payment
dates never breach ``minimum_balance_to_keep``), so criterion 1 is already
satisfied by construction; this module applies criteria 2-6 as a sort key.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import List, Optional

from .candidates import Candidate

_METHOD_TO_STATUS = {
    "full_payment": "affordable_now",       # only when it starts on request_date
    "partial_payment": "affordable_with_plan",
    "installments": "affordable_with_plan",
    "wait": "affordable_later",
}


def _option_id_key(candidate: Candidate):
    if candidate.payment_option_id is None:
        return (1, "")
    return (0, candidate.payment_option_id)


def _sort_key(candidate: Candidate):
    return (
        len(candidate.changes),
        candidate.total_paid,
        candidate.starts,
        candidate.num_payments,
        _option_id_key(candidate),
    )


def pick_winner(candidates: List[Candidate]) -> Optional[Candidate]:
    if not candidates:
        return None
    return sorted(candidates, key=_sort_key)[0]


def affordability_status(candidate: Candidate, request_date: date) -> str:
    if candidate.method == "full_payment" and candidate.starts == request_date:
        # A full payment that only clears thanks to a spending change is
        # "completed ... through ... permitted spending changes" - explicitly
        # one of the affordable_with_plan pathways, not affordable_now.
        return "affordable_with_plan" if candidate.changes else "affordable_now"
    return _METHOD_TO_STATUS[candidate.method]
