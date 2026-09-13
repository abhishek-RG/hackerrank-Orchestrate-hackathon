"""Final structural validation of one output row, independent of however it
was produced.

This is deliberately redundant with the checks already applied while
building candidates - it re-derives what a *valid* row must look like
straight from problem_statement.md and flags anything that slipped through,
so a bug in candidate generation cannot silently reach ``output.csv``.
Returns a list of human-readable problems; an empty list means the row is
structurally sound (it says nothing about whether the underlying financial
judgement is the *best* one).
"""

from __future__ import annotations

from decimal import Decimal
from typing import List, Optional

from .candidates import Candidate
from .dataset import PaymentOption, Profile, Request
from .output import fmt_amount

VALID_STATUSES = {"affordable_now", "affordable_with_plan", "affordable_later", "not_affordable"}
VALID_METHODS = {"full_payment", "partial_payment", "installments", "wait", "not_recommended"}


def validate_row(request: Request, profile: Profile, row: dict,
                 winner: Optional[Candidate], options: List[PaymentOption]) -> List[str]:
    problems: List[str] = []

    status = row["affordability_status"]
    method = row["recommended_payment_method"]
    if status not in VALID_STATUSES:
        problems.append(f"invalid affordability_status {status!r}")
    if method not in VALID_METHODS:
        problems.append(f"invalid recommended_payment_method {method!r}")

    amount_safe = Decimal(str(row["amount_safe_to_pay"]))
    if not (Decimal(0) <= amount_safe <= request.requested_amount):
        problems.append(f"amount_safe_to_pay {amount_safe} out of bounds [0, {request.requested_amount}]")

    changes = row["spending_changes_needed"]
    n_changes = 0 if changes == "none" else len(changes.split("|"))
    if n_changes > 3:
        problems.append(f"too many spending changes ({n_changes} > 3)")
    if changes != "none":
        seen_events = set()
        for part in changes.split("|"):
            pieces = part.split(":")
            event_id = pieces[1] if len(pieces) > 1 else ""
            if event_id in seen_events:
                problems.append(f"event {event_id} targeted by more than one spending change")
            seen_events.add(event_id)

    if status == "affordable_now":
        if row["earliest_date_for_full_payment"] != request.request_date.isoformat():
            problems.append("affordable_now requires earliest_date_for_full_payment == request_date")
        if method != "full_payment":
            problems.append("affordable_now should recommend full_payment")

    if method == "not_recommended":
        if row["payment_plan"] != "none":
            problems.append("not_recommended must have payment_plan = none")
        if status != "not_affordable":
            problems.append("not_recommended should pair with not_affordable")

    if winner is not None:
        total = sum((amt for _, amt in winner.schedule), Decimal(0))
        dates = [d for d, _ in winner.schedule]
        if dates != sorted(dates):
            problems.append("payment_plan is not chronological")

        if winner.method == "partial_payment":
            if len(winner.schedule) != 2:
                problems.append("partial_payment must have exactly two payments")
            elif total != request.requested_amount:
                problems.append(f"partial_payment payments sum to {total}, expected {request.requested_amount}")
            if not request.allows_partial_payment:
                problems.append("partial_payment used but request does not allow it")

        if winner.method == "installments":
            match = next((o for o in options
                         if o.payment_option_id == winner.payment_option_id), None)
            if match is None:
                problems.append(f"installments reference unknown option {winner.payment_option_id}")
            else:
                expected = match.schedule()
                if [d for d, _ in expected] != [d for d, _ in winner.schedule] or \
                        any(fmt_amount(a) != fmt_amount(b) for (_, a), (_, b) in zip(expected, winner.schedule)):
                    problems.append(f"installments schedule does not match {winner.payment_option_id}")

        if winner.method in ("full_payment", "wait") and total != request.requested_amount:
            problems.append(f"{winner.method} should pay the full requested amount, got {total}")

    return problems
