"""Renders the two free-text output columns: ``decision_explanation`` and
``spending_changes_needed``.

Every number and date placed into these strings is read off the same
candidate/state objects used to build the row - nothing here recomputes or
guesses a figure independently, so the explanation cannot drift from the
machine-checked plan it describes.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import List, Optional

from .candidates import Candidate, SpendingChange, fmt_plain
from .dataset import Profile, Request
from .state import State

_MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]


def human_date(d: date) -> str:
    return f"{d.day} {_MONTHS[d.month - 1]} {d.year}"


def money(amount: Decimal, currency: str) -> str:
    normalized = amount.normalize()
    if normalized == normalized.to_integral_value():
        whole, frac = f"{int(normalized):,}", ""
    else:
        text = f"{normalized:,.2f}"
        whole, frac = text.split(".")
        frac = "." + frac
    return f"{currency} {whole}{frac}"


def _flow_label(state: State, event_id: str) -> str:
    for flow in state.flows:
        if flow.event_id == event_id:
            return flow.label
    return event_id


def _change_clause(change: SpendingChange, state: State, currency: str) -> str:
    label = _flow_label(state, change.event_id).lower()
    if change.op == "stop":
        return f"stop the {label}"
    return f"reduce the {label} to {money(change.new_amount, currency)}"


def _changes_prefix(changes: List[SpendingChange], state: State, currency: str) -> str:
    if not changes:
        return ""
    clauses = [_change_clause(c, state, currency) for c in changes]
    if len(clauses) == 1:
        joined = clauses[0]
    else:
        joined = ", ".join(clauses[:-1]) + f", and {clauses[-1]}"
    return joined[0].upper() + joined[1:]


def spending_changes_text(changes: List[SpendingChange]) -> str:
    if not changes:
        return "none"
    return "|".join(c.as_text() for c in changes)


def decision_explanation(profile: Profile, request: Request, state: State,
                         winner: Optional[Candidate], status: str) -> str:
    currency = profile.home_currency
    minimum = profile.minimum_balance_to_keep

    if winner is None:
        return (
            f"Do not make this payment by {human_date(request.desired_completion_date)}. "
            f"None of the available options keeps the {money(minimum, currency)} minimum protected."
        )

    if winner.method == "wait":
        when, amount = winner.schedule[0]
        return (
            f"Wait until {human_date(when)}, then pay {money(amount, currency)} in full. "
            f"Paying sooner would put the {money(minimum, currency)} minimum at risk."
        )

    prefix = _changes_prefix(winner.changes, state, currency)

    if winner.method == "full_payment":
        when, amount = winner.schedule[0]
        tail = f"pay {money(amount, currency)} today"
        sentence = f"{prefix}, then {tail}." if prefix else f"{tail[0].upper()}{tail[1:]}."
        return f"{sentence} This leaves at least {money(minimum, currency)} available over the next 90 days."

    if winner.method == "partial_payment":
        (d1, a1), (d2, a2) = winner.schedule
        tail = (f"pay {money(a1, currency)} today and the remaining {money(a2, currency)} "
               f"on {human_date(d2)}")
        sentence = f"{prefix}, then {tail}." if prefix else f"{tail[0].upper()}{tail[1:]}."
        return f"{sentence} This leaves at least {money(minimum, currency)} available."

    if winner.method == "installments":
        n = winner.num_payments
        first_date, first_amount = winner.schedule[0]
        tail = f"use {n} installments of {money(first_amount, currency)}, starting {human_date(first_date)}"
        sentence = f"{prefix}, then {tail}." if prefix else f"{tail[0].upper()}{tail[1:]}."
        return f"{sentence} This leaves at least {money(minimum, currency)} available."

    return "No safe payment plan was found within the forecast period."
