"""Deterministic 90-day cash-flow safety checks.

Everything here operates on a fixed, already-reconstructed set of dated
:class:`~bow.state.Flow` objects plus an opening balance - it does no
evidence interpretation and makes no assumptions beyond what the caller
supplies. Given the same flows and opening balance, every function below
returns the same answer every time (plain ``Decimal`` arithmetic, no
randomness, no wall-clock reads).

Core idea: between two flow dates the balance is constant, so the minimum
balance over any date range is realized either at the range's start (the
level carried in from before it) or at one of the flow dates inside it. That
makes an exact per-day answer computable from the sparse event ledger,
without expanding to a day-by-day array.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from typing import Iterable, List, Optional, Sequence, Tuple

from .state import Flow


@dataclass(frozen=True)
class Point:
    when: date
    balance: Decimal   # balance immediately after this date's net flows


def build_points(opening_balance: Decimal, flows: Iterable[Flow],
                 start: date, end: date) -> List[Point]:
    """Collapse ``flows`` into one running-balance point per distinct date.

    Only flows with ``start <= when <= end`` are applied; ``opening_balance``
    is assumed to already reflect everything strictly before ``start``.
    """
    by_date = {}
    for flow in flows:
        if flow.when < start or flow.when > end:
            continue
        by_date[flow.when] = by_date.get(flow.when, Decimal(0)) + flow.amount
    running = opening_balance
    points: List[Point] = []
    for when in sorted(by_date):
        running += by_date[when]
        points.append(Point(when, running))
    return points


def balance_as_of(points: Sequence[Point], opening_balance: Decimal, when: date) -> Decimal:
    """The balance at end-of-day ``when``, after that day's own flows.

    Every flow lands as one net change per calendar day (see
    :func:`build_points`), so "the balance on day X" already means
    "after day X's own transactions" everywhere else in this module. Using
    ``<=`` here (not ``<``) keeps that consistent: a day that itself carries
    a big credit is not scored by the day *before* it landed.
    """
    level = opening_balance
    for point in points:
        if point.when <= when:
            level = point.balance
        else:
            break
    return level


def min_balance_over(points: Sequence[Point], opening_balance: Decimal,
                     start: date, end: date) -> Decimal:
    """Exact minimum balance across the closed interval ``[start, end]``."""
    level = balance_as_of(points, opening_balance, start)
    values = [level] + [p.balance for p in points if start < p.when <= end]
    return min(values)


def max_safe_amount(points: Sequence[Point], opening_balance: Decimal,
                    start: date, end: date, minimum_balance: Decimal,
                    requested_amount: Decimal) -> Decimal:
    """Largest lump sum payable on ``start`` without breaching the minimum.

    Paying X on ``start`` reduces the balance by X for every date from
    ``start`` onward, so the constraint is
    ``min_balance_over(start, end) - X >= minimum_balance`` for the whole
    window - a single closed-form subtraction, not a search.
    """
    headroom = min_balance_over(points, opening_balance, start, end) - minimum_balance
    return max(Decimal(0), min(requested_amount, headroom))


def earliest_full_payment_date(points: Sequence[Point], opening_balance: Decimal,
                               start: date, end: date, minimum_balance: Decimal,
                               requested_amount: Decimal) -> Optional[date]:
    """Earliest date a single lump sum of ``requested_amount`` is safe.

    ``min_balance_over(D, end)`` is non-decreasing as ``D`` increases (the
    window only shrinks), so the first candidate date that clears the
    threshold is the true earliest one - later dates never need checking.
    Candidates are ``start`` plus every date something actually changes;
    between two flow dates the answer cannot change, so checking only those
    dates is exact, not an approximation.
    """
    threshold = minimum_balance + requested_amount
    candidates = [start] + sorted({p.when for p in points if start < p.when <= end})
    for candidate in candidates:
        if min_balance_over(points, opening_balance, candidate, end) >= threshold:
            return candidate
    return None


def is_schedule_safe(baseline_flows: Sequence[Flow], opening_balance: Decimal,
                     start: date, end: date, minimum_balance: Decimal,
                     schedule: Sequence[Tuple[date, Decimal]]) -> bool:
    """True if paying ``schedule`` on top of ``baseline_flows`` never breaches
    ``minimum_balance`` anywhere in ``[start, end]``.

    Payments outside the window are dropped rather than rejected outright:
    the 90-day forecast simply cannot make a safety claim about a date
    beyond its own horizon (see problem_statement.md, "90-Day Safety
    Check"). Eligibility rules elsewhere ensure a schedule extending past
    the horizon is excluded before this is ever called for it.
    """
    extra = [Flow(when=when, amount=-amount, label="candidate payment", category="__plan__")
            for when, amount in schedule if start <= when <= end]
    points = build_points(opening_balance, list(baseline_flows) + extra, start, end)
    return min_balance_over(points, opening_balance, start, end) >= minimum_balance
