"""Candidate payment plans: what could be recommended, and is it safe.

A candidate is one concrete way to settle a request: pay it all today, split
it into two payments, follow one of the seller's installment offers, or wait
for a later date when a lump sum becomes safe. This module builds every
candidate the rules allow, tests each one against the 90-day forecast
(:mod:`forecast`), and records what it would take to make an otherwise-unsafe
candidate safe (which flexible expenses would need to stop or shrink).

Nothing here picks a winner - that is :mod:`rank`'s job. This module only
answers "what are the safe, rule-eligible options" for one request.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

from . import forecast
from .dataset import PaymentOption, Profile, Request
from .state import Flow, State

MAX_SPENDING_CHANGES = 3


@dataclass
class SpendingChange:
    op: str            # "stop" | "reduce"
    event_id: str
    new_amount: Optional[Decimal] = None   # only for "reduce"

    def as_text(self) -> str:
        if self.op == "stop":
            return f"stop:{self.event_id}"
        from .output import fmt_money
        return f"reduce_to:{self.event_id}:{fmt_money(self.new_amount)}"


def fmt_plain(amount: Decimal) -> str:
    """Plain decimal string; see output.fmt_amount for why not .normalize()."""
    text = format(amount.quantize(Decimal("0.01")), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text if text else "0"


@dataclass
class Candidate:
    method: str                                   # full_payment | partial_payment | installments | wait
    schedule: List[Tuple[date, Decimal]]
    payment_option_id: Optional[str] = None
    changes: List[SpendingChange] = field(default_factory=list)

    @property
    def total_paid(self) -> Decimal:
        return sum((amt for _, amt in self.schedule), Decimal(0))

    @property
    def starts(self) -> date:
        return self.schedule[0][0]

    @property
    def num_payments(self) -> int:
        return len(self.schedule)


# --------------------------------------------------------------------------
# flexible-event discovery (for spending-change search)
# --------------------------------------------------------------------------

@dataclass
class FlexibleEvent:
    event_id: str
    category: str
    occurrences: List[Flow]
    can_stop: bool
    can_reduce: bool
    min_allowed: Optional[Decimal]


def find_flexible_events(state: State, profile: Profile) -> Dict[str, FlexibleEvent]:
    """Recurring flows the user has explicitly said they'll stop or reduce.

    Three conditions must all hold for a given event: the event's own
    ``flexibility`` tag allows it, the user's own willingness lists name that
    *category*, and the category is not in their protected list. A reduce
    target only exists when the event itself supplies
    ``minimum_allowed_amount`` - nothing here invents a reduction figure.
    """
    groups: Dict[str, FlexibleEvent] = {}
    for flow in state.flows:
        if not flow.event_id or not flow.recurring or flow.category in profile.protect:
            continue
        can_stop = flow.flexible_stop and flow.category in profile.willing_to_stop
        can_reduce = (flow.flexible_reduce and flow.category in profile.willing_to_reduce
                     and flow.min_allowed is not None)
        if not can_stop and not can_reduce:
            continue
        entry = groups.setdefault(flow.event_id, FlexibleEvent(
            event_id=flow.event_id, category=flow.category, occurrences=[],
            can_stop=can_stop, can_reduce=can_reduce, min_allowed=flow.min_allowed,
        ))
        entry.occurrences.append(flow)
    return groups


def change_subsets(flexible: Dict[str, FlexibleEvent], max_size: int = MAX_SPENDING_CHANGES):
    """Yield every subset (as a list of :class:`SpendingChange`) up to
    ``max_size`` distinct events, branching over each event's available
    actions. Always yields the empty subset first.
    """
    yield []
    event_ids = sorted(flexible)
    for size in range(1, min(max_size, len(event_ids)) + 1):
        for combo in itertools.combinations(event_ids, size):
            option_lists = []
            for eid in combo:
                fe = flexible[eid]
                opts = []
                if fe.can_stop:
                    opts.append(SpendingChange("stop", eid))
                if fe.can_reduce:
                    opts.append(SpendingChange("reduce", eid, fe.min_allowed))
                option_lists.append(opts)
            for combo_changes in itertools.product(*option_lists):
                yield list(combo_changes)


def apply_changes(flows: List[Flow], changes: List[SpendingChange]) -> List[Flow]:
    if not changes:
        return flows
    by_event = {c.event_id: c for c in changes}
    out: List[Flow] = []
    for flow in flows:
        change = by_event.get(flow.event_id) if flow.event_id else None
        if change is None:
            out.append(flow)
        elif change.op == "stop":
            continue
        else:
            out.append(Flow(when=flow.when, amount=-change.new_amount, label=flow.label,
                            event_id=flow.event_id, category=flow.category,
                            recurring=flow.recurring))
    return out


# --------------------------------------------------------------------------
# candidate construction
# --------------------------------------------------------------------------

def build_candidates(profile: Profile, request: Request, state: State,
                     amount_safe_to_pay: Decimal,
                     earliest_full_date: Optional[date],
                     options: List[PaymentOption]) -> List[Candidate]:
    """Every safe, rule-eligible candidate for this request.

    ``amount_safe_to_pay`` and ``earliest_full_date`` are the request's fixed
    baseline facts (computed once, before any spending change - see
    problem_statement.md). They fix *what* a partial-payment schedule looks
    like; spending changes can only change whether a schedule tests as safe,
    never the schedule's own numbers.
    """
    horizon = state.horizon
    minimum = profile.minimum_balance_to_keep
    requested = request.requested_amount
    flexible = find_flexible_events(state, profile)

    fixed_schedules: List[Tuple[str, List[Tuple[date, Decimal]], Optional[str]]] = []

    full_option = next((o for o in options if o.payment_method == "full_payment"), None)
    if full_option is not None:
        # Use the option's own date (respecting the supplied schedule) but the
        # request's own requested_amount for the figure: "full payment" means
        # paying the complete request by definition, and every full_payment
        # option in the dataset already has payment_amount == requested_amount
        # (verified), so this is a no-op there - it only matters for a caller
        # (e.g. the what-if dashboard) that varies requested_amount on the fly.
        fixed_schedules.append(("full_payment", [(full_option.first_payment_date, requested)],
                                full_option.payment_option_id))
    else:
        fixed_schedules.append(("full_payment", [(request.request_date, requested)], None))

    if (request.allows_partial_payment and profile.accepts_partial
            and amount_safe_to_pay > 0 and amount_safe_to_pay < requested
            and earliest_full_date is not None and earliest_full_date <= request.desired_completion_date):
        fixed_schedules.append(("partial_payment", [
            (request.request_date, amount_safe_to_pay),
            (earliest_full_date, requested - amount_safe_to_pay),
        ], None))

    if profile.accepts_installments and profile.max_installment_months is not None:
        for option in options:
            if option.payment_method != "installments":
                continue
            if option.approx_months > profile.max_installment_months:
                continue
            schedule = option.schedule()
            if schedule[-1][0] > request.desired_completion_date:
                continue
            fixed_schedules.append(("installments", schedule, option.payment_option_id))

    def _method_accepted(method: str) -> bool:
        return {
            "full_payment": profile.accepts_full,
            "partial_payment": True,   # already gated above
            "installments": True,      # already gated above
        }[method]

    candidates: List[Candidate] = []
    for method, schedule, option_id in fixed_schedules:
        if not _method_accepted(method):
            continue
        for changes in change_subsets(flexible):
            trial_flows = apply_changes(state.flows, changes)
            if forecast.is_schedule_safe(trial_flows, state.opening_balance,
                                         request.request_date, horizon, minimum, schedule):
                candidates.append(Candidate(method=method, schedule=schedule,
                                            payment_option_id=option_id, changes=changes))
                break   # smallest safe subset for this schedule; larger ones add nothing

    if (profile.accepts_full and earliest_full_date is not None
            and earliest_full_date > request.request_date
            and earliest_full_date <= request.desired_completion_date):
        candidates.append(Candidate(method="wait",
                                    schedule=[(earliest_full_date, requested)]))

    return candidates
