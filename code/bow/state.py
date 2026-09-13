"""Reconstruct a user's financial position and project it forward.

The reconstruction turns a user's raw event rows into one dated ledger of
signed cash flows across the 90-day forecast window:

* **recurring streams** - every commitment :mod:`bow.recurrence` detects in
  settled history (monthly bills and payroll by day-of-month, and sub-monthly
  cadences such as a 7-day grocery run), projected at a robust amount and
  re-anchored onto any confirmed future-dated row for the same stream;
* **dated one-off flows** - pending debits to reserve, scheduled charges and
  confirmed income, each on its settlement date;
* **evidence adjustments** - interpreted messages can raise, cut, delay or end
  the salary stream, change a recurring cost, or suppress a credit that has
  not settled.

Irregular spending with no detectable cadence is *not* forecast: inventing a
figure for it would contradict problem_statement.md ("do not invent
unsupported ... expenses"). Variable income is likewise not projected.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal
from typing import Dict, List, Optional

from . import evidence as ev
from . import recurrence
from .dataset import Dataset, Event, Profile
from .evidence import Evidence, ImageAmountResolver

FORECAST_DAYS = 90

#: Statuses whose cash never moves.
DEAD_STATUSES = {"cancelled", "failed"}


@dataclass
class Config:
    """Knobs of the forecast model, with the reasoning for each default.

    These are *model-structure* choices, not per-request answers: the same
    values apply to every user and every request, seen or unseen.
    """

    #: How far back recurrence is read. ``None`` uses the user's full history;
    #: a shorter window would discard cadence evidence for no benefit, since
    #: re-anchoring always projects from the *latest* occurrence anyway.
    recurrence_window_days: Optional[int] = None

    #: Which observation of a recurring stream sets the projected amount.
    #: ``median`` is robust to a single atypical observation inside an
    #: otherwise regular stream (one bulk grocery shop inside a weekly
    #: cadence would otherwise be projected every week). Chosen by
    #: ``code/tools/calibrate.py``: lowest error of every convention tried.
    #: A confirmed newer figure (a dated future row, a payroll message) still
    #: overrides it - see :meth:`Reconstructor._project_recurring`.
    recurring_amount: str = "median"          # latest | mean | mean3 | median | max

    #: Whether a projected charge landing exactly on ``request_date`` is
    #: counted. ``current_available_balance`` is the balance *available* on
    #: that date, i.e. before that day's own standing orders clear, so a
    #: same-day recurring charge still has to be paid out of it.
    include_request_date: bool = True

    #: Minimum repeats before a sub-monthly cadence is trusted; forwarded to
    #: :mod:`bow.recurrence` so the whole model is configured in one place.
    periodic_min_repeats: int = recurrence.MIN_PERIODIC_REPEATS

    #: Whether a credit stream whose amount varies between occurrences is
    #: projected forward. Off by default: only a fixed, repeating figure is
    #: "confirmed" income in the sense problem_statement.md uses.
    project_variable_income: bool = False


@dataclass
class Flow:
    """One projected movement of cash."""

    when: date
    amount: Decimal            # signed: negative is money out
    label: str
    event_id: str = ""
    category: str = ""
    flexible_stop: bool = False
    flexible_reduce: bool = False
    min_allowed: Optional[Decimal] = None
    recurring: bool = False


@dataclass
class State:
    profile: Profile
    request_date: date
    horizon: date
    opening_balance: Decimal
    flows: List[Flow] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def sorted_flows(self) -> List[Flow]:
        return sorted(self.flows, key=lambda f: (f.when, f.event_id, f.label))


# --------------------------------------------------------------------------
# recurrence detection
# --------------------------------------------------------------------------

def _next_on_day(start: date, day: int) -> date:
    """First date on or after ``start`` falling on day-of-month ``day``."""
    year, month = start.year, start.month
    for _ in range(3):
        last = _days_in_month(year, month)
        candidate = date(year, month, min(day, last))
        if candidate >= start:
            return candidate
        month += 1
        if month > 12:
            month, year = 1, year + 1
    return start


def _days_in_month(year: int, month: int) -> int:
    if month == 12:
        return 31
    return (date(year, month + 1, 1) - timedelta(days=1)).day


def _monthly_dates(start: date, end: date, day: int) -> List[date]:
    out = []
    cursor = _next_on_day(start, day)
    while cursor <= end:
        out.append(cursor)
        year, month = cursor.year, cursor.month + 1
        if month > 12:
            month, year = 1, year + 1
        cursor = date(year, month, min(day, _days_in_month(year, month)))
    return out


# --------------------------------------------------------------------------
# reconstruction
# --------------------------------------------------------------------------

class Reconstructor:
    def __init__(self, data: Dataset, config: Optional[Config] = None,
                 images: Optional[ImageAmountResolver] = None,
                 use_cache: bool = True):
        self.data = data
        self.config = config or Config()
        self.images = images if images is not None else ImageAmountResolver(
            data.images_by_event, use_cache=use_cache
        )
        self.evidence = ev.interpret_all(data.messages_by_user, use_cache=use_cache)

    # -- amounts ----------------------------------------------------------

    def amount_in_home(self, event: Event, profile: Profile) -> Optional[Decimal]:
        """Event amount converted to the user's home currency.

        A blank amount is filled from the linked image; it is never read as zero.
        """
        amount, currency = event.amount, event.currency
        if amount is None:
            found = self.images.get(event.event_id)
            if found is None:
                raise ValueError(
                    f"{event.event_id} has a blank amount and no image evidence; "
                    "refusing to treat it as zero"
                )
            amount, currency = found
        if currency == profile.home_currency:
            return amount
        when = event.effective_date or self.data.rates and event.event_date
        return self.data.rates.convert(amount, currency, profile.home_currency, when)

    # -- evidence ---------------------------------------------------------

    def _user_evidence(self, user_id: str) -> List[Evidence]:
        return self.evidence.get(user_id, [])

    def _suppressed_events(self, items: List[Evidence]) -> set:
        """Event ids whose credit must not be counted until it settles."""
        out = set()
        for item in items:
            if item.intent in ev.UNCONFIRMED_INCOME and item.message and item.message.related_event_id:
                out.add(item.message.related_event_id)
        return out

    def _retried_debits(self, items: List[Evidence]) -> set:
        """Failed debits that the bank says will be attempted again."""
        return {
            item.message.related_event_id
            for item in items
            if item.intent == ev.DEBIT_WILL_RETRY and item.message and item.message.related_event_id
        }

    # -- main -------------------------------------------------------------

    def build(self, user_id: str, request_date: date) -> State:
        profile = self.data.profiles[user_id]
        horizon = request_date + timedelta(days=FORECAST_DAYS)
        events = self.data.events_by_user.get(user_id, [])
        items = self._user_evidence(user_id)
        suppressed = self._suppressed_events(items)
        retried = self._retried_debits(items)

        state = State(
            profile=profile,
            request_date=request_date,
            horizon=horizon,
            opening_balance=profile.current_available_balance,
        )

        history: List[Event] = []
        for event in events:
            when = event.effective_date
            if when is None:
                continue

            if event.is_non_cash or event.status == "unrealized":
                continue                                   # displayed value, not cash

            if event.status in DEAD_STATUSES:
                # A failed debit the bank will retry still has to be reserved.
                if not (event.status == "failed" and event.event_id in retried):
                    continue

            if event.event_id in suppressed:
                continue                                   # money that has not settled

            if when < request_date:
                if event.status == "settled":
                    history.append(event)
                elif event.status == "failed" and event.event_id in retried:
                    state.flows.append(Flow(
                        when=request_date,
                        amount=-self.amount_in_home(event, profile),
                        label=f"retry of failed {event.category} debit",
                        event_id=event.event_id,
                        category=event.category,
                    ))
                continue

            # dated future flow inside (or beyond) the window
            if when > horizon:
                continue
            if event.status == "pending" and event.is_credit:
                continue                                   # unsettled credit: ignore
            amount = self.amount_in_home(event, profile)
            state.flows.append(Flow(
                when=when,
                amount=amount if event.is_credit else -amount,
                label=event.description,
                event_id=event.event_id,
                category=event.category,
            ))

        explicit: Dict[tuple, List[Flow]] = collections.defaultdict(list)
        for flow in state.flows:
            direction = "credit" if flow.amount > 0 else "debit"
            explicit[(flow.category, direction)].append(flow)

        self._project_recurring(state, history, profile, request_date, horizon, explicit)
        self._continue_confirmed_salary(state, horizon)
        self._apply_evidence(state, items, profile, request_date, horizon)
        return state

    # -- salary -----------------------------------------------------------

    def _project_recurring(self, state: State, history: List[Event], profile: Profile,
                           request_date: date, horizon: date,
                           explicit: Dict[tuple, List[Flow]]) -> None:
        """Project every detected recurring stream across the forecast window.

        ``explicit`` maps ``(category, direction)`` to the dated flows already
        placed from real future-dated event rows. A stream with such a row is
        *re-anchored* onto the latest of them and takes its amount from it:
        that row is the newest record from the same source, which
        problem_statement.md's conflict-resolution order prefers, and it also
        stops one real charge being counted twice (once as itself, once as a
        projection landing on the same date).
        """
        series_list = recurrence.detect(history, request_date,
                                        self.config.recurrence_window_days,
                                        self.config.periodic_min_repeats)
        after = request_date - timedelta(days=1) if self.config.include_request_date             else request_date

        for series in series_list:
            if (series.is_credit and not self.config.project_variable_income
                    and not series.amounts_are_stable):
                # Variable income (gig payouts, commission, an irregular second
                # income) is not "confirmed salary"; projecting it would invent
                # unsupported future income.
                state.notes.append(f"unprojected_variable_income:{series.latest.event_id}")
                continue

            key = (series.category, series.direction)
            placed = explicit.get(key, [])
            if placed:
                newest = max(placed, key=lambda f: f.when)
                anchor = newest.when
                amount = abs(newest.amount)
            else:
                anchor = series.anchor
                amount = self._series_amount(series, profile)

            latest = series.latest
            signed = amount if series.is_credit else -amount
            for when in series.occurrences(anchor, after, horizon):
                state.flows.append(Flow(
                    when=when,
                    amount=signed,
                    label=latest.description,
                    event_id=latest.event_id,
                    category=series.category,
                    flexible_stop=(not series.is_credit) and latest.can_stop,
                    flexible_reduce=(not series.is_credit) and latest.can_reduce,
                    min_allowed=latest.minimum_allowed_amount,
                    recurring=True,
                ))
            state.notes.append(
                f"recurring:{latest.event_id}:{series.category}:"
                f"{series.kind}{series.period_days or ''}:{amount}"
            )

    def _continue_confirmed_salary(self, state: State, horizon: date) -> None:
        """Carry a confirmed future salary row forward monthly.

        A dated, confirmed salary (``financial_events.csv``'s "next confirmed
        salary") for a user with no established payroll history yet - a first
        job, a prorated first month - is still a monthly salary: employment
        does not end after one paycheck. With no history-based stream to
        re-anchor, the confirmed row itself anchors the cadence. Salary already
        covered by a detected stream is left alone, so nothing is doubled.
        """
        salary = [f for f in state.flows if f.category == "salary" and f.amount > 0]
        if not salary or any(f.recurring for f in salary):
            return
        anchor = max(salary, key=lambda f: f.when)
        for when in recurrence.Series("salary", "credit", "monthly", None, ()).occurrences(
                anchor.when, anchor.when, horizon):
            state.flows.append(Flow(when=when, amount=anchor.amount, label="confirmed salary",
                                    event_id=anchor.event_id, category="salary", recurring=True))
        state.notes.append(f"salary_continued_from:{anchor.event_id}")

    def _series_amount(self, series: recurrence.Series, profile: Profile) -> Decimal:
        """The stream's forward-looking amount, in the user's home currency."""
        amounts = [self.amount_in_home(event, profile) for event in series.events]
        how = self.config.recurring_amount
        if series.is_credit and len({e.description for e in series.events}) > 1:
            how = "latest"      # a handover chain: the successor's figure is the live one
        if how == "max":
            return max(amounts)
        if how == "mean":
            return sum(amounts) / Decimal(len(amounts))
        if how == "mean3":
            tail = amounts[-3:]
            return sum(tail) / Decimal(len(tail))
        if how == "median":
            ordered = sorted(amounts)
            return ordered[len(ordered) // 2]
        return amounts[-1]

    # -- evidence-driven adjustments --------------------------------------

    def _salary_flows(self, state: State) -> List[Flow]:
        return [f for f in state.flows if f.category == "salary" and f.amount > 0]

    #: Category -> a few common English/Indonesian glosses, used only to match a
    #: recurring-cost-change message to *which* recurring flow it refers to.
    #: Generic domain vocabulary, not a per-message answer table; a category
    #: with no gloss here still matches on its own name.
    _CATEGORY_GLOSSES = {
        "rent": ["rent", "sewa", "lease"],
        "housing": ["housing", "deposit", "sewa"],
        "utilities": ["utility", "utilities", "listrik", "tagihan"],
        "insurance": ["insurance", "asuransi"],
        "education": ["tuition", "course", "education", "kursus"],
    }

    def _apply_evidence(self, state: State, items: List[Evidence], profile: Profile,
                        request_date: date, horizon: date) -> None:
        """Fold each piece of interpreted evidence into the projected flows.

        Dispatch is on the *intent* plus which generic slots (amount, date,
        percent, count of amounts) the extractor filled in - not on the
        wording of any specific message - so a newly seen message that
        classifies into one of these intents is handled the same way a
        dataset message would be.
        """
        for item in items:
            intent = item.intent
            note_id = item.message.message_id if item.message else item.source_id

            if intent in ev.INERT or intent in ev.UNCONFIRMED_INCOME or intent == ev.UNKNOWN:
                continue

            if intent == ev.SALARY_STOPPED:
                state.flows = [f for f in state.flows
                               if not (f.category == "salary" and f.amount > 0)]
                state.notes.append(f"{note_id}:salary_stopped")
                continue

            if intent == ev.RECURRING_COST_CHANGE and item.percent is not None:
                factor = Decimal(1) + item.percent / Decimal(100)
                text = (item.message.message_text if item.message else "").lower()
                matched = False
                for flow in state.flows:
                    if not flow.recurring or not flow.category:
                        continue
                    glosses = self._CATEGORY_GLOSSES.get(flow.category, [flow.category])
                    if any(g in text for g in glosses) or flow.category in text:
                        flow.amount *= factor
                        matched = True
                state.notes.append(f"{note_id}:recurring_cost_x{factor}:{matched}")
                continue

            if intent == ev.OBLIGATION_CONFIRMED and item.amount is not None:
                when = item.when or request_date
                if request_date <= when <= horizon:
                    state.flows.append(Flow(
                        when=when, amount=item.amount,
                        label="confirmed obligation payment",
                        category="obligation", event_id=note_id,
                    ))
                state.notes.append(f"{note_id}:obligation")
                continue

            if intent == ev.SALARY_SCHEDULE_CHANGE and item.when is not None:
                salary = self._salary_flows(state)
                if salary:
                    first = min(salary, key=lambda f: f.when)
                    delta = item.when - first.when
                    for flow in salary:
                        flow.when = flow.when + delta
                state.notes.append(f"{note_id}:salary_moved")
                continue

            if intent == ev.SALARY_ONE_OFF:
                if len(item.amounts) >= 2:
                    # first figure is the ongoing salary, second a one-off credit
                    self._set_salary(state, item.amounts[0][1], item.when, request_date,
                                     horizon, currency=item.amounts[0][0], profile=profile)
                    salary = self._salary_flows(state)
                    if salary:
                        first = min(salary, key=lambda f: f.when)
                        state.flows.append(Flow(
                            when=first.when, amount=item.amounts[1][1],
                            label="one-time payroll adjustment",
                            category="arrears", event_id=note_id,
                        ))
                elif item.amount is not None:
                    self._set_salary(state, item.amount, item.when, request_date, horizon,
                                     currency=item.currency, profile=profile)
                state.notes.append(f"{note_id}:salary_one_off")
                continue

            if intent == ev.SALARY_CHANGE and item.amount is not None:
                # A generic linguistic cue distinguishes "this affects the next
                # cycle only" from "this is the new ongoing figure" - not a
                # lookup of this dataset's exact sentences, just the ordinary
                # words payroll correspondence uses for a single-period change.
                text = (item.message.message_text if item.message else "").lower()
                # Deliberately NOT the bare Indonesian "gaji berikutnya" ("next
                # salary/payslip") - that phrase also closes the *ongoing*-raise
                # template ("...akan terlihat pada slip gaji berikutnya", "the
                # updated amount will appear on your NEXT PAYSLIP"), so matching
                # it there would misread a permanent raise as single-cycle.
                single_cycle_markers = [
                    "temporary", "sementara", "next salary is reduced",
                    "affected pay cycle", "unpaid leave", "cuti tanpa gaji",
                    "gaji berikutnya dikurangi",
                ]
                if any(marker in text for marker in single_cycle_markers):
                    salary = self._salary_flows(state)
                    if salary:
                        first = min(salary, key=lambda f: f.when)
                        first.amount = self._converted(item.amount, item.currency,
                                                       profile, first.when)
                    state.notes.append(f"{note_id}:salary_single_cycle")
                else:
                    self._set_salary(state, item.amount, item.when, request_date, horizon,
                                     currency=item.currency, profile=profile)
                    state.notes.append(f"{note_id}:salary_ongoing")
                continue

    def _converted(self, amount: Decimal, currency: Optional[str], profile: Optional[Profile],
                   when: date) -> Decimal:
        if currency and profile and currency != profile.home_currency:
            return self.data.rates.convert(amount, currency, profile.home_currency, when)
        return amount

    def _set_salary(self, state: State, amount: Decimal, start: Optional[date],
                    request_date: date, horizon: date,
                    currency: Optional[str] = None,
                    profile: Optional[Profile] = None) -> None:
        """Replace the projected salary stream with a confirmed figure."""
        if currency and profile and currency != profile.home_currency:
            amount = self.data.rates.convert(amount, currency, profile.home_currency,
                                             start or request_date)
        salary = self._salary_flows(state)
        if start is not None:
            state.flows = [f for f in state.flows
                           if not (f.category == "salary" and f.amount > 0 and f.when >= start)]
            day = start.day
            for when in _monthly_dates(start, horizon, day):
                state.flows.append(Flow(when=when, amount=amount,
                                        label="confirmed salary", category="salary"))
        elif salary:
            for flow in salary:
                flow.amount = amount
