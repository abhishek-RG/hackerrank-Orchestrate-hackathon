"""Generic recurrence detection over a user's settled history.

The problem statement says to "distinguish recurring expenses from one-time
purchases, transfers, refunds, and unusual events" and to "detect recurrence
only when history supports it". This module is the whole of that judgement,
kept separate from cash-flow projection (:mod:`bow.state`) so it can be
tested on its own.

Two shapes of recurrence are recognised, and nothing else:

``monthly``
    The same commitment charged once a calendar month on a stable
    day-of-month - rent, utilities, insurance, subscriptions, payroll. Gaps
    between occurrences sit in the 27-32 day band precisely *because* the
    day-of-month is fixed, so the cadence is expressed as "same day next
    month", never as "+30 days" (which would drift off the 31st).

``periodic``
    A commitment repeating every ``period_days`` - a weekly grocery run, a
    5-day commute top-up, a fortnightly takeaway. Detected from the observed
    gaps only; the period is never assumed.

A series that fits neither shape is not recurring. Its history still informs
nothing about the future here - callers decide what to do with irregular
spending, and :mod:`bow.state` deliberately does *not* invent a forecast for
it (see ``problem_statement.md``: "Do not invent unsupported income,
expenses...").

Grouping has to work at two different grains, because the data uses both:

* One commitment, many descriptions. A user's transport cadence rotates
  through "Rail pass", "Parking and tolls", "Commuter pass" - grouping on
  description would shatter one real commitment into several one-off-looking
  fragments and lose the recurrence.
* One category, several independent streams. A single ``salary`` category can
  hold a monthly "Base salary" *and* a separate "Performance commission", or
  "Primary household salary" alongside "Second household income". Merging
  those by category destroys both cadences and invents a stream that never
  existed.

So detection tries the description grain first and keeps it only when those
per-description series explain most of the category's history
(:data:`DESCRIPTION_COVERAGE`); otherwise it falls back to one series for the
whole category. Events left unexplained either way are genuinely one-off (a
quarterly bonus, an arrears payment) and are deliberately not projected.
"""

from __future__ import annotations

import calendar
import collections
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

from .dataset import Event

#: Consecutive-occurrence gaps inside this band are consistent with a fixed
#: day-of-month (February through a 31-day month).
MONTHLY_GAP_MIN = 26
MONTHLY_GAP_MAX = 33

#: How far the day-of-month may wander before a stream stops looking monthly
#: (weekend/holiday shifting of an otherwise fixed charge date).
MONTHLY_DAY_TOLERANCE = 3

#: A monthly stream is credible from two occurrences: a start-of-employment
#: payroll pair is genuinely monthly and must not be discarded as noise.
MIN_MONTHLY_REPEATS = 2

#: A sub-monthly cadence needs more evidence, since three arbitrary dates can
#: look evenly spaced by accident.
MIN_PERIODIC_REPEATS = 4

#: Share of gaps that must match the candidate period (+/-1 day).
PERIODIC_GAP_AGREEMENT = 0.7

#: Share of a category's history that per-description series must explain
#: before the description grain is preferred over the category grain.
DESCRIPTION_COVERAGE = 0.6

#: Description words marking a stream's *last* occurrence (an employment
#: that has ended). Used only when that record is also the newest in its
#: category - see :func:`_resolve_handover`.
TERMINAL_MARKERS = ("final",)

#: Share of a credit stream's observations that must carry one identical
#: amount for the stream to count as confirmed (fixed) income.
STABLE_AMOUNT_SHARE = 0.5


def add_months(when: date, count: int) -> date:
    """``when`` shifted by ``count`` calendar months, clamped to month end."""
    year, month = when.year, when.month + count
    year += (month - 1) // 12
    month = (month - 1) % 12 + 1
    return date(year, month, min(when.day, calendar.monthrange(year, month)[1]))


@dataclass(frozen=True)
class Series:
    """One detected recurring stream for a single user."""

    category: str
    direction: str
    kind: str                      # "monthly" | "periodic"
    period_days: Optional[int]     # None for monthly
    events: Tuple[Event, ...]      # chronological, oldest first
    #: ``""`` when the series spans a whole category rather than one
    #: description - i.e. the cadence belongs to the category, not to a label.
    description: str = ""

    @property
    def amounts_are_stable(self) -> bool:
        """True when one figure accounts for most observations.

        Distinguishes a *confirmed* commitment (payroll at a fixed figure that
        may have changed once, rent on a standing order) from an inherently
        variable stream (gig payouts, sales commission, a household's irregular
        second income), where nearly every observation differs.
        problem_statement.md counts "confirmed salary" but says not to count
        bonuses or commissions before they settle, nor to invent unsupported
        future income - so the caller uses this to decide whether a credit
        stream may be projected at all.
        """
        amounts = [event.amount for event in self.events]
        if None in amounts:
            amounts = [a for a in amounts if a is not None]
        if not amounts:
            return False
        _, modal_count = collections.Counter(amounts).most_common(1)[0]
        return modal_count >= 2 and modal_count >= STABLE_AMOUNT_SHARE * len(amounts)

    @property
    def is_credit(self) -> bool:
        return self.direction == "credit"

    @property
    def anchor(self) -> date:
        """Date of the most recent observed occurrence."""
        return self.events[-1].effective_date

    @property
    def latest(self) -> Event:
        return self.events[-1]

    def occurrences(self, anchor: date, after: date, until: date) -> List[date]:
        """Projected dates in ``(after, until]``, stepping on from ``anchor``.

        ``anchor`` is passed in rather than read off ``self`` so a caller can
        re-anchor the stream onto a *confirmed* future occurrence (a dated
        payroll row, an amendment in a message) and have the rest of the
        cadence follow from there.
        """
        out: List[date] = []
        if self.kind == "monthly":
            step = 1
            while True:
                when = add_months(anchor, step)
                if when > until:
                    break
                if when > after:
                    out.append(when)
                step += 1
        else:
            when = anchor
            while True:
                when = when + timedelta(days=self.period_days)
                if when > until:
                    break
                if when > after:
                    out.append(when)
        return out


def _looks_monthly(dates: Sequence[date]) -> bool:
    gaps = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]
    if not gaps or not all(MONTHLY_GAP_MIN <= gap <= MONTHLY_GAP_MAX for gap in gaps):
        return False
    days = [d.day for d in dates]
    # A charge on the 31st lands on the 28th in February; compare against the
    # clamped day-of-month rather than rejecting the stream.
    if max(days) - min(days) <= MONTHLY_DAY_TOLERANCE:
        return True
    return all(day >= 28 for day in days)


def _period_of(dates: Sequence[date]) -> Optional[int]:
    """Modal gap, when it explains most of the gaps; otherwise ``None``."""
    gaps = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]
    if not gaps:
        return None
    counts = collections.Counter(gaps)
    period, _ = counts.most_common(1)[0]
    if period < 1:
        return None
    agree = sum(1 for gap in gaps if abs(gap - period) <= 1)
    if agree / len(gaps) < PERIODIC_GAP_AGREEMENT:
        return None
    return period


def _fit(events: Sequence[Event], periodic_min_repeats: int,
         description: str) -> Optional[Series]:
    """The series ``events`` forms, or ``None`` if they are not recurring."""
    if not events:
        return None
    ordered = sorted(events, key=lambda e: (e.effective_date, e.event_id))
    dates = [event.effective_date for event in ordered]
    category, direction = ordered[0].category, ordered[0].direction
    if len(dates) >= MIN_MONTHLY_REPEATS and _looks_monthly(dates):
        return Series(category, direction, "monthly", None, tuple(ordered), description)
    if len(dates) >= periodic_min_repeats:
        period = _period_of(dates)
        if period is not None:
            return Series(category, direction, "periodic", period, tuple(ordered), description)
    return None


def detect(events: Sequence[Event], as_of: date,
           window_days: Optional[int] = None,
           periodic_min_repeats: int = MIN_PERIODIC_REPEATS) -> List[Series]:
    """Recurring streams visible in ``events`` strictly before ``as_of``.

    Only settled cash movements are considered: a pending, failed, cancelled
    or non-cash row says nothing about an established pattern (it is handled
    individually as a dated flow instead). ``window_days`` bounds how far back
    the pattern is read from; ``None`` uses all available history.
    """
    start = as_of - timedelta(days=window_days) if window_days else date.min
    grouped: Dict[Tuple[str, str], List[Event]] = collections.defaultdict(list)
    for event in events:
        when = event.effective_date
        if when is None or when >= as_of or when < start:
            continue
        if event.status != "settled" or event.is_non_cash:
            continue
        grouped[(event.category, event.direction)].append(event)

    series: List[Series] = []
    for key, group in sorted(grouped.items()):
        handover = _resolve_handover(group)
        if handover is not None:
            if handover != "ended":
                series.append(handover)
            continue

        by_description: Dict[str, List[Event]] = collections.defaultdict(list)
        for event in group:
            by_description[event.description].append(event)

        fine = [fit for description, events_for in sorted(by_description.items())
                for fit in [_fit(events_for, periodic_min_repeats, description)] if fit]
        covered = sum(len(fit.events) for fit in fine)
        if covered >= DESCRIPTION_COVERAGE * len(group):
            series.extend(fine)
            continue

        coarse = _fit(group, periodic_min_repeats, "")
        if coarse is not None:
            series.append(coarse)
    return series


def _resolve_handover(group: Sequence[Event]):
    """Detect one stream handing over to (or ending with) a newer record.

    The newest record in a category can carry a different description from
    the stream before it while landing exactly where that stream's next
    occurrence was due: "Previous employer payroll" then "New employer
    payroll", "Payroll before leave" then "Payroll after returning from
    leave", "Payroll credit" then "Final employer payroll". Read by
    description alone, the newest record looks like a one-off and the old
    stream looks alive - the wrong way round on both counts.

    Following problem_statement.md's conflict order (a newer record from the
    same source wins), such a handover is resolved in favour of the newest
    record:

    * returns ``"ended"`` when that record is marked terminal - nothing is
      projected, because no further income is supported;
    * returns one monthly :class:`Series` continuing the combined stream at
      the newest record's figure when it is a successor;
    * returns ``None`` when there is no handover, leaving normal detection.
    """
    ordered = sorted(group, key=lambda e: (e.effective_date, e.event_id))
    if len(ordered) < 2 or ordered[-1].direction != "credit":
        return None
    newest = ordered[-1]
    previous = [e for e in ordered[:-1] if e.description != newest.description]
    if not previous or any(e.description == newest.description for e in ordered[:-1]):
        return None
    predecessor_desc = previous[-1].description
    stream = [e for e in ordered if e.description == predecessor_desc]
    if len(stream) < MIN_MONTHLY_REPEATS or not _looks_monthly(
            [e.effective_date for e in stream]):
        return None
    # The newest record must sit on the stream's cadence (a whole number of
    # months after its last occurrence, allowing for a leave gap), not be an
    # off-cycle extra such as a mid-month bonus or arrears payment.
    last = stream[-1].effective_date
    if abs(newest.effective_date.day - last.day) > MONTHLY_DAY_TOLERANCE:
        return None
    if newest.effective_date <= last:
        return None
    if any(marker in newest.description.lower() for marker in TERMINAL_MARKERS):
        return "ended"
    chain = tuple(stream) + (newest,)
    return Series(newest.category, newest.direction, "monthly", None, chain, newest.description)
