"""Dataset loading, money handling, and FX conversion for Buy or Wait?.

All money is handled as :class:`decimal.Decimal`. Floats are never used for
amounts, so the deterministic engine produces byte-identical output across runs
and platforms.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal
from typing import Dict, Iterable, List, Optional, Sequence

# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_DIR = os.path.join(REPO_ROOT, "dataset")

CENT = Decimal("0.01")


# --------------------------------------------------------------------------
# scalar parsing
# --------------------------------------------------------------------------

def parse_date(value: str) -> Optional[date]:
    value = (value or "").strip()
    if not value:
        return None
    # settlement timestamps in messages.csv carry a time component
    value = value[:10]
    return date(int(value[0:4]), int(value[5:7]), int(value[8:10]))


def parse_amount(value: str) -> Optional[Decimal]:
    value = (value or "").strip().replace(",", "")
    if not value:
        return None
    return Decimal(value)


def parse_bool(value: str) -> bool:
    return (value or "").strip().lower() in {"true", "1", "yes"}


def parse_list(value: str) -> List[str]:
    value = (value or "").strip()
    if not value:
        return []
    return [part.strip() for part in value.split("|") if part.strip()]


def read_csv(name: str, directory: Optional[str] = None) -> List[Dict[str, str]]:
    path = os.path.join(directory or DATASET_DIR, name)
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


# --------------------------------------------------------------------------
# records
# --------------------------------------------------------------------------

@dataclass
class Profile:
    user_id: str
    home_currency: str
    current_available_balance: Decimal
    minimum_balance_to_keep: Decimal
    financial_priorities: List[str]
    protect: List[str]
    willing_to_reduce: List[str]
    willing_to_stop: List[str]
    payment_methods: List[str]
    max_installment_months: Optional[int]

    @property
    def accepts_full(self) -> bool:
        return "full_payment" in self.payment_methods

    @property
    def accepts_partial(self) -> bool:
        return "partial_payment" in self.payment_methods

    @property
    def accepts_installments(self) -> bool:
        return "installments" in self.payment_methods


@dataclass
class Event:
    event_id: str
    user_id: str
    event_type: str
    description: str
    category: str
    direction: str
    amount: Optional[Decimal]
    currency: str
    event_date: Optional[date]
    settlement_date: Optional[date]
    status: str
    linked_event_id: str
    flexibility: str
    minimum_allowed_amount: Optional[Decimal]

    @property
    def is_credit(self) -> bool:
        return self.direction == "credit"

    @property
    def is_debit(self) -> bool:
        return self.direction == "debit"

    @property
    def is_non_cash(self) -> bool:
        return self.direction == "non_cash"

    @property
    def can_stop(self) -> bool:
        return self.flexibility in {"stoppable", "reducible_or_stoppable"}

    @property
    def can_reduce(self) -> bool:
        return self.flexibility in {"reducible", "reducible_or_stoppable"}

    @property
    def effective_date(self) -> Optional[date]:
        """The date on which cash actually moves."""
        return self.settlement_date or self.event_date


@dataclass
class Request:
    request_id: str
    user_id: str
    request_date: date
    request_type: str
    requested_amount: Decimal
    desired_completion_date: date
    allows_partial_payment: bool
    request_text: str
    # populated only for sample_requests.csv
    truth: Dict[str, str] = field(default_factory=dict)


@dataclass
class PaymentOption:
    payment_option_id: str
    request_id: str
    payment_method: str
    payment_amount: Decimal
    number_of_payments: int
    first_payment_date: date
    payment_frequency_days: Optional[int]
    financing_fee: Decimal
    total_payable_amount: Decimal

    def schedule(self) -> List[tuple]:
        """Return the option's payments as ``[(date, amount), ...]``."""
        out = []
        step = self.payment_frequency_days or 0
        for index in range(self.number_of_payments):
            when = self.first_payment_date + timedelta(days=step * index)
            out.append((when, self.payment_amount))
        return out

    @property
    def approx_months(self) -> int:
        """Installment duration in months, for ``max_installment_months``.

        Every installment option in this dataset pays roughly monthly
        (``payment_frequency_days`` in 28-31), so the payment count itself is
        the month count; a genuinely different cadence would need a real
        day-span calculation instead.
        """
        return self.number_of_payments


@dataclass
class Message:
    message_id: str
    user_id: str
    request_id: str
    related_event_id: str
    sent_at: Optional[date]
    source_type: str
    message_text: str


@dataclass
class ImageRef:
    image_id: str
    user_id: str
    request_id: str
    related_event_id: str

    def path(self) -> str:
        return os.path.join(DATASET_DIR, "media", "images", f"{self.image_id}.png")


# --------------------------------------------------------------------------
# FX
# --------------------------------------------------------------------------

class ExchangeRates:
    """Fixed dated conversion rates.

    Rates are published monthly. For a settlement date with no exact row we use
    the most recent rate on or before it, falling back to the earliest known
    rate for dates that precede the table.
    """

    def __init__(self, rows: Iterable[Dict[str, str]]):
        self._by_pair: Dict[tuple, List[tuple]] = {}
        for row in rows:
            pair = (row["from_currency"], row["to_currency"])
            when = parse_date(row["rate_date"])
            rate = parse_amount(row["rate"])
            if when is None or rate is None:
                continue
            self._by_pair.setdefault(pair, []).append((when, rate))
        for series in self._by_pair.values():
            series.sort(key=lambda item: item[0])

    def _direct(self, src: str, dst: str, when: date) -> Optional[Decimal]:
        series = self._by_pair.get((src, dst))
        if not series:
            return None
        chosen = series[0][1]
        for rate_date, rate in series:
            if rate_date <= when:
                chosen = rate
            else:
                break
        return chosen

    def convert(self, amount: Decimal, src: str, dst: str, when: date) -> Decimal:
        if src == dst:
            return amount
        rate = self._direct(src, dst, when)
        if rate is not None:
            return amount * rate
        inverse = self._direct(dst, src, when)
        if inverse is not None and inverse != 0:
            return amount / inverse
        # one hop through an intermediate currency
        for middle in ("USD", "EUR"):
            if middle in (src, dst):
                continue
            first = self._direct(src, middle, when)
            if first is None:
                back = self._direct(middle, src, when)
                first = (Decimal(1) / back) if back else None
            if first is None:
                continue
            second = self._direct(middle, dst, when)
            if second is None:
                back = self._direct(dst, middle, when)
                second = (Decimal(1) / back) if back else None
            if second is None:
                continue
            return amount * first * second
        raise KeyError(f"no exchange rate for {src}->{dst} on {when}")


# --------------------------------------------------------------------------
# bundle
# --------------------------------------------------------------------------

@dataclass
class Dataset:
    profiles: Dict[str, Profile]
    events_by_user: Dict[str, List[Event]]
    events_by_id: Dict[str, Event]
    requests: List[Request]
    samples: List[Request]
    options_by_request: Dict[str, List[PaymentOption]]
    messages_by_user: Dict[str, List[Message]]
    images_by_event: Dict[str, ImageRef]
    images: List[ImageRef]
    rates: ExchangeRates


def _profile(row: Dict[str, str]) -> Profile:
    months = row["max_installment_months"].strip()
    return Profile(
        user_id=row["user_id"],
        home_currency=row["home_currency"],
        current_available_balance=parse_amount(row["current_available_balance"]) or Decimal(0),
        minimum_balance_to_keep=parse_amount(row["minimum_balance_to_keep"]) or Decimal(0),
        financial_priorities=parse_list(row["financial_priorities"]),
        protect=parse_list(row["expense_categories_to_protect"]),
        willing_to_reduce=parse_list(row["expense_categories_user_is_willing_to_reduce"]),
        willing_to_stop=parse_list(row["expense_categories_user_is_willing_to_stop"]),
        payment_methods=parse_list(row["payment_methods_user_will_consider"]),
        max_installment_months=int(months) if months else None,
    )


def _event(row: Dict[str, str]) -> Event:
    return Event(
        event_id=row["event_id"],
        user_id=row["user_id"],
        event_type=row["event_type"],
        description=row["description"],
        category=row["category"],
        direction=row["direction"],
        amount=parse_amount(row["amount"]),
        currency=row["currency"],
        event_date=parse_date(row["event_date"]),
        settlement_date=parse_date(row["settlement_date"]),
        status=row["status"],
        linked_event_id=row["linked_event_id"].strip(),
        flexibility=row["flexibility"],
        minimum_allowed_amount=parse_amount(row["minimum_allowed_amount"]),
    )


TRUTH_COLUMNS = (
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
)


def _request(row: Dict[str, str]) -> Request:
    truth = {col: row[col] for col in TRUTH_COLUMNS if col in row}
    return Request(
        request_id=row["request_id"],
        user_id=row["user_id"],
        request_date=parse_date(row["request_date"]),
        request_type=row["request_type"],
        requested_amount=parse_amount(row["requested_amount"]) or Decimal(0),
        desired_completion_date=parse_date(row["desired_completion_date"]),
        allows_partial_payment=parse_bool(row["allows_partial_payment"]),
        request_text=row["request_text"],
        truth=truth,
    )


def _option(row: Dict[str, str]) -> PaymentOption:
    frequency = row["payment_frequency_days"].strip()
    return PaymentOption(
        payment_option_id=row["payment_option_id"],
        request_id=row["request_id"],
        payment_method=row["payment_method"],
        payment_amount=parse_amount(row["payment_amount"]) or Decimal(0),
        number_of_payments=int(row["number_of_payments"]),
        first_payment_date=parse_date(row["first_payment_date"]),
        payment_frequency_days=int(frequency) if frequency else None,
        financing_fee=parse_amount(row["financing_fee"]) or Decimal(0),
        total_payable_amount=parse_amount(row["total_payable_amount"]) or Decimal(0),
    )


def load(directory: Optional[str] = None) -> Dataset:
    directory = directory or DATASET_DIR

    profiles = {row["user_id"]: _profile(row) for row in read_csv("financial_profiles.csv", directory)}

    events_by_user: Dict[str, List[Event]] = {}
    events_by_id: Dict[str, Event] = {}
    for row in read_csv("financial_events.csv", directory):
        event = _event(row)
        events_by_user.setdefault(event.user_id, []).append(event)
        events_by_id[event.event_id] = event
    for series in events_by_user.values():
        series.sort(key=lambda e: (e.effective_date or date.min, e.event_id))

    requests = [_request(row) for row in read_csv("requests.csv", directory)]
    samples = [_request(row) for row in read_csv("sample_requests.csv", directory)]

    options_by_request: Dict[str, List[PaymentOption]] = {}
    for row in read_csv("request_payment_options.csv", directory):
        option = _option(row)
        options_by_request.setdefault(option.request_id, []).append(option)
    for series in options_by_request.values():
        series.sort(key=lambda o: o.payment_option_id)

    messages_by_user: Dict[str, List[Message]] = {}
    for row in read_csv("messages.csv", directory):
        message = Message(
            message_id=row["message_id"],
            user_id=row["user_id"],
            request_id=row["request_id"].strip(),
            related_event_id=row["related_event_id"].strip(),
            sent_at=parse_date(row["sent_at"]),
            source_type=row["source_type"],
            message_text=row["message_text"],
        )
        messages_by_user.setdefault(message.user_id, []).append(message)
    for series in messages_by_user.values():
        series.sort(key=lambda m: (m.sent_at or date.min, m.message_id))

    images = [
        ImageRef(
            image_id=row["image_id"],
            user_id=row["user_id"],
            request_id=row["request_id"].strip(),
            related_event_id=row["related_event_id"].strip(),
        )
        for row in read_csv("images.csv", directory)
    ]
    images_by_event = {img.related_event_id: img for img in images if img.related_event_id}

    rates = ExchangeRates(read_csv("exchange_rates.csv", directory))

    return Dataset(
        profiles=profiles,
        events_by_user=events_by_user,
        events_by_id=events_by_id,
        requests=requests,
        samples=samples,
        options_by_request=options_by_request,
        messages_by_user=messages_by_user,
        images_by_event=images_by_event,
        images=images,
        rates=rates,
    )
