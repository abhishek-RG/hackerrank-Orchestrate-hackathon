"""CSV row formatting and writing.

Formatting is centralized here so the same rules (plain decimal strings, no
thousands separators, ``none`` for empty plans, ``YYYY-MM-DD`` dates) apply
identically whether a row is written by the real run or by the regression
evaluator.
"""

from __future__ import annotations

import csv
from datetime import date
from decimal import Decimal
from typing import List, Optional, Tuple

REQUIRED_COLUMNS = [
    "request_id", "amount_safe_to_pay", "affordability_status",
    "recommended_payment_method", "payment_plan", "earliest_date_for_full_payment",
    "spending_changes_needed", "decision_explanation",
]


def fmt_amount(amount: Decimal) -> str:
    """Plain decimal string, no scientific notation, no trailing zeros.

    ``Decimal.normalize()`` collapses trailing zeros by adjusting the
    exponent (``Decimal("12693000").normalize()`` becomes
    ``Decimal("1.2693E+7")``), and ``str()`` on that renders in scientific
    notation - exactly the wrong output for a CSV amount. Formatting with the
    explicit ``f`` spec keeps it a plain decimal regardless of exponent.
    """
    quantized = amount.quantize(Decimal("0.01"))
    text = format(quantized, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text if text else "0"


def fmt_money(amount: Decimal) -> str:
    """Amount inside ``payment_plan`` / ``reduce_to``: whole numbers stay
    integral (``25256``), anything with cents keeps two decimals (``620.40``),
    matching the published sample rows."""
    quantized = amount.quantize(Decimal("0.01"))
    if quantized == quantized.to_integral_value():
        return format(quantized.to_integral_value(), "f")
    return format(quantized, "f")


def fmt_date(d: Optional[date]) -> str:
    return d.isoformat() if d else ""


def fmt_plan(schedule: List[Tuple[date, Decimal]]) -> str:
    if not schedule:
        return "none"
    return "|".join(f"{d.isoformat()}:{fmt_money(a)}" for d, a in schedule)


def write_csv(path: str, rows: List[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in REQUIRED_COLUMNS})
