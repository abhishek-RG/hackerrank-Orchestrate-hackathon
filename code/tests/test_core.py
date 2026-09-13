"""Unit tests for the deterministic core (stdlib ``unittest`` only).

Run from the repository root::

    python -m unittest discover -s code/tests -v

These are synthetic, hand-built cases - none of them read ``dataset/`` - so
they exercise the rules themselves rather than any particular request.
"""

from __future__ import annotations

import os
import sys
import unittest
from datetime import date, timedelta
from decimal import Decimal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bow import forecast, rank, recurrence                      # noqa: E402
from bow.candidates import Candidate, SpendingChange             # noqa: E402
from bow.dataset import Event, ExchangeRates                     # noqa: E402
from bow.output import fmt_amount, fmt_money, fmt_plan           # noqa: E402
from bow.state import Flow                                       # noqa: E402

D = Decimal


def ev(event_id, when, amount, category="rent", direction="debit",
       description="Rent", status="settled", flexibility="fixed"):
    return Event(event_id=event_id, user_id="u", event_type="expense",
                 description=description, category=category, direction=direction,
                 amount=None if amount is None else D(str(amount)), currency="EUR",
                 event_date=when, settlement_date=when, status=status,
                 linked_event_id="", flexibility=flexibility, minimum_allowed_amount=None)


class RecurrenceTests(unittest.TestCase):
    def test_monthly_by_day_of_month(self):
        events = [ev(f"e{i}", date(2025, m, 3), 500) for i, m in enumerate(range(1, 6))]
        series = recurrence.detect(events, date(2025, 6, 1))
        self.assertEqual(len(series), 1)
        self.assertEqual(series[0].kind, "monthly")
        dates = series[0].occurrences(series[0].anchor, date(2025, 6, 1), date(2025, 8, 30))
        self.assertEqual(dates, [date(2025, 6, 3), date(2025, 7, 3), date(2025, 8, 3)])

    def test_month_end_clamps(self):
        self.assertEqual(recurrence.add_months(date(2025, 1, 31), 1), date(2025, 2, 28))

    def test_periodic_cadence_across_varying_descriptions(self):
        names = ["Rail pass", "Local taxi", "Fuel refill"]
        start = date(2025, 1, 2)
        events = [ev(f"t{i}", start + timedelta(days=7 * i), 20 + i, category="transport",
                     description=names[i % 3]) for i in range(10)]
        series = recurrence.detect(events, date(2025, 3, 20))
        self.assertEqual([(s.kind, s.period_days) for s in series], [("periodic", 7)])

    def test_one_off_is_not_recurring(self):
        events = [ev("a", date(2025, 1, 5), 900, category="shopping"),
                  ev("b", date(2025, 2, 19), 120, category="shopping")]
        self.assertEqual(recurrence.detect(events, date(2025, 3, 1)), [])

    def test_non_settled_rows_ignored(self):
        events = [ev(f"e{m}", date(2025, m, 3), 500, status="cancelled") for m in range(1, 6)]
        self.assertEqual(recurrence.detect(events, date(2025, 6, 1)), [])

    def test_bonus_does_not_break_payroll_stream(self):
        events = [ev(f"p{m}", date(2025, m, 15), 3000, category="salary", direction="credit",
                     description="Payroll credit") for m in range(1, 6)]
        events.append(ev("bonus", date(2025, 3, 22), 900, category="salary",
                         direction="credit", description="Quarterly performance bonus"))
        series = recurrence.detect(events, date(2025, 6, 1))
        self.assertEqual(len(series), 1)
        self.assertEqual(series[0].description, "Payroll credit")
        self.assertTrue(series[0].amounts_are_stable)

    def test_variable_income_is_not_stable(self):
        events = [ev(f"g{i}", date(2025, 1, 1) + timedelta(days=7 * i), 400 + 37 * i,
                     category="salary", direction="credit", description="Weekly app earnings")
                  for i in range(8)]
        series = recurrence.detect(events, date(2025, 3, 1))
        self.assertEqual(len(series), 1)
        self.assertFalse(series[0].amounts_are_stable)

    def test_final_payroll_ends_stream(self):
        events = [ev(f"p{m}", date(2025, m, 15), 3000, category="salary", direction="credit",
                     description="Payroll credit") for m in range(1, 5)]
        events.append(ev("f", date(2025, 5, 15), 3000, category="salary", direction="credit",
                         description="Final employer payroll"))
        self.assertEqual(recurrence.detect(events, date(2025, 6, 1)), [])

    def test_new_employer_continues_stream(self):
        events = [ev(f"p{m}", date(2025, m, 15), 3000, category="salary", direction="credit",
                     description="Previous employer payroll") for m in range(1, 5)]
        events.append(ev("n", date(2025, 5, 15), 3200, category="salary", direction="credit",
                         description="New employer payroll"))
        series = recurrence.detect(events, date(2025, 6, 1))
        self.assertEqual(len(series), 1)
        self.assertEqual(series[0].latest.event_id, "n")


class FxTests(unittest.TestCase):
    def setUp(self):
        self.rates = ExchangeRates([
            {"rate_date": "2025-01-01", "from_currency": "USD", "to_currency": "EUR", "rate": "0.9"},
            {"rate_date": "2025-02-01", "from_currency": "USD", "to_currency": "EUR", "rate": "0.8"},
        ])

    def test_same_currency_is_identity(self):
        self.assertEqual(self.rates.convert(D("10"), "EUR", "EUR", date(2025, 1, 5)), D("10"))

    def test_uses_dated_rate(self):
        self.assertEqual(self.rates.convert(D("100"), "USD", "EUR", date(2025, 1, 20)), D("90.0"))
        self.assertEqual(self.rates.convert(D("100"), "USD", "EUR", date(2025, 2, 1)), D("80.0"))

    def test_missing_rate_raises_instead_of_guessing(self):
        with self.assertRaises(KeyError):
            self.rates.convert(D("1"), "ZAR", "IDR", date(2025, 1, 5))

    def test_inverse_direction(self):
        self.assertEqual(self.rates.convert(D("80"), "EUR", "USD", date(2025, 2, 3)), D("100"))


class ForecastTests(unittest.TestCase):
    def setUp(self):
        self.start = date(2025, 1, 1)
        self.end = self.start + timedelta(days=90)
        self.flows = [Flow(date(2025, 1, 10), D("-300"), "rent"),
                      Flow(date(2025, 1, 15), D("1000"), "salary")]

    def points(self):
        return forecast.build_points(D("1000"), self.flows, self.start, self.end)

    def test_amount_safe_is_minimum_headroom(self):
        safe = forecast.max_safe_amount(self.points(), D("1000"), self.start, self.end,
                                        D("200"), D("5000"))
        self.assertEqual(safe, D("500"))          # low point 700 - minimum 200

    def test_amount_safe_capped_and_floored(self):
        self.assertEqual(forecast.max_safe_amount(self.points(), D("1000"), self.start,
                                                  self.end, D("200"), D("50")), D("50"))
        self.assertEqual(forecast.max_safe_amount(self.points(), D("1000"), self.start,
                                                  self.end, D("900"), D("50")), D("0"))

    def test_earliest_full_payment_after_salary(self):
        when = forecast.earliest_full_payment_date(self.points(), D("1000"), self.start,
                                                   self.end, D("200"), D("1200"))
        self.assertEqual(when, date(2025, 1, 15))

    def test_schedule_safety_boundary(self):
        ok = forecast.is_schedule_safe(self.flows, D("1000"), self.start, self.end, D("200"),
                                       [(self.start, D("500"))])
        breach = forecast.is_schedule_safe(self.flows, D("1000"), self.start, self.end, D("200"),
                                           [(self.start, D("500.01"))])
        self.assertTrue(ok)
        self.assertFalse(breach)


class RankingTests(unittest.TestCase):
    def cand(self, method, schedule, option=None, changes=()):
        return Candidate(method=method, schedule=schedule, payment_option_id=option,
                         changes=list(changes))

    def test_no_spending_change_beats_cheaper_with_change(self):
        a = self.cand("installments", [(date(2025, 1, 5), D("600"))] * 2, "payment_option_2")
        b = self.cand("full_payment", [(date(2025, 1, 1), D("1000"))], "payment_option_1",
                      [SpendingChange("stop", "e1")])
        self.assertIs(rank.pick_winner([b, a]), a)

    def test_lower_total_then_earlier_then_fewer_then_option_id(self):
        cheap = self.cand("full_payment", [(date(2025, 1, 9), D("1000"))], "payment_option_9")
        dear = self.cand("installments", [(date(2025, 1, 1), D("510"))] * 2, "payment_option_1")
        self.assertIs(rank.pick_winner([dear, cheap]), cheap)

        early = self.cand("full_payment", [(date(2025, 1, 1), D("1000"))], "payment_option_5")
        late = self.cand("wait", [(date(2025, 2, 1), D("1000"))])
        self.assertIs(rank.pick_winner([late, early]), early)

        one = self.cand("full_payment", [(date(2025, 1, 1), D("1000"))], "payment_option_7")
        two = self.cand("installments", [(date(2025, 1, 1), D("500")), (date(2025, 2, 1), D("500"))],
                        "payment_option_3")
        self.assertIs(rank.pick_winner([two, one]), one)

        x = self.cand("installments", [(date(2025, 1, 1), D("500"))] * 2, "payment_option_4")
        y = self.cand("installments", [(date(2025, 1, 1), D("500"))] * 2, "payment_option_2")
        self.assertIs(rank.pick_winner([x, y]), y)

    def test_status_mapping(self):
        today = date(2025, 1, 1)
        full = self.cand("full_payment", [(today, D("1"))])
        self.assertEqual(rank.affordability_status(full, today), "affordable_now")
        full.changes = [SpendingChange("stop", "e1")]
        self.assertEqual(rank.affordability_status(full, today), "affordable_with_plan")
        self.assertEqual(rank.affordability_status(self.cand("wait", [(today, D("1"))]), today),
                         "affordable_later")


class FormattingTests(unittest.TestCase):
    def test_amount_column_is_plain(self):
        self.assertEqual(fmt_amount(D("12693000")), "12693000")
        self.assertEqual(fmt_amount(D("603.30")), "603.3")

    def test_plan_money_keeps_cents(self):
        self.assertEqual(fmt_money(D("620.4")), "620.40")
        self.assertEqual(fmt_money(D("25256")), "25256")
        self.assertEqual(fmt_plan([(date(2025, 1, 1), D("10.5")), (date(2025, 2, 1), D("20"))]),
                         "2025-01-01:10.50|2025-02-01:20")
        self.assertEqual(fmt_plan([]), "none")

    def test_reduce_to_format(self):
        self.assertEqual(SpendingChange("reduce", "event_9", D("23.5")).as_text(),
                         "reduce_to:event_9:23.50")


if __name__ == "__main__":
    unittest.main()
