#!/usr/bin/env python3
"""Development tool: measure the forecast model against the solved samples.

``sample_requests.csv`` publishes ``amount_safe_to_pay`` for 25 requests. Where
that figure is below ``requested_amount`` it is not capped, so the definition in
problem_statement.md inverts exactly:

    amount_safe_to_pay = min_balance_over_90_days - minimum_balance_to_keep
 => min_balance_over_90_days = amount_safe_to_pay + minimum_balance_to_keep

That gives a directly observable target for the forecast model - the one number
the whole engine is built to compute - without going through candidate
generation, ranking or formatting. This tool reports the model's error against
those targets and can grid-search :class:`bow.state.Config` over them.

This is calibration of *model structure* (how a recurring amount is chosen, and
whether a same-day charge counts), not fitting per-request answers: every knob
is a single global choice applied to every user. Run it after changing anything
in :mod:`bow.recurrence`, :mod:`bow.state` or :mod:`bow.forecast`.

Usage::

    python code/tools/calibrate.py                # error report for the defaults
    python code/tools/calibrate.py --grid         # search the knob combinations
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from dataclasses import replace
from decimal import Decimal
from typing import List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bow import dataset, engine, forecast
from bow.state import Config


def targets(data: dataset.Dataset) -> List[Tuple[dataset.Request, dataset.Profile, Decimal]]:
    """Samples whose published ``amount_safe_to_pay`` is not capped."""
    out = []
    for request in data.samples:
        profile = data.profiles[request.user_id]
        published = Decimal(request.truth["amount_safe_to_pay"])
        if published >= request.requested_amount:
            continue          # capped at the request: reveals nothing about the model
        out.append((request, profile, published + profile.minimum_balance_to_keep))
    return out


def model_min_balance(eng: engine.Engine, request: dataset.Request,
                      profile: dataset.Profile) -> Optional[Decimal]:
    state = eng.reconstructor.build(request.user_id, request.request_date)
    points = forecast.build_points(state.opening_balance, state.flows,
                                   request.request_date, state.horizon)
    return forecast.min_balance_over(points, state.opening_balance,
                                     request.request_date, state.horizon)


def score(data: dataset.Dataset, config: Config) -> Tuple[int, Decimal, Decimal, List[tuple]]:
    """``(exact_hits, median_rel_error, mean_rel_error, per_sample_rows)``."""
    eng = engine.Engine(data, config=config)
    rows = []
    for request, profile, target in targets(data):
        try:
            got = model_min_balance(eng, request, profile)
        except Exception as exc:                      # noqa: BLE001 - reported, not hidden
            rows.append((request.request_id, target, None, Decimal(1), repr(exc)))
            continue
        rel = abs(got - target) / max(abs(target), Decimal(1))
        rows.append((request.request_id, target, got, rel, ""))
    errors = sorted(row[3] for row in rows)
    exact = sum(1 for row in rows if row[2] is not None and abs(row[2] - row[1]) <= Decimal("0.05"))
    median = errors[len(errors) // 2] if errors else Decimal(1)
    mean = sum(errors) / Decimal(len(errors)) if errors else Decimal(1)
    return exact, median, mean, rows


def report(data: dataset.Dataset, config: Config) -> None:
    exact, median, mean, rows = score(data, config)
    print(f"{'request':<12}{'target_min_bal':>20}{'model_min_bal':>20}{'rel_err':>10}")
    for rid, target, got, rel, err in rows:
        shown = f"{got:.2f}" if got is not None else err[:40]
        print(f"{rid:<12}{target:>20}{shown:>20}{float(rel):>10.4f}")
    print(f"\n{len(rows)} uncapped samples | exact={exact} "
          f"median_rel_err={float(median):.4f} mean_rel_err={float(mean):.4f}")


def grid(data: dataset.Dataset) -> None:
    amounts = ["latest", "mean", "mean3", "median", "max"]
    windows = [None, 365, 180, 120, 90]
    same_day = [True, False]
    results = []
    for amount, window, incl in itertools.product(amounts, windows, same_day):
        config = Config(recurrence_window_days=window, recurring_amount=amount,
                        include_request_date=incl)
        exact, median, mean, _ = score(data, config)
        results.append((exact, -median, amount, window, incl, mean))
    results.sort(key=lambda item: (-item[0], item[1]))
    print(f"{'amount':<9}{'window':>8}{'same_day':>10}{'exact':>7}{'median_rel':>12}{'mean_rel':>11}")
    for exact, neg_median, amount, window, incl, mean in results:
        print(f"{amount:<9}{str(window):>8}{str(incl):>10}{exact:>7}"
              f"{float(-neg_median):>12.4f}{float(mean):>11.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", action="store_true", help="search knob combinations")
    args = parser.parse_args()
    data = dataset.load()
    if args.grid:
        grid(data)
    else:
        report(data, Config())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
