#!/usr/bin/env python3
"""Regression evaluator: run the engine over ``dataset/sample_requests.csv``
and report field-level agreement against its solved columns.

This is a sanity check, not the grading mechanism - the 25 samples are a
tiny, hand-picked slice, and problem_statement.md is the actual source of
truth. A field that mismatches here is worth understanding (it usually means
a real bug), but the goal is a system that follows the written rules
correctly on any request, not one curve-fit to these 25 rows.

Usage::

    python code/tools/evaluate.py             # summary + first mismatch per row
    python code/tools/evaluate.py -v          # every mismatched field, every row
    python code/tools/evaluate.py -v request_06   # just one request, in full
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bow import dataset, engine

FIELDS = [
    "amount_safe_to_pay", "affordability_status", "recommended_payment_method",
    "payment_plan", "earliest_date_for_full_payment", "spending_changes_needed",
]


def main() -> int:
    verbose = "-v" in sys.argv
    only = next((a for a in sys.argv[1:] if a.startswith("request_")), None)

    data = dataset.load()
    eng = engine.Engine(data)

    per_field_ok = {f: 0 for f in FIELDS}
    fully_ok = 0
    crashed = []
    invalid = []
    total = 0

    for request in data.samples:
        if only and request.request_id != only:
            continue
        total += 1
        try:
            decision = eng.decide(request)
        except Exception as exc:  # noqa: BLE001 - report, don't hide
            crashed.append((request.request_id, exc))
            continue

        row = decision.row
        truth = request.truth
        mism = []
        for field in FIELDS:
            if str(row[field]) == str(truth.get(field, "")):
                per_field_ok[field] += 1
            else:
                mism.append(field)

        if decision.problems:
            invalid.append((request.request_id, decision.problems))

        if not mism and not decision.problems:
            fully_ok += 1
        elif verbose or only:
            print(f"\n{request.request_id}  user={request.user_id}  "
                 f"{request.request_type}  {request.requested_amount} due {request.desired_completion_date}")
            if decision.problems:
                print("  STRUCTURAL PROBLEMS:", decision.problems)
            for field in mism:
                print(f"  {field}:")
                print(f"    got:   {row[field]}")
                print(f"    truth: {truth.get(field, '')}")
            if only:
                print(f"  decision_explanation: {row['decision_explanation']}")

    print(f"\n=== {total} sample requests ===")
    print(f"fully correct (all 6 predicted fields + structurally valid): {fully_ok}/{total}")
    for field in FIELDS:
        print(f"  {field:32s} {per_field_ok[field]:3d}/{total}")
    if invalid:
        print(f"\nstructurally invalid rows: {len(invalid)}")
        for rid, problems in invalid:
            print(f"  {rid}: {problems}")
    if crashed:
        print(f"\nCRASHED: {len(crashed)}")
        for rid, exc in crashed:
            print(f"  {rid}: {type(exc).__name__}: {exc}")

    return 1 if crashed else 0


if __name__ == "__main__":
    raise SystemExit(main())
