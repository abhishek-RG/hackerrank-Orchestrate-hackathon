#!/usr/bin/env python3
"""Pre-submission audit of ``output.csv``, independent of the engine.

Reads only the written CSV plus the input files, so it catches anything a bug
in the engine (or a stale file) could let through: schema, row coverage,
enums, numeric bounds, date validity, plan structure, installment matching and
spending-change syntax.

Usage::

    python code/tools/audit_output.py [path/to/output.csv]
"""

from __future__ import annotations

import csv
import os
import re
import sys
from decimal import Decimal, InvalidOperation

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bow import dataset                                          # noqa: E402
from bow.dataset import parse_date                               # noqa: E402
from bow.output import REQUIRED_COLUMNS, fmt_money               # noqa: E402

STATUSES = {"affordable_now", "affordable_with_plan", "affordable_later", "not_affordable"}
METHODS = {"full_payment", "partial_payment", "installments", "wait", "not_recommended"}
CHANGE = re.compile(r"^(stop:event_\d+|reduce_to:event_\d+:\d+(\.\d+)?)$")


def audit(path: str) -> list:
    data = dataset.load()
    requests = {r.request_id: r for r in data.requests}
    problems = []
    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != REQUIRED_COLUMNS:
            problems.append(f"columns {reader.fieldnames} != {REQUIRED_COLUMNS}")
        rows = list(reader)

    ids = [row["request_id"] for row in rows]
    if len(rows) != len(requests):
        problems.append(f"{len(rows)} rows, expected {len(requests)}")
    if len(set(ids)) != len(ids):
        problems.append("duplicate request_id rows")
    missing = set(requests) - set(ids)
    if missing:
        problems.append(f"missing request_ids: {sorted(missing)[:5]}")

    for row in rows:
        rid = row["request_id"]
        req = requests.get(rid)
        if req is None:
            problems.append(f"{rid}: not in requests.csv")
            continue
        profile = data.profiles[req.user_id]
        bad = lambda msg: problems.append(f"{rid}: {msg}")          # noqa: E731

        try:
            safe = Decimal(row["amount_safe_to_pay"])
            if not Decimal(0) <= safe <= req.requested_amount:
                bad(f"amount_safe_to_pay {safe} out of bounds")
        except InvalidOperation:
            bad("amount_safe_to_pay not numeric")
            continue
        status, method = row["affordability_status"], row["recommended_payment_method"]
        if status not in STATUSES:
            bad(f"status {status}")
        if method not in METHODS:
            bad(f"method {method}")

        earliest = row["earliest_date_for_full_payment"]
        if earliest:
            if parse_date(earliest) is None or len(earliest) != 10:
                bad("earliest date malformed")
        if status == "affordable_now" and earliest != req.request_date.isoformat():
            bad("affordable_now requires earliest == request_date")

        plan = row["payment_plan"]
        schedule = []
        if plan != "none":
            for part in plan.split("|"):
                when, _, amount = part.partition(":")
                schedule.append((parse_date(when), Decimal(amount)))
            dates = [d for d, _ in schedule]
            if dates != sorted(dates):
                bad("plan not chronological")
            if any(d < req.request_date for d in dates):
                bad("payment before request_date")
            if dates[-1] > req.desired_completion_date:
                bad("plan finishes after desired_completion_date")
        if method in ("not_recommended",) and plan != "none":
            bad("not_recommended must have plan none")
        if method != "not_recommended" and method not in profile.payment_methods and method != "wait":
            bad(f"method {method} not accepted by user")
        if method == "wait" and "full_payment" not in profile.payment_methods:
            bad("wait requires the user to accept full_payment")

        total = sum((a for _, a in schedule), Decimal(0))
        if method in ("full_payment", "wait", "partial_payment") and total != req.requested_amount:
            bad(f"{method} pays {total}, expected {req.requested_amount}")
        if method == "partial_payment":
            if not req.allows_partial_payment:
                bad("partial payment not allowed by request")
            if len(schedule) != 2 or schedule[0] != (req.request_date, safe) \
                    or schedule[1][0].isoformat() != earliest:
                bad("partial payment must be safe-today then remainder on earliest date")
            if status != "affordable_with_plan":
                bad("partial_payment requires affordable_with_plan")
        if method == "installments":
            options = [o for o in data.options_by_request.get(rid, [])
                       if o.payment_method == "installments"]
            rendered = {"|".join(f"{d.isoformat()}:{fmt_money(a)}" for d, a in o.schedule())
                        for o in options}
            if plan not in rendered:
                bad("installment plan matches no supplied option")

        changes = row["spending_changes_needed"]
        if changes != "none":
            parts = changes.split("|")
            if len(parts) > 3:
                bad("more than three spending changes")
            if any(not CHANGE.match(p) for p in parts):
                bad(f"malformed spending change {changes}")
            if len({p.split(":")[1] for p in parts}) != len(parts):
                bad("same event changed twice")
        if not row["decision_explanation"].strip():
            bad("empty explanation")
    return problems


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(dataset.REPO_ROOT, "output.csv")
    problems = audit(path)
    if problems:
        print(f"AUDIT FAILED: {len(problems)} problem(s)")
        for problem in problems[:50]:
            print("  " + problem)
        return 1
    print(f"AUDIT OK: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
