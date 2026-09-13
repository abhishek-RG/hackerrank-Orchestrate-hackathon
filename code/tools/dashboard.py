#!/usr/bin/env python3
"""Local dashboard for interactively exploring and testing the engine.

Not part of the graded pipeline - main.py never imports this, and it changes
nothing about how output.csv gets produced. A small, dependency-free
(stdlib only) HTTP server in front of the same bow.* modules main.py uses,
plus a static single-page frontend, so you can:

  * browse every request (the 250 eval rows and the 25 solved samples) with
    its computed decision, filterable by status/method and searchable
  * open one request to see its full reconstructed cash-flow timeline, every
    candidate plan the engine considered (not just the winner) and why it
    was accepted or rejected, and - for a sample - a field-by-field diff
    against the solved truth
  * run "what-if" recomputations - override requested_amount,
    desired_completion_date, allows_partial_payment, minimum_balance_to_keep,
    or current_available_balance - and see the decision update instantly, in
    the same process, without touching dataset/ on disk or restarting
    anything

Usage:
    python code/tools/dashboard.py [--port 8765]

Then open http://localhost:8765/ in a browser.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import os
import sys
from datetime import date
from decimal import Decimal, InvalidOperation
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bow import candidates as cand_mod
from bow import dataset, explain, forecast, output, rank, validate
from bow.dataset import Dataset, Profile, Request
from bow.state import Reconstructor

HERE = os.path.dirname(os.path.abspath(__file__))
UI_PATH = os.path.join(HERE, "dashboard_ui.html")


# --------------------------------------------------------------------------
# JSON-safe serialization helpers
# --------------------------------------------------------------------------

def jd(value):
    """Make a value JSON-serializable (Decimal -> str, date -> ISO)."""
    if isinstance(value, Decimal):
        return output.fmt_amount(value)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, list):
        return [jd(v) for v in value]
    if isinstance(value, tuple):
        return [jd(v) for v in value]
    if isinstance(value, dict):
        return {k: jd(v) for k, v in value.items()}
    return value


def candidate_to_dict(c: cand_mod.Candidate) -> dict:
    return {
        "method": c.method,
        "schedule": [{"date": jd(d), "amount": jd(a)} for d, a in c.schedule],
        "payment_option_id": c.payment_option_id,
        "changes": [
            {"op": ch.op, "event_id": ch.event_id, "new_amount": jd(ch.new_amount)}
            for ch in c.changes
        ],
        "total_paid": jd(c.total_paid),
        "starts": jd(c.starts),
        "num_payments": c.num_payments,
    }


# --------------------------------------------------------------------------
# core: decide() re-implemented here (not imported from bow.engine) so this
# dev tool can surface every candidate the engine considered, not only the
# winner - bow.engine.Engine stays untouched, since it is the graded path.
# --------------------------------------------------------------------------

class Explorer:
    def __init__(self, data: Dataset):
        self.data = data
        self.reconstructor = Reconstructor(data)
        self._by_id: Dict[str, Request] = {r.request_id: r for r in data.requests + data.samples}
        self._sample_ids = {r.request_id for r in data.samples}

    def all_ids(self) -> List[str]:
        # samples first (they carry ground truth), then eval requests
        return [r.request_id for r in self.data.samples] + [r.request_id for r in self.data.requests]

    def get_request(self, request_id: str) -> Optional[Request]:
        return self._by_id.get(request_id)

    def evaluate(self, request: Request, profile: Optional[Profile] = None) -> dict:
        profile = profile or self.data.profiles[request.user_id]
        state = self.reconstructor.build(request.user_id, request.request_date)
        options = self.data.options_by_request.get(request.request_id, [])

        points = forecast.build_points(state.opening_balance, state.flows,
                                       request.request_date, state.horizon)
        amount_safe = forecast.max_safe_amount(
            points, state.opening_balance, request.request_date, state.horizon,
            profile.minimum_balance_to_keep, request.requested_amount,
        )
        earliest_full = forecast.earliest_full_payment_date(
            points, state.opening_balance, request.request_date, state.horizon,
            profile.minimum_balance_to_keep, request.requested_amount,
        )

        candidate_list = cand_mod.build_candidates(
            profile, request, state, amount_safe, earliest_full, options,
        )
        winner = rank.pick_winner(candidate_list)

        if winner is None:
            status, method, plan_text = "not_affordable", "not_recommended", "none"
        else:
            status = rank.affordability_status(winner, request.request_date)
            method = winner.method
            plan_text = output.fmt_plan(winner.schedule)

        row = {
            "request_id": request.request_id,
            "amount_safe_to_pay": output.fmt_amount(amount_safe),
            "affordability_status": status,
            "recommended_payment_method": method,
            "payment_plan": plan_text,
            "earliest_date_for_full_payment": output.fmt_date(earliest_full),
            "spending_changes_needed": explain.spending_changes_text(winner.changes if winner else []),
            "decision_explanation": explain.decision_explanation(profile, request, state, winner, status),
        }
        problems = validate.validate_row(request, profile, row, winner, options)

        timeline = [
            {"date": jd(f.when), "amount": jd(f.amount), "label": f.label,
            "category": f.category, "recurring": f.recurring}
            for f in state.sorted_flows()
        ]

        return {
            "row": row,
            "problems": problems,
            "winner": candidate_to_dict(winner) if winner else None,
            "candidates": [candidate_to_dict(c) for c in candidate_list],
            "timeline": timeline,
            "opening_balance": jd(state.opening_balance),
            "minimum_balance_to_keep": jd(profile.minimum_balance_to_keep),
            "options": [
                {"payment_option_id": o.payment_option_id, "payment_method": o.payment_method,
                 "number_of_payments": o.number_of_payments,
                 "schedule": [{"date": jd(d), "amount": jd(a)} for d, a in o.schedule()],
                 "total_payable_amount": jd(o.total_payable_amount)}
                for o in options
            ],
        }

    def summary_row(self, request: Request) -> dict:
        result = self.evaluate(request)
        row = result["row"]
        truth = request.truth if request.request_id in self._sample_ids else None
        matches = None
        if truth:
            fields = ["amount_safe_to_pay", "affordability_status", "recommended_payment_method",
                     "payment_plan", "earliest_date_for_full_payment", "spending_changes_needed"]
            matches = sum(1 for f in fields if str(row[f]) == str(truth.get(f, "")))
        profile = self.data.profiles[request.user_id]
        return {
            "request_id": request.request_id,
            "user_id": request.user_id,
            "request_type": request.request_type,
            "requested_amount": jd(request.requested_amount),
            "currency": profile.home_currency,
            "request_date": jd(request.request_date),
            "desired_completion_date": jd(request.desired_completion_date),
            "affordability_status": row["affordability_status"],
            "recommended_payment_method": row["recommended_payment_method"],
            "amount_safe_to_pay": row["amount_safe_to_pay"],
            "is_sample": request.request_id in self._sample_ids,
            "sample_match": matches,
            "sample_total_fields": 6 if truth else None,
            "problems": result["problems"],
        }


# --------------------------------------------------------------------------
# what-if overrides
# --------------------------------------------------------------------------

def _dec(value, default=None):
    if value in (None, ""):
        return default
    try:
        return Decimal(str(value))
    except InvalidOperation:
        return default


def _date(value, default=None):
    if not value:
        return default
    return dataset.parse_date(value)


def apply_overrides(explorer: Explorer, base_request: Request, overrides: dict):
    """Return (request, profile) with any supplied overrides applied.

    Never mutates the loaded dataset - both objects are shallow copies.
    """
    request = dataclasses.replace(
        base_request,
        requested_amount=_dec(overrides.get("requested_amount"), base_request.requested_amount),
        desired_completion_date=_date(overrides.get("desired_completion_date"),
                                      base_request.desired_completion_date),
        request_date=_date(overrides.get("request_date"), base_request.request_date),
        allows_partial_payment=bool(overrides.get("allows_partial_payment")) if
            "allows_partial_payment" in overrides else base_request.allows_partial_payment,
    )
    base_profile = explorer.data.profiles[base_request.user_id]
    profile = dataclasses.replace(
        base_profile,
        minimum_balance_to_keep=_dec(overrides.get("minimum_balance_to_keep"),
                                     base_profile.minimum_balance_to_keep),
        current_available_balance=_dec(overrides.get("current_available_balance"),
                                       base_profile.current_available_balance),
    )
    return request, profile


# --------------------------------------------------------------------------
# HTTP layer
# --------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    explorer: Explorer = None  # set by main()

    def log_message(self, fmt, *args):
        pass  # keep stdout quiet; errors still raise

    def _send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, path, status=200):
        with open(path, "rb") as handle:
            body = handle.read()
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path in ("/", "/index.html"):
                self._send_html(UI_PATH)
            elif parsed.path == "/api/requests":
                qs = parse_qs(parsed.query)
                query = (qs.get("q", [""])[0] or "").lower()
                rows = []
                for rid in self.explorer.all_ids():
                    req = self.explorer.get_request(rid)
                    if query and query not in rid.lower() and query not in req.user_id.lower() \
                            and query not in req.request_type.lower():
                        continue
                    rows.append(self.explorer.summary_row(req))
                self._send_json({"requests": rows})
            elif parsed.path.startswith("/api/request/"):
                rid = parsed.path.rsplit("/", 1)[-1]
                req = self.explorer.get_request(rid)
                if req is None:
                    self._send_json({"error": f"unknown request_id {rid}"}, status=404)
                    return
                profile = self.explorer.data.profiles[req.user_id]
                result = self.explorer.evaluate(req)
                truth = req.truth if rid in self.explorer._sample_ids else None
                self._send_json({
                    "request": {
                        "request_id": req.request_id, "user_id": req.user_id,
                        "request_type": req.request_type,
                        "requested_amount": jd(req.requested_amount),
                        "request_date": jd(req.request_date),
                        "desired_completion_date": jd(req.desired_completion_date),
                        "allows_partial_payment": req.allows_partial_payment,
                        "request_text": req.request_text,
                    },
                    "profile": {
                        "home_currency": profile.home_currency,
                        "current_available_balance": jd(profile.current_available_balance),
                        "minimum_balance_to_keep": jd(profile.minimum_balance_to_keep),
                        "financial_priorities": profile.financial_priorities,
                        "protect": profile.protect,
                        "willing_to_reduce": profile.willing_to_reduce,
                        "willing_to_stop": profile.willing_to_stop,
                        "payment_methods": profile.payment_methods,
                        "max_installment_months": profile.max_installment_months,
                    },
                    "result": result,
                    "truth": truth,
                })
            else:
                self._send_json({"error": "not found"}, status=404)
        except Exception as exc:  # noqa: BLE001
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=500)

    def do_POST(self):
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(body or b"{}")
            if parsed.path == "/api/recompute":
                rid = payload.get("request_id")
                base_request = self.explorer.get_request(rid)
                if base_request is None:
                    self._send_json({"error": f"unknown request_id {rid}"}, status=404)
                    return
                overrides = payload.get("overrides", {})
                request, profile = apply_overrides(self.explorer, base_request, overrides)
                result = self.explorer.evaluate(request, profile=profile)
                self._send_json({
                    "request": {
                        "requested_amount": jd(request.requested_amount),
                        "request_date": jd(request.request_date),
                        "desired_completion_date": jd(request.desired_completion_date),
                        "allows_partial_payment": request.allows_partial_payment,
                    },
                    "profile": {
                        "current_available_balance": jd(profile.current_available_balance),
                        "minimum_balance_to_keep": jd(profile.minimum_balance_to_keep),
                    },
                    "result": result,
                })
            else:
                self._send_json({"error": "not found"}, status=404)
        except Exception as exc:  # noqa: BLE001
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=500)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--dataset", default=None)
    args = parser.parse_args()

    data = dataset.load(args.dataset)
    Handler.explorer = Explorer(data)

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Buy or Wait? dashboard: http://localhost:{args.port}/")
    print(f"  {len(data.samples)} solved samples + {len(data.requests)} eval requests loaded")
    print("  Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
