"""Per-request orchestration: the only module that calls every stage in order.

    financial-state reconstruction (state.Reconstructor)
        -> forecasting (forecast.py, on the untouched baseline)
        -> candidate generation (candidates.py)
        -> ranking (rank.py)
        -> explanation (explain.py)
        -> row assembly (output.py)
        -> validation (validate.py)

Each stage is independently testable; this module just wires them together
and is deliberately thin.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from . import candidates as cand_mod
from . import explain
from . import forecast
from . import output
from . import rank
from . import validate
from .dataset import Dataset, Request
from .state import Config, Reconstructor


@dataclass
class Decision:
    row: dict
    winner: Optional[cand_mod.Candidate]
    problems: List[str]


class Engine:
    def __init__(self, data: Dataset, config: Optional[Config] = None, use_cache: bool = True):
        self.data = data
        self.reconstructor = Reconstructor(data, config=config, use_cache=use_cache)

    def decide(self, request: Request) -> Decision:
        profile = self.data.profiles[request.user_id]
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
            status = "not_affordable"
            method = "not_recommended"
            plan_text = "none"
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
        return Decision(row=row, winner=winner, problems=problems)
