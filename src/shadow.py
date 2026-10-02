"""Observational, bounded comparison of production attack choices and MCTS.

This module is intentionally not connected to the runtime decision path. It
copies the supplied policy context, builds the production candidate board once,
and gives that exact board to each deterministic MCTS budget.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

from .mcts import MCTSPlanner
from .policy import ActionCandidate, PolicyController, PolicyInput
from .search_opponent import OPPONENT_RESPONSE_MODEL
from .teamplan import TeamPlan

DEFAULT_BUDGETS = (4, 12, 32)
MAX_SHADOW_ITERATIONS = 256
MAX_SHADOW_BUDGETS = 5


def _rounded(value: float | None) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    rounded = round(number, 6)
    return 0.0 if rounded == 0.0 else rounded


def candidate_signature(candidate: ActionCandidate) -> str:
    """Stable action-semantic identity; human-readable reason is excluded."""
    intent = candidate.intent
    signature = [
        intent.pid,
        intent.action_type,
        _rounded(intent.tx),
        _rounded(intent.ty),
        _rounded(intent.speed),
        _rounded(intent.face_x),
        _rounded(intent.face_y),
        list(map(_rounded, intent.action_target))
        if intent.action_target is not None
        else None,
        intent.receiver_id,
        list(map(_rounded, intent.collection_point))
        if intent.collection_point is not None
        else None,
        _rounded(intent.action_power),
    ]
    return json.dumps(signature, separators=(",", ":"), ensure_ascii=True)


def candidate_action_type(candidate: ActionCandidate, inp: PolicyInput) -> str:
    """Return pass/shoot/carry/none without consulting reason strings."""
    action = candidate.intent.action_type
    if action != "none":
        return action
    player = next(
        (player for player in inp.state.us if player.id == candidate.intent.pid),
        None,
    )
    if player is not None and math.hypot(
        candidate.intent.tx - player.x, candidate.intent.ty - player.y
    ) > 1e-6:
        return "carry"
    return "none"


@dataclass(frozen=True)
class BudgetResult:
    budget: int
    selected_index: int | None
    selected_signature: str | None
    selected_action_type: str | None
    selected_production_value: float | None
    iterations: int
    nodes_created: int
    max_depth: int
    simulated_steps: int


@dataclass(frozen=True)
class ShadowResult:
    """Compact primitive diagnostics for one root state and candidate board."""

    schema_version: int
    opponent_model: str
    eligible: bool
    skip_reason: str | None
    mcts_ran: bool
    agreement: bool | None
    production_action_type: str | None
    mcts_action_type: str | None
    production_signature: str | None
    mcts_signature: str | None
    production_value: float | None
    mcts_production_value: float | None
    production_ev_difference: float | None
    mcts_preference: float | None
    root_candidate_count: int
    production_index: int | None
    mcts_index: int | None
    iterations: int
    nodes_created: int
    max_depth: int
    simulated_steps: int
    root_visits: tuple[int, ...]
    root_mean_values: tuple[float, ...]
    candidate_board: tuple[dict[str, Any], ...]
    root_position: tuple[float, float]
    pitch_zone: str
    root_snapshot: dict[str, Any]
    possession_player: str | None
    simulation_tick: int
    budget_stable: bool | None
    budget_results: tuple[BudgetResult, ...]

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON/NDJSON-compatible primitives."""
        return asdict(self)


def _ineligible(inp: PolicyInput, reason: str) -> ShadowResult:
    possessor = inp.state.our_possessor()
    return ShadowResult(
        schema_version=2,
        opponent_model=OPPONENT_RESPONSE_MODEL,
        eligible=False,
        skip_reason=reason,
        mcts_ran=False,
        agreement=None,
        production_action_type=None,
        mcts_action_type=None,
        production_signature=None,
        mcts_signature=None,
        production_value=None,
        mcts_production_value=None,
        production_ev_difference=None,
        mcts_preference=None,
        root_candidate_count=0,
        production_index=None,
        mcts_index=None,
        iterations=0,
        nodes_created=0,
        max_depth=0,
        simulated_steps=0,
        root_visits=(),
        root_mean_values=(),
        candidate_board=(),
        root_position=(_rounded(inp.state.ball.x) or 0.0, _rounded(inp.state.ball.y) or 0.0),
        pitch_zone=inp.world.ball_zone,
        root_snapshot=_snapshot(inp),
        possession_player=possessor.id if possessor else None,
        simulation_tick=inp.state.simulation_tick,
        budget_stable=None,
        budget_results=(),
    )


def _snapshot(inp: PolicyInput) -> dict[str, Any]:
    state = inp.state
    ball = state.ball
    players = sorted((*state.us, *state.them), key=lambda player: (player.team, player.id))
    return {
        "phase": state.phase,
        "score_us": state.score_us,
        "score_them": state.score_them,
        "time_remaining": _rounded(inp.tr),
        "ball": {
            "x": _rounded(ball.x),
            "y": _rounded(ball.y),
            "vx": _rounded(ball.vx),
            "vy": _rounded(ball.vy),
            "possessing_team": ball.possessing_team,
            "possessing_player": ball.possessing_player,
        },
        "players": [
            {
                "id": player.id,
                "team": player.team,
                "role": player.role,
                "x": _rounded(player.x),
                "y": _rounded(player.y),
                "vx": _rounded(player.vx),
                "vy": _rounded(player.vy),
                "facing": _rounded(player.facing),
                "can_act": player.can_act,
            }
            for player in players
        ],
    }


def evaluate_shadow(
    inp: PolicyInput,
    controller: PolicyController | None = None,
    team_plan: TeamPlan | None = None,
    *,
    budgets: tuple[int, ...] | list[int] = DEFAULT_BUDGETS,
    horizon: float = 1.5,
    max_decision_depth: int = 3,
    max_advance_steps: int = 8,
) -> ShadowResult:
    """Compare the top production candidate with MCTS on exactly one board.

    Input, controller, TeamPlan, candidate generation and search all run on
    deep-copied context. The original objects are not committed, logged to, or
    otherwise changed. Budgets are explicit and capped at 256 iterations each.
    The recorded MCTS choice is from the largest supplied budget.
    """
    state = inp.state
    possessor = state.our_possessor()
    if possessor is None:
        return _ineligible(inp, "no_our_possessor")
    if state.phase != "openPlay":
        return _ineligible(inp, "not_open_play")
    if not possessor.can_act:
        return _ineligible(inp, "possessor_unable_to_act")
    if inp.tr <= 0.0:
        return _ineligible(inp, "time_expired")

    try:
        requested_budgets = tuple(int(budget) for budget in budgets)
    except (TypeError, ValueError):
        raise ValueError("budgets must be a sequence of integer iteration counts") from None
    if (
        not requested_budgets
        or len(requested_budgets) > MAX_SHADOW_BUDGETS
        or any(budget < 0 or budget > MAX_SHADOW_ITERATIONS for budget in requested_budgets)
    ):
        raise ValueError(
            f"supply 1..{MAX_SHADOW_BUDGETS} budgets between 0 and "
            f"{MAX_SHADOW_ITERATIONS} iterations"
        )
    if not math.isfinite(horizon) or horizon < 0.0:
        raise ValueError("horizon must be finite and non-negative")

    isolated_input = copy.deepcopy(inp)
    isolated_controller = copy.deepcopy(controller) if controller is not None else PolicyController()
    isolated_plan = copy.deepcopy(team_plan)
    if isolated_plan is None:
        live_manager = getattr(isolated_controller, "_team_plan_manager", None)
        isolated_plan = copy.deepcopy(
            getattr(live_manager, "current_plan", None)
        )

    isolated_possessor = isolated_input.state.our_possessor()
    if isolated_possessor is None or not isolated_possessor.can_act:
        return _ineligible(inp, "possessor_unavailable_in_snapshot")

    # This is the same production pipeline used by _evaluate_attack_actions;
    # its ranked list becomes the immutable-at-use root board for every budget.
    board = isolated_controller._build_attack_candidates(
        isolated_input, isolated_possessor, isolated_plan
    )
    board = isolated_controller._filter_attack_candidates(isolated_possessor, board)
    board = isolated_controller._rank_attack_candidates(board)
    board = [candidate for candidate in board if math.isfinite(candidate.value)]
    if not board:
        return _ineligible(inp, "empty_production_candidate_board")

    production = board[0]
    signatures = [candidate_signature(candidate) for candidate in board]
    production_signature = signatures[0]
    production_action = candidate_action_type(production, isolated_input)
    candidate_rows = tuple(
        {
            "index": index,
            "signature": signatures[index],
            "action_type": candidate_action_type(candidate, isolated_input),
            "production_value": _rounded(candidate.value),
        }
        for index, candidate in enumerate(board)
    )

    budget_results: list[BudgetResult] = []
    budget_planners: list[MCTSPlanner] = []
    for budget in requested_budgets:
        planner = MCTSPlanner(
            genome=isolated_input.config.genome,
            weights=isolated_input.config.reward_weights,
            iterations=budget,
            horizon=horizon,
            max_decision_depth=max_decision_depth,
            max_advance_steps=max_advance_steps,
        )
        selected = planner.search(
            isolated_input,
            controller=isolated_controller,
            team_plan=isolated_plan,
            root_candidates=board,
        )
        selected_index = next(
            (index for index, candidate in enumerate(board) if candidate is selected),
            None,
        )
        if selected_index is None and selected is not None:
            selected_signature = candidate_signature(selected)
            selected_index = next(
                (index for index, signature in enumerate(signatures) if signature == selected_signature),
                None,
            )
        else:
            selected_signature = (
                signatures[selected_index] if selected_index is not None else None
            )
        selected_candidate = board[selected_index] if selected_index is not None else None
        stats = planner.stats
        budget_results.append(
            BudgetResult(
                budget=budget,
                selected_index=selected_index,
                selected_signature=selected_signature,
                selected_action_type=(
                    candidate_action_type(selected_candidate, isolated_input)
                    if selected_candidate is not None
                    else None
                ),
                selected_production_value=(
                    _rounded(selected_candidate.value)
                    if selected_candidate is not None
                    else None
                ),
                iterations=stats.iterations,
                nodes_created=stats.nodes_created,
                max_depth=stats.max_depth,
                simulated_steps=stats.simulated_steps,
            )
        )
        budget_planners.append(planner)

    # Highest iteration budget is the headline choice, independently of the
    # caller's order. Equal budgets resolve to the later supplied run.
    headline_position = max(
        range(len(requested_budgets)), key=lambda index: (requested_budgets[index], index)
    )
    headline = budget_results[headline_position]
    headline_planner = budget_planners[headline_position]
    mcts_index = headline.selected_index
    mcts_candidate = board[mcts_index] if mcts_index is not None else None
    stable = len({result.selected_signature for result in budget_results}) <= 1
    root_stats = headline_planner.stats.root_actions
    root_visits = tuple(stat.visits for stat in root_stats)
    root_means = tuple(_rounded(stat.mean_value) or 0.0 for stat in root_stats)
    candidate_rows = tuple(
        {
            **row,
            "root_visits": (
                root_stats[index].visits if index < len(root_stats) else 0
            ),
            "root_mean_value": (
                _rounded(root_stats[index].mean_value)
                if index < len(root_stats)
                else None
            ),
            "score_delta_us": (
                root_stats[index].score_delta_us
                if index < len(root_stats)
                else None
            ),
            "score_delta_them": (
                root_stats[index].score_delta_them
                if index < len(root_stats)
                else None
            ),
            "goal_scored": (
                root_stats[index].goal_scored
                if index < len(root_stats)
                else None
            ),
            "goal_conceded": (
                root_stats[index].goal_conceded
                if index < len(root_stats)
                else None
            ),
            "goal_terminal": (
                root_stats[index].goal_terminal
                if index < len(root_stats)
                else None
            ),
            "branch_outcome": (
                root_stats[index].branch_outcome
                if index < len(root_stats)
                else "not_simulated"
            ),
        }
        for index, row in enumerate(candidate_rows)
    )
    mcts_preference = None
    if mcts_index is not None and mcts_index < len(root_means):
        mcts_preference = _rounded(root_means[mcts_index] - root_means[0])

    return ShadowResult(
        schema_version=2,
        opponent_model=OPPONENT_RESPONSE_MODEL,
        eligible=True,
        skip_reason=None,
        mcts_ran=True,
        agreement=(
            candidate_signature(mcts_candidate) == production_signature
            if mcts_candidate is not None
            else False
        ),
        production_action_type=production_action,
        mcts_action_type=(
            candidate_action_type(mcts_candidate, isolated_input)
            if mcts_candidate is not None
            else None
        ),
        production_signature=production_signature,
        mcts_signature=(candidate_signature(mcts_candidate) if mcts_candidate else None),
        production_value=_rounded(production.value),
        mcts_production_value=(
            _rounded(mcts_candidate.value) if mcts_candidate is not None else None
        ),
        production_ev_difference=(
            _rounded(production.value - mcts_candidate.value)
            if mcts_candidate is not None
            else None
        ),
        mcts_preference=mcts_preference,
        root_candidate_count=len(board),
        production_index=0,
        mcts_index=mcts_index,
        iterations=headline.iterations,
        nodes_created=headline.nodes_created,
        max_depth=headline.max_depth,
        simulated_steps=headline.simulated_steps,
        root_visits=root_visits,
        root_mean_values=root_means,
        candidate_board=candidate_rows,
        root_position=(
            _rounded(state.ball.x) or 0.0,
            _rounded(state.ball.y) or 0.0,
        ),
        pitch_zone=isolated_input.world.ball_zone,
        root_snapshot=_snapshot(inp),
        possession_player=possessor.id,
        simulation_tick=state.simulation_tick,
        budget_stable=stable,
        budget_results=tuple(budget_results),
    )


__all__ = [
    "DEFAULT_BUDGETS",
    "BudgetResult",
    "ShadowResult",
    "candidate_action_type",
    "candidate_signature",
    "evaluate_shadow",
]
