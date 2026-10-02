#!/usr/bin/env python3
"""Bounded offline comparison of production attack candidates and shadow MCTS.

The simulated match always executes the ordinary PolicyController decision.
MCTS is only evaluated on a deep-copied context and never controls the match.
Reports append to a dedicated NDJSON file; experiments.ndjson is untouched.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DECISION_INTERVAL, RuntimeConfig, default_genome
from src.light import LIntent, LightEngine, make_state
from src.policy import ActionCandidate, PolicyController, PolicyInput, PlayerIntent
from src.shadow import (
    DEFAULT_BUDGETS,
    MAX_SHADOW_BUDGETS,
    MAX_SHADOW_ITERATIONS,
    ShadowResult,
    candidate_signature,
    evaluate_shadow,
)
from src.search_opponent import OPPONENT_RESPONSE_MODEL
from src.sim import BASE_LINEUP, OPPONENTS, mirrored_lineup, plan_to_obs
from src.scenarios import SCENARIOS
from src.state import GameState, WorldModel
from src.tactics import PressPlan, TacticalState, ROLE_DEFENDER, ROLE_STRIKER, ROLE_WIDE_LEFT, ROLE_WIDE_RIGHT
from src.teamplan import TeamPlanManager

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
SLOT_ROLES = {
    "def": ROLE_DEFENDER,
    "left": ROLE_WIDE_LEFT,
    "right": ROLE_WIDE_RIGHT,
    "st": ROLE_STRIKER,
}


def _policy_input(
    state,
    game_id: str,
    step: int,
    genome: dict[str, float],
    *,
    enable_mcts: bool = False,
    mcts_iterations: int = 8,
) -> PolicyInput:
    observation = plan_to_obs(state, game_id, step)
    game_state = GameState.from_observation(observation)
    world = WorldModel.build(game_state)
    tactical = (
        TacticalState.ATTACK
        if game_state.has_control()
        else TacticalState.DEFENSIVE_TRANSITION
    )
    return PolicyInput(
        state=game_state,
        world=world,
        config=RuntimeConfig(
            genome=dict(genome),
            enable_mcts=enable_mcts,
            mcts_iterations=mcts_iterations,
        ),
        tactical_state=tactical,
        press_plan=PressPlan(),
        roles=dict(SLOT_ROLES),
        time_remaining=game_state.time_remaining,
        score_us=game_state.score_us,
        score_them=game_state.score_them,
        match_duration=300.0,
    )


def _to_light_intents(intents: dict[str, PlayerIntent]) -> dict[tuple[str, str], LIntent]:
    return {
        ("us", intent.pid): LIntent(
            tx=intent.tx,
            ty=intent.ty,
            speed=intent.speed,
            act=intent.action_type,
            action_target=intent.action_target,
            power=intent.action_power,
            face_target=(intent.face_x, intent.face_y),
        )
        for intent in intents.values()
    }


def _example_rows(records: list[dict]) -> list[dict]:
    disagreements = [row for row in records if not row["result"]["agreement"]]
    picks: list[tuple[str, dict | None]] = []
    picks.append((
        "largest_production_ev_difference",
        max(
            disagreements,
            key=lambda row: abs(row["result"]["production_ev_difference"] or 0.0),
            default=None,
        ),
    ))
    positive_preference = [
        row for row in disagreements if (row["result"]["mcts_preference"] or 0.0) > 0.0
    ]
    picks.append((
        "strongest_mcts_preference",
        max(positive_preference, key=lambda row: row["result"]["mcts_preference"], default=None),
    ))
    picks.append((
        "budget_unstable",
        next((row for row in disagreements if row["result"]["budget_stable"] is False), None),
    ))
    picks.append((
        "shoot_vs_non_shoot",
        next((
            row for row in disagreements
            if (row["result"]["production_action_type"] == "shoot")
            != (row["result"]["mcts_action_type"] == "shoot")
        ), None),
    ))
    picks.append((
        "pass_vs_carry",
        next((
            row for row in disagreements
            if {row["result"]["production_action_type"], row["result"]["mcts_action_type"]}
            == {"pass", "carry"}
        ), None),
    ))
    output: list[dict] = []
    seen: set[tuple[str, int, str]] = set()
    for category, row in picks:
        if row is None:
            continue
        key = (
            row.get("scenario", ""),
            row["sequence"],
            row["result"]["mcts_signature"] or "",
        )
        if key in seen:
            continue
        seen.add(key)
        output.append({"category": category, **row})
    return output


def _run_match(seed, opponent, decisions, budgets, scenario):
    rng = random.Random(seed)
    engine = LightEngine(seed=rng.randint(0, 2**31 - 1))
    state = (
        SCENARIOS[scenario](rng)
        if scenario != "match_start"
        else make_state(
            list(BASE_LINEUP),
            mirrored_lineup(),
            ball=(30.0, 20.0),
            possess=("us", "st"),
        )
    )
    if scenario == "down_a_goal_late":
        # This practice builder specifies the score/shape but leaves PlanState
        # time at zero; place it at the intended final 20% of the 300 s match.
        state.time = 240.0
    opponent_controller = OPPONENTS[opponent](rng)
    controller = PolicyController()
    game_id = f"shadow-{opponent}-{seed}"
    controller._current_game_id = game_id
    controller._team_plan_manager = TeamPlanManager()
    controller._last_sequence = None
    plan_manager = controller._team_plan_manager
    genome = default_genome()
    records: list[dict] = []
    skip_reasons: Counter[str] = Counter()

    for step in range(decisions):
        state.time += DECISION_INTERVAL
        inp = _policy_input(state, game_id, step, genome)

        # Preview TeamPlan lifecycle on a branch so the evaluator sees the plan
        # the ordinary decision will use, without updating live policy memory.
        preview_manager = copy.deepcopy(plan_manager)
        preview_manager.update(inp, inp.state.our_possessor(), inp.state.simulation_tick)
        result: ShadowResult = evaluate_shadow(
            inp,
            controller,
            preview_manager.current_plan,
            budgets=budgets,
        )
        # This is the only decision executed by the sample match. It remains
        # the ordinary deterministic production PolicyController output.
        production_intents = controller.decide(inp)
        if result.eligible:
            production_intent = production_intents.get(result.possession_player or "")
            pipeline_match = (
                production_intent is not None
                and candidate_signature(
                    ActionCandidate(0.0, production_intent, "")
                )
                == result.production_signature
            )
            records.append({
                "scenario": scenario,
                "sequence": step,
                "policy_pipeline_match": pipeline_match,
                "policy_action_type": production_intent.action_type if production_intent else None,
                "result": result.to_dict(),
            })
        elif result.skip_reason:
            skip_reasons[result.skip_reason] += 1
        merged = _to_light_intents(production_intents)
        merged.update(opponent_controller.decide(state, "them"))
        state = engine.step(state, merged)

    return records, skip_reasons


def run_shadow_sample(
    *,
    seed: int = 7,
    opponent: str = "possession",
    decisions: int = 120,
    budgets: tuple[int, ...] = DEFAULT_BUDGETS,
    scenarios: tuple[str, ...] | None = None,
) -> dict:
    """Run bounded ordinary-policy matches from match-start/practice states."""
    if not 1 <= decisions <= 5000:
        raise ValueError("decisions must be between 1 and 5000 per scenario")
    if not budgets or len(budgets) > MAX_SHADOW_BUDGETS or any(
        budget < 0 or budget > MAX_SHADOW_ITERATIONS for budget in budgets
    ):
        raise ValueError(
            f"supply 1..{MAX_SHADOW_BUDGETS} budgets between 0 and "
            f"{MAX_SHADOW_ITERATIONS} iterations"
        )
    if opponent not in OPPONENTS:
        raise ValueError(f"unknown simulator opponent: {opponent}")
    selected_scenarios = tuple(scenarios or ("match_start",))
    if not selected_scenarios or any(
        scenario != "match_start" and scenario not in SCENARIOS
        for scenario in selected_scenarios
    ):
        raise ValueError("scenarios must be match_start or a registered practice scenario")
    if decisions * len(selected_scenarios) > 5000:
        raise ValueError("total scenario decision states are capped at 5000")

    records: list[dict] = []
    skip_reasons: Counter[str] = Counter()
    for scenario_index, scenario in enumerate(selected_scenarios):
        scenario_records, scenario_skips = _run_match(
            seed + scenario_index, opponent, decisions, budgets, scenario
        )
        records.extend(scenario_records)
        skip_reasons.update(scenario_skips)

    eligible = [row["result"] for row in records]
    disagreements = [row for row in eligible if not row["agreement"]]
    stable_rows = [row for row in eligible if row["budget_stable"] is not None]
    candidates = [row["root_candidate_count"] for row in eligible]
    nodes = [row["nodes_created"] for row in eligible]
    steps = [row["simulated_steps"] for row in eligible]
    pipeline_matches = [row["policy_pipeline_match"] for row in records]
    production_types = Counter(row["production_action_type"] for row in disagreements)
    mcts_types = Counter(row["mcts_action_type"] for row in disagreements)
    production_actions = Counter(row["production_action_type"] for row in eligible)
    mcts_actions = Counter(row["mcts_action_type"] for row in eligible)
    ev_gaps = [
        abs(row["production_ev_difference"])
        for row in disagreements
        if row["production_ev_difference"] is not None
    ]
    preferences = [
        row["mcts_preference"]
        for row in disagreements
        if row["mcts_preference"] is not None
    ]
    zones: Counter[str] = Counter()
    for row in disagreements:
        zone = {
            "own": "defensive third",
            "mid": "middle third",
            "final": "final third",
        }.get(row["pitch_zone"], "unknown")
        zones[zone] += 1

    evaluated = len(eligible)
    summary = {
        "decision_states": decisions * len(selected_scenarios),
        "mcts_eligible_states": evaluated,
        "production_pipeline_match_count": sum(pipeline_matches),
        "production_pipeline_match_rate": round(
            sum(pipeline_matches) / max(1, len(pipeline_matches)), 4
        ),
        "production_pipeline_mismatch_count": sum(not matched for matched in pipeline_matches),
        "agreement_count": sum(1 for row in eligible if row["agreement"]),
        "agreement_rate": round(
            sum(1 for row in eligible if row["agreement"]) / max(1, evaluated), 4
        ),
        "production_action_counts": dict(sorted(production_actions.items())),
        "mcts_action_counts": dict(sorted(mcts_actions.items())),
        "disagreement_count": len(disagreements),
        "disagreement_rate": round(len(disagreements) / max(1, evaluated), 4),
        "disagreement_by_production_action": dict(sorted(production_types.items())),
        "disagreement_by_mcts_action": dict(sorted(mcts_types.items())),
        "disagreement_by_pitch_zone": dict(sorted(zones.items())),
        "production_ev_gap_on_disagreements": {
            "average_absolute": round(statistics.mean(ev_gaps), 6) if ev_gaps else 0.0,
            "max_absolute": round(max(ev_gaps), 6) if ev_gaps else 0.0,
        },
        "mcts_preference_on_disagreements": {
            "average": round(statistics.mean(preferences), 6) if preferences else 0.0,
            "max": round(max(preferences), 6) if preferences else 0.0,
        },
        "candidate_count": {
            "min": min(candidates) if candidates else 0,
            "median": statistics.median(candidates) if candidates else 0,
            "average": round(statistics.mean(candidates), 3) if candidates else 0.0,
            "max": max(candidates) if candidates else 0,
        },
        "budget_stable_count": sum(1 for row in stable_rows if row["budget_stable"]),
        "budget_stability_rate": round(
            sum(1 for row in stable_rows if row["budget_stable"]) / max(1, len(stable_rows)),
            4,
        ),
        "budget_unstable_count": sum(1 for row in stable_rows if not row["budget_stable"]),
        "nodes_created": {
            "average": round(statistics.mean(nodes), 3) if nodes else 0.0,
            "max": max(nodes) if nodes else 0,
        },
        "simulated_steps": {
            "average": round(statistics.mean(steps), 3) if steps else 0.0,
            "max": max(steps) if steps else 0,
        },
        "ineligible_by_reason": dict(sorted(skip_reasons.items())),
        "opponent_model": OPPONENT_RESPONSE_MODEL,
        "opponent_model_description": (
            "Branch-local deterministic PressBot carrier pressure, physics-based "
            "loose-ball/pass pursuit, and goal-side outfield shape; LightEngine "
            "retains goalkeeper control. This is not adversarial minimax."
        ),
    }
    return {
        "kind": "mcts_shadow",
        "seed": seed,
        "opponent": opponent,
        "scenarios": list(selected_scenarios),
        "decisions": decisions,
        "budgets": list(budgets),
        "mcts_limits": {
            "horizon_seconds": 1.5,
            "max_decision_depth": 3,
            "max_advance_steps": 8,
            "max_iterations_per_budget": 256,
        },
        "summary": summary,
        "examples": _example_rows(records),
    }


def _run_active_ab_match(
    *, seed: int, opponent: str, decisions: int, scenario: str,
    active_mcts: bool, iterations: int,
) -> dict:
    """Run one seeded baseline or active-selector LightEngine scenario."""
    rng = random.Random(seed)
    engine = LightEngine(seed=rng.randint(0, 2**31 - 1))
    state = (
        SCENARIOS[scenario](rng)
        if scenario != "match_start"
        else make_state(
            list(BASE_LINEUP), mirrored_lineup(), ball=(30.0, 20.0),
            possess=("us", "st"),
        )
    )
    if scenario == "down_a_goal_late":
        state.time = 240.0
    opponent_controller = OPPONENTS[opponent](rng)
    controller = PolicyController()
    game_id = f"active-ab-{scenario}-{seed}-{'mcts' if active_mcts else 'baseline'}"
    controller._current_game_id = game_id
    controller._team_plan_manager = TeamPlanManager()
    controller._last_sequence = None

    counts: Counter[str] = Counter()
    mcts_actions: Counter[str] = Counter()
    fallback_reasons: Counter[str] = Counter()
    eligible = disagreements = fallbacks = nodes_total = steps_total = 0
    candidates_total = depth_total = 0
    nodes_max = steps_max = candidates_max = depth_max = 0
    shots = passes = completed_passes = our_possession_ticks = 0
    shots_conceded = turnovers = 0
    decision_diagnostics: list[dict] = []

    for step in range(decisions):
        state.time += DECISION_INTERVAL
        if state.ball.possessing_team == "us":
            our_possession_ticks += 1
        inp = _policy_input(
            state,
            game_id,
            step,
            default_genome(),
            enable_mcts=active_mcts,
            mcts_iterations=iterations,
        )
        intents = controller.decide(inp)
        diagnostic = dict(controller.last_mcts_diagnostics)
        if active_mcts:
            decision_diagnostics.append(copy.deepcopy(diagnostic))
        for intent in intents.values():
            if state.ball.possessing_team == "us" and intent.pid == state.ball.possessing_player:
                counts[intent.action_type] += 1
                shots += intent.action_type == "shoot"
                passes += intent.action_type == "pass"

        if diagnostic.get("mcts_ran"):
            eligible += 1
            disagreements += diagnostic.get("agreement") is False
            action_type = diagnostic.get("mcts_action_type")
            if action_type:
                mcts_actions[str(action_type)] += 1
            nodes_value = diagnostic.get("nodes_created", 0)
            steps_value = diagnostic.get("simulated_steps", 0)
            candidates_value = diagnostic.get("candidate_count", 0)
            depth_value = diagnostic.get("max_depth_reached", 0)
            nodes = nodes_value if isinstance(nodes_value, int) else 0
            simulated = steps_value if isinstance(steps_value, int) else 0
            candidate_count = candidates_value if isinstance(candidates_value, int) else 0
            depth = depth_value if isinstance(depth_value, int) else 0
            nodes_total += nodes
            steps_total += simulated
            candidates_total += candidate_count
            depth_total += depth
            nodes_max = max(nodes_max, nodes)
            steps_max = max(steps_max, simulated)
            candidates_max = max(candidates_max, candidate_count)
            depth_max = max(depth_max, depth)
            fallback = diagnostic.get("fallback_reason")
            if fallback:
                fallbacks += 1
                fallback_reasons[str(fallback)] += 1

        merged = _to_light_intents(intents)
        opponent_intents = opponent_controller.decide(state, "them")
        if (
            state.ball.possessing_team == "them"
            and any(intent.act == "shoot" for intent in opponent_intents.values())
        ):
            shots_conceded += 1
        merged.update(opponent_intents)
        previous_possession = state.ball.possessing_team
        state = engine.step(state, merged)
        if previous_possession is None and state.ball.possessing_team == "us":
            completed_passes += 1
        if previous_possession == "us" and state.ball.possessing_team == "them":
            turnovers += 1

    score_us, score_them = state.score_us, state.score_them
    return {
        "score_us": score_us,
        "score_them": score_them,
        "goal_diff": score_us - score_them,
        "result": "win" if score_us > score_them else "loss" if score_us < score_them else "draw",
        "shots": shots,
        "shots_conceded": shots_conceded,
        "passes": passes,
        "completed_passes": completed_passes,
        "turnovers": turnovers,
        "possession%": round(100.0 * our_possession_ticks / max(1, decisions), 1),
        "possessor_action_counts": dict(sorted(counts.items())),
        "mcts": {
            "eligible_decisions": eligible,
            "disagreements": disagreements,
            "average_candidate_count": round(candidates_total / max(1, eligible), 3),
            "max_candidate_count": candidates_max,
            "action_type_counts": dict(sorted(mcts_actions.items())),
            "fallback_count": fallbacks,
            "fallback_reasons": dict(sorted(fallback_reasons.items())),
            "average_nodes": round(nodes_total / max(1, eligible), 3),
            "max_nodes": nodes_max,
            "average_simulated_steps": round(steps_total / max(1, eligible), 3),
            "max_simulated_steps": steps_max,
            "average_depth": round(depth_total / max(1, eligible), 3),
            "max_depth": depth_max,
            "decision_diagnostics": decision_diagnostics,
        },
    }


def run_active_ab_sample(
    *,
    seed: int = 7,
    opponent: str = "possession",
    decisions: int = 40,
    iterations: int = 8,
    scenarios: tuple[str, ...] = ("match_start", "final_third", "one_v_one_gk"),
) -> dict:
    """Compare paired ordinary and active-MCTS LightEngine runs."""
    if not 1 <= decisions <= 5000 or decisions * len(scenarios) > 5000:
        raise ValueError("active A/B decisions are capped at 5000 total")
    if not 1 <= iterations <= 64:
        raise ValueError("active MCTS iterations must be between 1 and 64")
    if opponent not in OPPONENTS:
        raise ValueError(f"unknown simulator opponent: {opponent}")
    if not scenarios or any(
        scenario != "match_start" and scenario not in SCENARIOS
        for scenario in scenarios
    ):
        raise ValueError("scenarios must be match_start or a registered practice scenario")

    paired = []
    for scenario_index, scenario in enumerate(scenarios):
        match_seed = seed + scenario_index
        baseline = _run_active_ab_match(
            seed=match_seed, opponent=opponent, decisions=decisions,
            scenario=scenario, active_mcts=False, iterations=iterations,
        )
        active = _run_active_ab_match(
            seed=match_seed, opponent=opponent, decisions=decisions,
            scenario=scenario, active_mcts=True, iterations=iterations,
        )
        paired.append(
            {
                "seed": match_seed,
                "scenario": scenario,
                "baseline": baseline,
                "active_mcts": active,
                "goal_diff_delta": active["goal_diff"] - baseline["goal_diff"],
            }
        )

    def aggregate(mode: str) -> dict:
        rows = [pair[mode] for pair in paired]
        mcts_rows = [row["mcts"] for row in rows]
        eligible_count = sum(int(row["eligible_decisions"]) for row in mcts_rows)
        return {
            "matches": len(rows),
            "runs": len(rows),
            "wins": sum(row["result"] == "win" for row in rows),
            "draws": sum(row["result"] == "draw" for row in rows),
            "losses": sum(row["result"] == "loss" for row in rows),
            "goals_for": sum(row["score_us"] for row in rows),
            "goals_against": sum(row["score_them"] for row in rows),
            "shots": sum(row["shots"] for row in rows),
            "shots_conceded": sum(row["shots_conceded"] for row in rows),
            "passes": sum(row["passes"] for row in rows),
            "completed_passes": sum(row["completed_passes"] for row in rows),
            "turnovers": sum(row["turnovers"] for row in rows),
            "average_possession_percent": round(
                statistics.mean(row["possession%"] for row in rows), 3
            ),
            "mcts_decisions": {
                "eligible": eligible_count,
                "disagreements": sum(row["disagreements"] for row in mcts_rows),
                "average_candidate_count": round(
                    sum(
                        float(row["average_candidate_count"]) * int(row["eligible_decisions"])
                        for row in mcts_rows
                    )
                    / max(1, eligible_count),
                    3,
                ),
                "max_candidate_count": max(
                    (row["max_candidate_count"] for row in mcts_rows), default=0
                ),
                "action_type_counts": dict(
                    sorted(sum_counters(row["action_type_counts"] for row in mcts_rows).items())
                ),
                "fallback_count": sum(row["fallback_count"] for row in mcts_rows),
                "fallback_reasons": dict(
                    sorted(
                        sum_counters(row["fallback_reasons"] for row in mcts_rows).items()
                    )
                ),
                "average_nodes": round(
                    sum(
                        float(row["average_nodes"]) * int(row["eligible_decisions"])
                        for row in mcts_rows
                    )
                    / max(1, eligible_count),
                    3,
                ),
                "max_nodes": max((row["max_nodes"] for row in mcts_rows), default=0),
                "average_simulated_steps": round(
                    sum(
                        float(row["average_simulated_steps"]) * int(row["eligible_decisions"])
                        for row in mcts_rows
                    )
                    / max(1, eligible_count),
                    3,
                ),
                "max_simulated_steps": max(
                    (row["max_simulated_steps"] for row in mcts_rows), default=0
                ),
                "average_depth": round(
                    sum(
                        float(row["average_depth"]) * int(row["eligible_decisions"])
                        for row in mcts_rows
                    )
                    / max(1, eligible_count),
                    3,
                ),
                "max_depth": max((row["max_depth"] for row in mcts_rows), default=0),
            },
        }

    return {
        "kind": "mcts_active_ab",
        "seed": seed,
        "seeds": [pair["seed"] for pair in paired],
        "opponent": opponent,
        "scenarios": list(scenarios),
        "decisions_per_match": decisions,
        "mcts_iterations": iterations,
        "baseline": aggregate("baseline"),
        "active_mcts": aggregate("active_mcts"),
        "paired_runs": paired,
    }


def sum_counters(rows) -> Counter[str]:
    total: Counter[str] = Counter()
    for row in rows:
        total.update(row)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("shadow", "active-ab"), default="shadow")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--opponent", choices=sorted(OPPONENTS), default="possession")
    parser.add_argument("--decisions", type=int, default=120)
    parser.add_argument("--budgets", type=int, nargs="+", default=list(DEFAULT_BUDGETS))
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument(
        "--scenario",
        choices=("match_start", *sorted(SCENARIOS)),
        nargs="+",
        help="one or more LightEngine practice starts (default: match_start)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="append one aggregate/replay NDJSON record (use '-' for stdout only)",
    )
    args = parser.parse_args()
    if args.mode == "active-ab":
        report = run_active_ab_sample(
            seed=args.seed,
            opponent=args.opponent,
            decisions=args.decisions,
            iterations=args.iterations,
            scenarios=(
                tuple(args.scenario)
                if args.scenario
                else ("match_start", "final_third", "one_v_one_gk")
            ),
        )
        report_summary = {
            "baseline": report["baseline"],
            "active_mcts": report["active_mcts"],
        }
        default_output = ARTIFACT_DIR / "mcts_active_ab.ndjson"
    else:
        report = run_shadow_sample(
            seed=args.seed,
            opponent=args.opponent,
            decisions=args.decisions,
            budgets=tuple(args.budgets),
            scenarios=tuple(args.scenario) if args.scenario else None,
        )
        report_summary = report["summary"]
        default_output = ARTIFACT_DIR / "mcts_shadow.ndjson"
    output = args.output or default_output
    if str(output) == "-":
        print(json.dumps(report, sort_keys=True, indent=2))
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n")
        print(json.dumps(report_summary, sort_keys=True, indent=2))
        print(f"Report appended to {output}")


if __name__ == "__main__":
    main()
