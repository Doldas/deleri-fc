"""Attacking shape measurement harness for TeamPlan baseline/validation.

This measures the key attacking coordination metrics specified in RULE 0.
Run against the authoritative engine via sim.py.
"""

from __future__ import annotations

import json
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import RuntimeConfig, default_genome
from src.policy import PolicyController, PolicyInput
from src.runtime import RuntimeManager
from src.sim import play_match, SimResult, plan_to_obs
from src.state import GameState, WorldModel
from src.tactics import (
    TacticalState, PressPlan, assign_roles, detect, plan_press,
    ROLE_DEFENDER, ROLE_WIDE_LEFT, ROLE_WIDE_RIGHT, ROLE_STRIKER
)
from src.geom import (
    GOAL_CENTER_Y,
    GOAL_HIGH_Y,
    GOAL_LOW_Y,
    OPP_GOAL_X,
    PITCH_LENGTH,
    PITCH_WIDTH,
)
from tests.test_policy import SLOTS

CFG = RuntimeConfig(genome=default_genome())


@dataclass
class PossessionSnapshot:
    tick: int
    ball_x: float
    ball_y: float
    carrier_id: str | None
    carrier_x: float | None
    carrier_y: float | None
    teammates_final_third: int
    teammates_in_box: int
    players_ahead_of_carrier: int
    players_behind_carrier: int
    team_spread_x: float
    team_spread_y: float
    off_ball_median_x: float
    off_ball_median_y: float


@dataclass
class Possession:
    start_tick: int
    end_tick: int
    snapshots: list[PossessionSnapshot] = field(default_factory=list)
    final_third_entries: int = 0
    box_entries: int = 0
    positive_ev_shots: int = 0
    shots: int = 0
    goals: int = 0
    turnovers: int = 0
    counterattacks_conceded: int = 0
    touches_before_shot: int = 0
    second_ball_recoveries: int = 0
    far_post_occupancy_when_shot: float = 0.0
    cutback_occupancy_when_shot: float = 0.0
    attacking_width: float = 0.0
    in_final_third: bool = False
    in_box: bool = False
    had_positive_ev_shot: bool = False
    ticks_to_final_third: int = 0
    ticks_to_box: int = 0
    ticks_to_positive_ev_shot: int = 0
    players_ahead_sum: int = 0
    players_behind_sum: int = 0
    snapshot_count: int = 0


@dataclass
class AttackMetrics:
    total_possessions: int = 0
    total_possession_ticks: int = 0
    total_final_third_snapshots: int = 0
    total_box_snapshots: int = 0
    zero_teammates_final_third: int = 0
    zero_teammates_box: int = 0
    carrier_x_values: list[float] = field(default_factory=list)
    off_ball_x_by_role: dict[str, list[float]] = field(default_factory=lambda: {"striker": [], "wide_left": [], "wide_right": [], "defender": []})
    team_longitudinal_spread_values: list[float] = field(default_factory=list)
    final_third_entries_per_possession: float = 0.0
    box_entries_per_possession: float = 0.0
    positive_ev_shots_per_possession: float = 0.0
    shots_per_possession: float = 0.0
    goals_per_possession: float = 0.0
    ticks_to_final_third: list[int] = field(default_factory=list)
    ticks_to_box: list[int] = field(default_factory=list)
    ticks_to_positive_ev_shot: list[int] = field(default_factory=list)
    duplicate_targets: int = 0
    players_ahead_of_carrier_total: float = 0.0
    players_behind_carrier_total: float = 0.0
    turnovers_by_region: dict[str, int] = field(default_factory=dict)
    counterattacks_conceded_total: int = 0
    touches_before_positive_ev_shot: list[int] = field(default_factory=list)
    second_ball_recoveries_total: int = 0
    far_post_occupancy_sum: float = 0.0
    cutback_occupancy_sum: float = 0.0
    attacking_width_sum: float = 0.0
    shot_events: int = 0
    possessions: list[Possession] = field(default_factory=list)

    def finalize(self):
        if self.total_possessions > 0:
            self.final_third_entries_per_possession = sum(
                p.final_third_entries for p in self.possessions
            ) / self.total_possessions
            self.box_entries_per_possession = sum(
                p.box_entries for p in self.possessions
            ) / self.total_possessions
            self.positive_ev_shots_per_possession = sum(
                p.positive_ev_shots for p in self.possessions
            ) / self.total_possessions
            self.shots_per_possession = sum(p.shots for p in self.possessions) / self.total_possessions
            self.goals_per_possession = sum(p.goals for p in self.possessions) / self.total_possessions
        if self.ticks_to_final_third:
            self.median_ticks_to_final_third = sorted(self.ticks_to_final_third)[len(self.ticks_to_final_third) // 2]
        if self.ticks_to_box:
            self.median_ticks_to_box = sorted(self.ticks_to_box)[len(self.ticks_to_box) // 2]
        if self.ticks_to_positive_ev_shot:
            self.median_ticks_to_positive_ev_shot = sorted(self.ticks_to_positive_ev_shot)[len(self.ticks_to_positive_ev_shot) // 2]

    def to_dict(self) -> dict[str, Any]:
        carrier_median = sorted(self.carrier_x_values)[len(self.carrier_x_values) // 2] if self.carrier_x_values else 0.0
        off_ball_median = {}
        for role, values in self.off_ball_x_by_role.items():
            if values:
                off_ball_median[role] = sorted(values)[len(values) // 2]
        team_spread = sum(self.team_longitudinal_spread_values) / max(1, len(self.team_longitudinal_spread_values))
        
        return {
            "total_possessions": self.total_possessions,
            "avg_possession_duration": self.total_possession_ticks / max(1, self.total_possessions),
            "teammates_in_final_third_pct": (
                (self.total_final_third_snapshots - self.zero_teammates_final_third)
                / max(1, self.total_final_third_snapshots)
            ),
            "snapshots_with_zero_teammates_final_third_pct": (
                self.zero_teammates_final_third / max(1, self.total_final_third_snapshots)
            ),
            "teammates_in_box_pct": (
                (self.total_box_snapshots - self.zero_teammates_box)
                / max(1, self.total_box_snapshots)
            ),
            "snapshots_with_zero_teammates_box_pct": (
                self.zero_teammates_box / max(1, self.total_box_snapshots)
            ),
            "carrier_median_x": carrier_median,
            "off_ball_median_x_by_role": off_ball_median,
            "team_longitudinal_spread": team_spread,
            "final_third_entries_per_possession": self.final_third_entries_per_possession,
            "box_entries_per_possession": self.box_entries_per_possession,
            "positive_ev_shots_per_possession": self.positive_ev_shots_per_possession,
            "shots_per_possession": self.shots_per_possession,
            "goals_per_possession": self.goals_per_possession,
            "median_ticks_to_final_third": getattr(self, "median_ticks_to_final_third", None),
            "median_ticks_to_box": getattr(self, "median_ticks_to_box", None),
            "median_ticks_to_positive_ev_shot": getattr(self, "median_ticks_to_positive_ev_shot", None),
            "duplicate_targets": self.duplicate_targets,
            "avg_players_ahead_of_carrier": self.players_ahead_of_carrier_total
            / max(1, self.total_possessions),
            "avg_players_behind_carrier": self.players_behind_carrier_total
            / max(1, self.total_possessions),
            "turnovers_by_region": self.turnovers_by_region,
            "counterattacks_conceded": self.counterattacks_conceded_total,
            "avg_touches_before_positive_ev_shot": (
                sum(self.touches_before_positive_ev_shot) / max(1, len(self.touches_before_positive_ev_shot))
                if self.touches_before_positive_ev_shot else None
            ),
            "second_ball_recoveries_per_possession": self.second_ball_recoveries_total
            / max(1, self.total_possessions),
            "avg_far_post_occupancy_when_shot": self.far_post_occupancy_sum / max(1, self.shot_events),
            "avg_cutback_occupancy_when_shot": self.cutback_occupancy_sum / max(1, self.shot_events),
            "avg_attacking_width": self.attacking_width_sum / max(1, self.total_possessions),
        }


def build_policy_input(state: GameState) -> PolicyInput:
    world = WorldModel.build(state)
    roles = assign_roles(state, SLOTS)
    tactical_state = detect(state, world, CFG.genome, False)
    press_plan = plan_press(state, world, CFG.genome, roles)
    return PolicyInput(
        state=state,
        world=world,
        config=CFG,
        tactical_state=tactical_state,
        press_plan=press_plan,
        roles=roles,
        time_remaining=state.time_remaining,
        match_duration=60.0,
    )


def is_in_final_third(x: float) -> bool:
    return x >= PITCH_LENGTH * 0.66  # x >= 39.6


def is_in_box(x: float, y: float) -> bool:
    # Opponent box: x >= 49 (goal at 60, box is 11m from goal line)
    # and y between goal posts (17 to 23)
    return x >= OPP_GOAL_X - 11.0 and GOAL_LOW_Y <= y <= GOAL_HIGH_Y


def compute_ev_shot(inp: PolicyInput, carrier_id: str) -> float:
    from src.policy import ExpectedValueCalculator, _shot_geometry_target, _shot_worth_taking
    carrier = next((p for p in inp.state.outfield_us() if p.id == carrier_id), None)
    if carrier is None or not carrier.can_act:
        return -1e9
    if not _shot_worth_taking(inp, carrier):
        return -1e9
    (tx, ty), power = _shot_geometry_target(inp, carrier)
    return ExpectedValueCalculator(inp).ev_shot(carrier, (tx, ty), power)


class AttackShapeMeasurer:
    def __init__(self, seed: int = 42, num_matches: int = 50):
        self.seed = seed
        self.num_matches = num_matches
        self.metrics = AttackMetrics()
        self.rng = random.Random(seed)

    def run_match(self, opponent: str = "reference") -> None:
        """Run one match and collect metrics."""
        manager = RuntimeManager()
        rng = random.Random(self.rng.randint(0, 2**31 - 1))
        
        # Play match and get trace
        result = play_match(
            manager=manager,
            opponent_kind=opponent,
            rng=rng,
            decisions=3600,  # 60s * 60Hz / 10Hz decision rate = 600 decisions, but sim runs at 60Hz
            collect_trace=True,
        )
        
        # Process the trace
        self._process_replay(result.trace)

    def _process_replay(self, replay: list) -> None:
        """Process a full match replay (PlanState trace) and extract attacking metrics."""
        if not replay:
            return

        current_possession: Possession | None = None
        prev_state: GameState | None = None
        
        for step, st in enumerate(replay):
            # Convert PlanState to observation, then to GameState
            obs = plan_to_obs(st, "measure", step)
            state = GameState.from_observation(obs)
            
            # Track possession changes
            has_control = state.has_control()
            
            # Check for shot events BEFORE updating possession state
            # Shots happen on the frame the ball is released (loose)
            if hasattr(st, 'events') and "kick:shoot" in st.events:
                if current_possession is not None:
                    current_possession.shots += 1
                    self.metrics.shot_events += 1
            
            if has_control and current_possession is None:
                # New possession started
                current_possession = Possession(
                    start_tick=step * 6,  # Each decision step = 6 engine ticks (0.1s at 60Hz)
                    end_tick=step * 6,
                )
            elif not has_control and current_possession is not None:
                # Possession ended
                current_possession.end_tick = step * 6
                self._finalize_possession(current_possession)
                current_possession = None
            
            if current_possession is not None:
                inp = build_policy_input(state)
                self._record_snapshot(current_possession, state, prev_state, inp, st, step)
            
            prev_state = state

        # Handle possession that ended at match end
        if current_possession is not None:
            current_possession.end_tick = len(replay) * 6
            self._finalize_possession(current_possession)

    def _record_snapshot(self, possession: Possession, state: GameState, prev_state: GameState | None, inp: PolicyInput, plan_state, step: int) -> None:
        ball = state.ball
        carrier = state.our_possessor()
        
        # Count teammates in final third and box
        teammates_final_third = 0
        teammates_in_box = 0
        off_ball_x = []
        off_ball_y = []
        players_ahead = 0
        players_behind = 0
        
        if carrier:
            for p in state.outfield_us():
                if p.id == carrier.id:
                    continue
                if is_in_final_third(p.x):
                    teammates_final_third += 1
                if is_in_box(p.x, p.y):
                    teammates_in_box += 1
                off_ball_x.append(p.x)
                off_ball_y.append(p.y)
                if p.x > carrier.x:
                    players_ahead += 1
                elif p.x < carrier.x:
                    players_behind += 1

        # Track first entry into final third / box
        any_in_final_third = any(is_in_final_third(p.x) for p in state.outfield_us())
        any_in_box = any(is_in_box(p.x, p.y) for p in state.outfield_us())
        
        if not possession.in_final_third and any_in_final_third:
            possession.in_final_third = True
            possession.final_third_entries += 1
            ticks_elapsed = step * 6 - possession.start_tick
            if ticks_elapsed > 0:
                self.metrics.ticks_to_final_third.append(ticks_elapsed)
        
        if not possession.in_box and any_in_box:
            possession.in_box = True
            possession.box_entries += 1
            ticks_elapsed = step * 6 - possession.start_tick
            if ticks_elapsed > 0:
                self.metrics.ticks_to_box.append(ticks_elapsed)

        # Check for positive EV shot opportunity
        if carrier:
            ev = compute_ev_shot(build_policy_input(state), carrier.id)
            if ev > 5.0:  # EV_SHOT_WORTH threshold
                possession.positive_ev_shots += 1
                if not possession.had_positive_ev_shot:
                    possession.had_positive_ev_shot = True
                    possession.ticks_to_positive_ev_shot = step * 6 - possession.start_tick
                    if possession.ticks_to_positive_ev_shot > 0:
                        self.metrics.ticks_to_positive_ev_shot.append(possession.ticks_to_positive_ev_shot)

        # Check for shot action from LightEngine events
        if hasattr(plan_state, 'events') and "kick:shoot" in plan_state.events:
            possession.shots += 1
            self.metrics.shot_events += 1
            
            # Measure far-post occupancy
            if carrier:
                far_post_y = GOAL_HIGH_Y if carrier.y < GOAL_CENTER_Y else GOAL_LOW_Y
                far_post_occupied = any(
                    abs(p.y - far_post_y) < 3.0 and p.x > 45.0
                    for p in state.outfield_us() if p.id != carrier.id
                )
                possession.far_post_occupancy_when_shot = 1.0 if far_post_occupied else 0.0
                self.metrics.far_post_occupancy_sum += possession.far_post_occupancy_when_shot
                
                # Measure cutback occupancy (central zone near box edge)
                cutback_occupied = any(
                    40.0 < p.x < 50.0 and 15.0 < p.y < 25.0
                    for p in state.outfield_us() if p.id != carrier.id
                )
                possession.cutback_occupancy_when_shot = 1.0 if cutback_occupied else 0.0
                self.metrics.cutback_occupancy_sum += possession.cutback_occupancy_when_shot

        # Measure attacking width (y-spread of attacking players)
        attackers = [p for p in state.outfield_us() if p.x > 30.0]
        if len(attackers) >= 2:
            ys = [p.y for p in attackers]
            width = max(ys) - min(ys)
            possession.attacking_width = width
            self.metrics.attacking_width_sum += width

        # Count duplicate targets (teammates with very similar targets)
        # This would need policy intents - approximate with positions for now
        if len(state.outfield_us()) >= 2:
            positions = [(p.x, p.y) for p in state.outfield_us()]
            for i, (x1, y1) in enumerate(positions):
                for x2, y2 in positions[i+1:]:
                    if math.hypot(x1 - x2, y1 - y2) < 2.0:
                        self.metrics.duplicate_targets += 1

        snapshot = PossessionSnapshot(
            tick=state.simulation_tick,
            ball_x=ball.x,
            ball_y=ball.y,
            carrier_id=carrier.id if carrier else None,
            carrier_x=carrier.x if carrier else None,
            carrier_y=carrier.y if carrier else None,
            teammates_final_third=teammates_final_third,
            teammates_in_box=teammates_in_box,
            players_ahead_of_carrier=players_ahead,
            players_behind_carrier=players_behind,
            team_spread_x=max((p.x for p in state.outfield_us()), default=0) - min((p.x for p in state.outfield_us()), default=0),
            team_spread_y=max((p.y for p in state.outfield_us()), default=0) - min((p.y for p in state.outfield_us()), default=0),
            off_ball_median_x=sorted(off_ball_x)[len(off_ball_x) // 2] if off_ball_x else 0.0,
            off_ball_median_y=sorted(off_ball_y)[len(off_ball_y) // 2] if off_ball_y else 0.0,
        )
        possession.snapshots.append(snapshot)
        
        # Accumulate metrics
        self.metrics.total_possession_ticks += 1
        if teammates_final_third >= 0:
            self.metrics.total_final_third_snapshots += 1
            if teammates_final_third == 0:
                self.metrics.zero_teammates_final_third += 1
        if teammates_in_box >= 0:
            self.metrics.total_box_snapshots += 1
            if teammates_in_box == 0:
                self.metrics.zero_teammates_box += 1
        
        if carrier:
            self.metrics.carrier_x_values.append(carrier.x)
        # Track per-possession averages for players ahead/behind
        if carrier:
            possession.players_ahead_sum += players_ahead
            possession.players_behind_sum += players_behind
            possession.snapshot_count += 1
        
        # Track off-ball positions by role
        for p in state.outfield_us():
            if p.id == carrier.id if carrier else False:
                continue
            role_key = inp.roles.get(p.id, "defender")
            if role_key == ROLE_STRIKER:
                self.metrics.off_ball_x_by_role["striker"].append(p.x)
            elif role_key == ROLE_WIDE_LEFT:
                self.metrics.off_ball_x_by_role["wide_left"].append(p.x)
            elif role_key == ROLE_WIDE_RIGHT:
                self.metrics.off_ball_x_by_role["wide_right"].append(p.x)
            else:
                self.metrics.off_ball_x_by_role["defender"].append(p.x)
        
        # Track team longitudinal spread
        xs = [p.x for p in state.outfield_us()]
        if xs:
            self.metrics.team_longitudinal_spread_values.append(max(xs) - min(xs))

    def _finalize_possession(self, possession: Possession) -> None:
        self.metrics.possessions.append(possession)
        self.metrics.total_possessions += 1
        duration = possession.end_tick - possession.start_tick
        self.metrics.total_possession_ticks += duration
        
        # Compute per-possession averages
        if possession.snapshot_count > 0:
            self.metrics.players_ahead_of_carrier_total += possession.players_ahead_sum / possession.snapshot_count
            self.metrics.players_behind_carrier_total += possession.players_behind_sum / possession.snapshot_count
        
        # Track turnovers by region
        # Simplified: check where possession ended
        if possession.snapshots:
            last_snap = possession.snapshots[-1]
            if last_snap.ball_x < 20:
                region = "own_third"
            elif last_snap.ball_x < 40:
                region = "middle_third"
            else:
                region = "final_third"
            self.metrics.turnovers_by_region[region] = self.metrics.turnovers_by_region.get(region, 0) + 1

    def run_all(self) -> AttackMetrics:
        opponents = ["possession", "press", "direct", "defensive", "counter", "parkbus", "wall"]
        for i in range(self.num_matches):
            opp = opponents[i % len(opponents)]
            print(f"Match {i+1}/{self.num_matches} vs {opp}...")
            self.run_match(opp)
        return self.metrics


def main():
    print("=" * 60)
    print("ATTACKING SHAPE MEASUREMENT - BEFORE TeamPlan")
    print("=" * 60)
    
    measurer = AttackShapeMeasurer(seed=20261001, num_matches=20)
    metrics = measurer.run_all()
    metrics.finalize()
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    results = metrics.to_dict()
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    # Save to file
    output_path = Path(__file__).parent / "attack_metrics_before.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()