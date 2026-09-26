"""Evaluation — shaped tactical reward (AISTRATEGI §31–32, §45).

`reward()` scores a state with configurable weights. `compute_metrics()` records
aggregate tactical metrics for experiment tracking and the replay analyzer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .geom import PITCH_LENGTH
from .config import REWARD_DEFAULTS
from .state import GameState, WorldModel


def reward(
    state: GameState,
    world: WorldModel,
    weights: dict[str, float] | None = None,
    prev: dict | None = None,
) -> float:
    """Shaped reward for a (possibly hypothetical) state.

    `prev` may carry event flags produced by the simulator or MCTS forward step:
    goal, goal_conceded, won_lose possession, turnover, press success, bad shot.
    """
    w = weights or REWARD_DEFAULTS
    r = 0.0

    events = prev if prev is not None else {}
    if events.get("goal"):
        r += w["goal"]
    if events.get("goal_conceded"):
        r += w["goal_conceded"]
    if events.get("dangerous_turnover"):
        r += w["dangerous_turnover"]
    if events.get("broken_shape"):
        r += w["broken_shape"]
    if events.get("failed_press"):
        r += w["failed_press"]
    if events.get("bad_shot"):
        r += w["bad_shot"]

    # Territorial progression: how far up the pitch we control the ball.
    x_frac = state.ball.x / PITCH_LENGTH
    if state.has_control():
        r += (
            w["territorial_progression"] * (x_frac - 0.5) * PITCH_LENGTH
            + w["possession_quality"] * world.our_control * 10.0
        )
    else:
        r += w["territorial_progression"] * (x_frac - 0.5) * PITCH_LENGTH * 0.5

    # Space created & defensive compactness.
    compact = _compactness(state, world)
    r += w["defensive_compactness"] * compact * 5.0
    r += w["space_created"] * max(0.0, world.our_control) * 4.0

    # Unnecessary risk penalty: possession far down the pitch while opponents
    # press tightly.
    if state.has_control() and world.pressure_on_ball > 0.7 and state.ball.x < 12.0:
        r += w["unnecessary_risk"] * 0.5

    return r


def _compactness(state: GameState, world: WorldModel) -> float:
    """0..1: how tight and goal-oriented our defensive unit is.

    Both for shape quality (when we defend) and rest-defence discipline (when
    we attack)."""
    ours = state.outfield_us()
    if len(ours) < 2:
        return 1.0
    xs = [p.x for p in ours]
    ys = [p.y for p in ours]
    spread_x = max(xs) - min(xs)
    spread_y = max(ys) - min(ys)
    if not state.has_control():
        # Defensive compactness measured around the ball zone.
        d_max = 10.0
    else:
        d_max = 18.0
    compact = 1.0 - min(1.0, (spread_x + spread_y) / (2.0 * d_max))
    return max(0.0, compact)


@dataclass
class Metrics:
    possession_ticks_ours: int = 0
    possession_ticks_theirs: int = 0
    shots: int = 0
    shots_conceded: int = 0
    goals: int = 0
    goals_conceded: int = 0
    passes: int = 0
    completed_passes: int = 0
    turnovers: int = 0
    press_attempts: int = 0
    presses_won: int = 0
    counterpress_attempts: int = 0
    counterpress_recoveries: int = 0
    wall_passes: int = 0
    wall_passes_completed: int = 0
    wall_shots: int = 0
    width_accum: float = 0.0
    depth_accum: float = 0.0
    ticks: int = 0
    events: list[dict] = field(default_factory=list)

    def observe(self, state: GameState, world: WorldModel) -> None:
        self.ticks += 1
        if state.has_control():
            self.possession_ticks_ours += 1
        elif state.ball.possessing_team == "them":
            self.possession_ticks_theirs += 1
        xs = [p.x for p in state.outfield_us()]
        ys = [p.y for p in state.outfield_us()]
        if len(xs) >= 2:
            self.width_accum += max(ys) - min(ys)
            self.depth_accum += max(xs) - min(xs)

    def record(self, event: dict) -> None:
        kind = event.get("type")
        self.events.append(event)
        if kind == "shot":
            self.shots += 1
        elif kind == "shot_conceded":
            self.shots_conceded += 1
        elif kind == "goal":
            self.goals += 1
        elif kind == "goal_conceded":
            self.goals_conceded += 1
        elif kind == "pass":
            self.passes += 1
        elif kind == "pass_complete":
            self.completed_passes += 1
        elif kind == "turnover":
            self.turnovers += 1
        elif kind == "press":
            self.press_attempts += 1
        elif kind == "press_win":
            self.presses_won += 1
        elif kind == "counterpress":
            self.counterpress_attempts += 1
        elif kind == "counterpress_recovery":
            self.counterpress_recoveries += 1
        elif kind == "wall_pass":
            self.wall_passes += 1
        elif kind == "wall_pass_complete":
            self.wall_passes_completed += 1
        elif kind == "wall_shot":
            self.wall_shots += 1

    def summary(self) -> dict:
        total = max(1, self.possession_ticks_ours + self.possession_ticks_theirs)
        return {
            "possession%": round(100.0 * self.possession_ticks_ours / total, 2),
            "shots": self.shots,
            "shots_conceded": self.shots_conceded,
            "goals": self.goals,
            "goals_conceded": self.goals_conceded,
            "passes": self.passes,
            "completed_passes": self.completed_passes,
            "turnovers": self.turnovers,
            "press_attempts": self.press_attempts,
            "presses_won": self.presses_won,
            "counterpress_attempts": self.counterpress_attempts,
            "counterpress_recoveries": self.counterpress_recoveries,
            "wall_passes": self.wall_passes,
            "wall_passes_completed": self.wall_passes_completed,
            "wall_shots": self.wall_shots,
            "avg_width": round(self.width_accum / max(1, self.ticks), 2),
            "avg_depth": round(self.depth_accum / max(1, self.ticks), 2),
        }