"""Deterministic scenario corpus for skill-focused training.

Full matches collapse into a single kickoff/first-transition event (every real
60-game engine battery comes back 1-0 or 0-1). The scenario corpus replays
isolated, high-leverage situations against scripted opponents so a genome is
scored on skill rather than on one lucky event (AISTRATEGI §33-36, §55).

Each scenario constructs a `PlanState` directly, so evaluation is fast,
deterministic, and exercises one decision type at a time:

- final_third      : break down a packed defence and finish under pressure
- one_v_one_gk     : convert a clean look at the keeper
- deep_block       : crack a parked bus and take the shot
- press_recovery   : win the ball back and clear
- down_a_goal_late : score with goal difference against us and time running out
- kickoff_defend   : survive their kickoff first touch
- counter_break    : track a losing transition and stay compact
- wall_attack      : deploy wall plays to the top corner

Runtime impact: this module is never imported by the decision path.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

from .config import CONTROLLED_BALL_AHEAD, REWARD_DEFAULTS
from .light import LIntent, LightEngine, PlanState, make_state
from .runtime import RuntimeManager
from .sim import BASE_LINEUP, OPPONENTS, plan_reward

_ME = list(BASE_LINEUP)


@dataclass
class ScenarioDef:
    """One training scenario: a labeled PlanState builder plus grading knobs."""

    name: str
    label: str
    ticks: int
    build: Callable[[random.Random], PlanState]
    defensive: bool = False
    bonus_shots: bool = False

    def __call__(self, rng: random.Random) -> PlanState:
        return self.build(rng)


@dataclass
class ScenarioResult:
    """Metrics from one scenario run; mirrors SimResult.summary() keys."""

    name: str
    score_us: int = 0
    score_them: int = 0
    shots: int = 0
    shots_conceded: int = 0
    passes: int = 0
    completed_passes: int = 0
    possession_ours: float = 0.0
    reward: float = 0.0

    def summary(self) -> dict:
        return {
            "score_us": self.score_us,
            "score_them": self.score_them,
            "goal_diff": self.score_us - self.score_them,
            "possession%": round(self.possession_ours * 100.0, 1),
            "shots": self.shots,
            "shots_conceded": self.shots_conceded,
            "passes": self.passes,
            "completed_passes": self.completed_passes,
            "reward": round(self.reward, 4),
        }

    def fitness(self, scene: ScenarioDef) -> float:
        """Scenario-specific fitness: shape reward + decisive outcomes.

        Shooting drills are rewarded for *attempting* quality shots (a genome
        that never shoots caps out). Defensive drills are rewarded for a clean
        sheet far more than for any possession stat.
        """
        f = self.reward
        f += 100.0 * (self.score_us - self.score_them)
        if scene.bonus_shots and self.shots > 0:
            f += 3.0
        if scene.defensive and self.score_them == 0:
            f += 25.0
        return f


def _base(rng: random.Random) -> PlanState:
    return make_state(list(_ME), list(BASE_LINEUP), ball=(30.0, 20.0), possess=("us", "st"))


def _set(st: PlanState, team: str, pid: str, x: float, y: float) -> None:
    p = st.player(team, pid)
    if p is not None:
        p.x, p.y = x, y


def _final_third(rng: random.Random) -> PlanState:
    st = _base(rng)
    _set(st, "us", "st", 50.0, 18.0)
    st.ball.possessing_team, st.ball.possessing_player = "us", "st"
    st.ball.x = 50.0 + CONTROLLED_BALL_AHEAD
    st.ball.y = 18.0
    _set(st, "them", "gk", 58.5, 20.0)
    _set(st, "them", "def", 53.0, 24.0)
    _set(st, "them", "left", 52.0, 13.0)
    _set(st, "them", "right", 46.0, 28.0)
    return st


def _one_v_one_gk(rng: random.Random) -> PlanState:
    st = _base(rng)
    _set(st, "us", "st", 54.0, 20.0)
    st.ball.possessing_team, st.ball.possessing_player = "us", "st"
    st.ball.x = 54.0 + CONTROLLED_BALL_AHEAD
    st.ball.y = 20.0
    _set(st, "them", "gk", 58.5, 20.0)
    _set(st, "them", "def", 3.0, 20.0)
    _set(st, "them", "left", 3.0, 8.0)
    _set(st, "them", "right", 3.0, 32.0)
    return st


def _deep_block(rng: random.Random) -> PlanState:
    st = _base(rng)
    _set(st, "us", "def", 30.0, 20.0)
    st.ball.possessing_team, st.ball.possessing_player = "us", "def"
    st.ball.x = 30.0 + CONTROLLED_BALL_AHEAD
    st.ball.y = 20.0
    _set(st, "them", "gk", 4.0, 20.0)
    _set(st, "them", "def", 17.0, 24.0)
    _set(st, "them", "left", 17.0, 16.0)
    _set(st, "them", "right", 17.0, 30.0)
    return st


def _press_recovery(rng: random.Random) -> PlanState:
    st = _base(rng)
    _set(st, "them", "st", 34.0, 20.0)
    st.ball.possessing_team, st.ball.possessing_player = "them", "st"
    st.ball.x = 34.0 + CONTROLLED_BALL_AHEAD
    st.ball.y = 20.0
    _set(st, "us", "def", 30.0, 20.0)
    _set(st, "us", "left", 28.0, 14.0)
    _set(st, "us", "right", 28.0, 26.0)
    _set(st, "us", "st", 42.0, 20.0)
    _set(st, "them", "gk", 56.0, 20.0)
    return st


def _down_a_goal_late(rng: random.Random) -> PlanState:
    st = make_state(list(_ME), list(BASE_LINEUP), ball=(28.0, 20.0), possess=("us", "def"), score=(0, 1))
    _set(st, "us", "def", 28.0, 20.0)
    st.ball.possessing_team, st.ball.possessing_player = "us", "def"
    st.ball.x = 28.0 + CONTROLLED_BALL_AHEAD
    st.ball.y = 20.0
    _set(st, "them", "gk", 58.5, 20.0)
    _set(st, "them", "def", 46.0, 22.0)
    _set(st, "them", "left", 44.0, 14.0)
    _set(st, "them", "right", 44.0, 28.0)
    return st


def _kickoff_defend(rng: random.Random) -> PlanState:
    st = _base(rng)
    _set(st, "them", "st", 30.0, 20.0)
    st.ball.possessing_team, st.ball.possessing_player = "them", "st"
    st.ball.x = 30.0 + CONTROLLED_BALL_AHEAD
    st.ball.y = 20.0
    st.ball.protection = 1.0
    _set(st, "us", "st", 34.0, 20.0)
    _set(st, "us", "def", 26.0, 20.0)
    _set(st, "us", "left", 30.0, 10.0)
    _set(st, "us", "right", 30.0, 30.0)
    _set(st, "them", "gk", 58.5, 20.0)
    return st


def _counter_break(rng: random.Random) -> PlanState:
    st = _base(rng)
    _set(st, "them", "st", 40.0, 20.0)
    st.ball.possessing_team, st.ball.possessing_player = "them", "st"
    st.ball.x = 40.0 + CONTROLLED_BALL_AHEAD
    st.ball.y = 20.0
    _set(st, "them", "left", 36.0, 12.0)
    _set(st, "them", "right", 36.0, 28.0)
    _set(st, "them", "def", 44.0, 16.0)
    _set(st, "us", "def", 30.0, 20.0)
    _set(st, "us", "left", 28.0, 12.0)
    _set(st, "us", "right", 28.0, 28.0)
    _set(st, "us", "st", 44.0, 20.0)
    _set(st, "them", "gk", 58.5, 20.0)
    return st


def _wall_attack(rng: random.Random) -> PlanState:
    st = _base(rng)
    _set(st, "us", "left", 40.0, 4.0)
    st.ball.possessing_team, st.ball.possessing_player = "us", "left"
    st.ball.x = 40.0 + CONTROLLED_BALL_AHEAD
    st.ball.y = 4.0
    _set(st, "them", "gk", 58.5, 20.0)
    _set(st, "them", "def", 48.0, 10.0)
    _set(st, "them", "right", 46.0, 24.0)
    return st


SCENARIOS: dict[str, ScenarioDef] = {
    "final_third": ScenarioDef("final_third", "Break the packed defence and finish", 40, _final_third, bonus_shots=True),
    "one_v_one_gk": ScenarioDef("one_v_one_gk", "Convert a clean 1v1 vs the keeper", 20, _one_v_one_gk, bonus_shots=True),
    "deep_block": ScenarioDef("deep_block", "Crack a parked bus", 80, _deep_block, bonus_shots=True),
    "press_recovery": ScenarioDef("press_recovery", "Win the ball back and clear", 30, _press_recovery, defensive=True),
    "down_a_goal_late": ScenarioDef("down_a_goal_late", "Score while trailing and late", 60, _down_a_goal_late, bonus_shots=True),
    "kickoff_defend": ScenarioDef("kickoff_defend", "Survive their kickoff first touch", 25, _kickoff_defend, defensive=True),
    "counter_break": ScenarioDef("counter_break", "Track a losing transition", 25, _counter_break, defensive=True),
    "wall_attack": ScenarioDef("wall_attack", "Attack off the touchline wall", 30, _wall_attack, bonus_shots=True),
}


def scenario_fitness(results: list[ScenarioResult], scenes: list[ScenarioDef]) -> float:
    return sum(r.fitness(s) for r, s in zip(results, scenes))


def run_scenario(
    manager: RuntimeManager | None,
    scenario_name: str,
    opponent: str,
    rng: random.Random,
    ticks: int | None = None,
    weights: dict[str, float] | None = None,
) -> ScenarioResult:
    """Run one scenario with `manager` controlling our side.

    `manager.decide` receives an engine-shaped observation per step; the
    scripted `opponent` controls the away side from the raw PlanState, exactly
    like `play_match`. Result carries the same summary keys as SimResult.
    """
    scene = SCENARIOS[scenario_name]
    engine = LightEngine(seed=rng.randint(0, 2**31 - 1))
    opp = OPPONENTS[opponent](rng)
    weights = weights or REWARD_DEFAULTS

    st = scene(rng)
    ticks = ticks or scene.ticks
    res = ScenarioResult(name=scenario_name)
    our_poss_ticks = 0
    for step in range(ticks):
        st.time += 0.1
        if st.ball.possessing_team == "us":
            our_poss_ticks += 1

        merged: dict[tuple[str, str], LIntent] = {}
        if manager is not None:
            merged.update(_intents_from_decision("us", manager.decide(_obs(st, step)), st))
        merged.update(opp.decide(st, "them"))
        prev_poss = st.ball.possessing_team
        st = engine.step(st, merged)
        for e in st.events:
            if e == "goal":
                res.score_us += 1
            elif e == "goal_conceded":
                res.score_them += 1
            elif e == "kick:shoot":
                res.shots += 1
            elif e == "kick:shoot_conceded":
                res.shots_conceded += 1
            elif e.startswith("kick:"):
                res.passes += 1
            elif e == "tackle:win":
                pass
        # Pass completion
        if prev_poss is None and st.ball.possessing_team == "us":
            res.completed_passes += 1
        res.reward += plan_reward(st, weights, st.events)

    res.possession_ours = our_poss_ticks / max(1, ticks)
    res.reward = float(res.reward)
    return res


def _obs(st: PlanState, step: int) -> dict:
    from .sim import plan_to_obs

    return plan_to_obs(st, "scenario", step)


def _intents_from_decision(team: str, decision: dict, st: PlanState) -> dict[tuple[str, str], LIntent]:
    merged: dict[tuple[str, str], LIntent] = {}
    for intent in decision.get("intents", []):
        pid = intent["playerId"]
        mv = intent.get("move", {})
        tgt = mv.get("target", {})
        act = intent.get("action", {})
        at = act.get("target")
        merged[(team, pid)] = LIntent(
            tx=tgt.get("x", st.ball.x),
            ty=tgt.get("y", st.ball.y),
            speed=mv.get("speed", 0.5),
            act=act.get("type", "none"),
            action_target=(at["x"], at["y"]) if at else None,
            power=act.get("power"),
        )
    return merged


def evaluate_scenarios(
    genome: dict[str, float],
    rng: random.Random,
    lab: list[tuple[str, str]],
    ticks: int | None = None,
    weights: dict[str, float] | None = None,
) -> tuple[list[ScenarioResult], float]:
    """Score a genome over a scenario lab: returns (results, fitness).

    A fresh RuntimeManager (with `genome`) is used per scenario so no match
    state carries over (§5). Deterministic for a seeded `rng`.
    """
    results: list[ScenarioResult] = []
    for scenario_name, opponent in lab:
        manager = RuntimeManager()
        manager.genome_base = dict(genome)
        results.append(run_scenario(manager, scenario_name, opponent, rng, ticks=ticks, weights=weights))
    scenes = [SCENARIOS[name] for name, _ in lab]
    return results, scenario_fitness(results, scenes)


__all__ = [
    "SCENARIOS",
    "ScenarioDef",
    "ScenarioResult",
    "run_scenario",
    "evaluate_scenarios",
    "scenario_fitness",
]