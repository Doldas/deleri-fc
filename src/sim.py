"""Self-play / self-evaluation simulator on top of the LightEngine.

`SimMatch` runs a full or shortened match at decision cadence (0.1 s) with two
controllers: our runtime brain (via RuntimeManager.decide) and a built-in
scripted opponent. Events are converted to engine-observation form so all
reward / metric code paths reuse the same definitions (AGENTS.md §7).

Built-in opponents mirror the archetypes from AISTRATEGI §24/§62: possession,
direct, defensive, pressing, and random.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Protocol
from dataclasses import dataclass, field

from . import geom
from .config import DECISION_INTERVAL, REWARD_DEFAULTS
from .evaluate import reward
from .light import LIntent, LightEngine, PlanState, make_state
from .runtime import RuntimeManager
from .state import GameState, WorldModel

# Shortened default: 120 s of football (~1200 decisions) keeps evolution fast
# while still exercising most tactical states.
DEFAULT_DECISIONS = 1200

ROLE_GK = "goalkeeper"
ROLE_OF = "outfield"

# Base line-up for the home team (attack toward +x), mirroring the real 5v5
# format: 1 goalkeeper + 4 outfield (defender, left, right, striker per
# tactics.json). make_state reads each row as (pid, role, x, y). Mirror the
# x-coordinate for the away side.
BASE_LINEUP: tuple[tuple[str, str, float, float], ...] = (
    ("gk", ROLE_GK, 6.0, 20.0),
    ("def", ROLE_OF, 16.0, 20.0),
    ("left", ROLE_OF, 26.0, 8.0),
    ("right", ROLE_OF, 26.0, 32.0),
    ("st", ROLE_OF, 42.0, 20.0),
)


def mirrored_lineup() -> list[tuple[str, str, float, float]]:
    return [(r, i, geom.PITCH_LENGTH - x, y) for (r, i, x, y) in BASE_LINEUP]


@dataclass
class SimResult:
    score_us: int = 0
    score_them: int = 0
    possession_ours: float = 0.0
    shots: int = 0
    shots_conceded: int = 0
    goals: int = 0
    goals_conceded: int = 0
    passes: int = 0
    completed_passes: int = 0
    rewards: float = 0.0
    events: list[str] = field(default_factory=list)
    trace: list[PlanState] = field(default_factory=list)

    @property
    def goal_diff(self) -> int:
        return self.goals - self.goals_conceded

    def summary(self) -> dict:
        return {
            "score_us": self.goals,
            "score_them": self.goals_conceded,
            "goal_diff": self.goal_diff,
            "possession%": round(self.possession_ours * 100.0, 1),
            "shots": self.shots,
            "shots_conceded": self.shots_conceded,
            "passes": self.passes,
            "completed_passes": self.completed_passes,
            "reward": round(self.rewards, 4),
        }


class OpponentController:
    """Scripted archetype controllers; return a dict of per-player intents."""

    kind = "base"

    def __init__(self, rng: random.Random):
        self.rng = rng

    def decide(self, st: PlanState, team: str) -> dict[tuple[str, str], LIntent]:
        raise NotImplementedError


class PossessionBot(OpponentController):
    kind = "possession"

    def decide(self, st: PlanState, team: str) -> dict[tuple[str, str], LIntent]:
        out: dict[tuple[str, str], LIntent] = {}
        bx, by = st.ball.x, st.ball.y
        goal_x = 60.0 if team == "us" else 0.0
        for p in st.players:
            if p.team != team:
                continue
            if p.role == ROLE_GK:
                out[(team, p.pid)] = LIntent(tx=p.x, ty=p.y, speed=0.5)
                continue
            if st.ball.possessing_team == team and st.ball.possessing_player == p.pid:
                # Shoot if close to goal
                if abs(p.x - goal_x) <= 15.0:
                    out[(team, p.pid)] = LIntent(tx=goal_x, ty=by, speed=0.5, act="shoot", action_target=(goal_x, 18.0 if by >= 20.0 else 22.0), power=0.9)
                else:
                    # Keep the ball: probe a forward pass
                    mates = [q for q in st.outfield(team) if q.pid != p.pid]
                    if mates:
                        t = max(mates, key=lambda q: q.x)
                        out[(team, p.pid)] = LIntent(tx=t.x, ty=t.y, speed=0.6, act="pass", action_target=(t.x, t.y), power=0.5)
                        out[(team, t.pid)] = LIntent(tx=t.x, ty=t.y, speed=0.8)
            elif geom.distance(p.x, p.y, bx, by) < 5.0:
                out[(team, p.pid)] = LIntent(tx=bx, ty=by, speed=0.85)
            else:
                out[(team, p.pid)] = LIntent(tx=p.x, ty=p.y, speed=0.5)
        return out


class PressBot(OpponentController):
    kind = "press"

    def decide(self, st: PlanState, team: str) -> dict[tuple[str, str], LIntent]:
        out: dict[tuple[str, str], LIntent] = {}
        bx, by = st.ball.x, st.ball.y
        field_players = [p for p in st.players if p.team == team and p.role != ROLE_GK]
        if not field_players:
            return out
        # Check if we can tackle the ball holder
        if st.ball.possessing_team and st.ball.possessing_team != team:
            holder = st.player(st.ball.possessing_team, st.ball.possessing_player or "")
            if holder:
                presser = min(field_players, key=lambda p: geom.distance(p.x, p.y, holder.x, holder.y))
                if geom.distance(presser.x, presser.y, holder.x, holder.y) <= 1.5:
                    out[(team, presser.pid)] = LIntent(tx=holder.x, ty=holder.y, speed=1.0, act="tackle")
                    # Other players mark
                    for p in field_players:
                        if p.pid != presser.pid:
                            tx = geom.clamp(p.x + (bx - p.x) * 0.2, 0.0, 60.0)
                            ty = geom.clamp(p.y + (by - p.y) * 0.2, 0.0, 40.0)
                            out[(team, p.pid)] = LIntent(tx=tx, ty=ty, speed=0.7)
                    return out
        # Standard press
        presser = min(field_players, key=lambda p: geom.distance(p.x, p.y, bx, by))
        for p in field_players:
            if p.pid == presser.pid:
                out[(team, p.pid)] = LIntent(tx=bx, ty=by, speed=1.0)
            else:
                tx = geom.clamp(p.x + (bx - p.x) * 0.2, 0.0, 60.0)
                ty = geom.clamp(p.y + (by - p.y) * 0.2, 0.0, 40.0)
                out[(team, p.pid)] = LIntent(tx=tx, ty=ty, speed=0.7)
        return out


class DirectBot(OpponentController):
    kind = "direct"

    def decide(self, st: PlanState, team: str) -> dict[tuple[str, str], LIntent]:
        out: dict[tuple[str, str], LIntent] = {}
        strike_x = 60.0 if team == "us" else 0.0
        bx, by = st.ball.x, st.ball.y
        for p in st.players:
            if p.team != team:
                continue
            if p.role == ROLE_GK:
                out[(team, p.pid)] = LIntent(tx=p.x, ty=p.y, speed=0.5)
                continue
            if st.ball.possessing_team == team and st.ball.possessing_player == p.pid:
                stricker = max(
                    [q for q in st.outfield(team) if "st" in q.pid or q.x > 45.0],
                    key=lambda q: q.x,
                    default=None,
                )
                if stricker is not None:
                    out[(team, p.pid)] = LIntent(tx=stricker.x, ty=stricker.y, speed=0.5, act="pass", action_target=(stricker.x, stricker.y), power=0.8)
                    out[(team, stricker.pid)] = LIntent(tx=stricker.x + 2.0, ty=stricker.y, speed=0.8)
                elif abs(p.x - strike_x) <= 18.0:
                    out[(team, p.pid)] = LIntent(tx=strike_x, ty=by, speed=0.5, act="shoot", action_target=(strike_x, 18.0 if by >= 20.0 else 22.0), power=0.9)
                else:
                    out[(team, p.pid)] = LIntent(tx=strike_x, ty=by, speed=0.6)
            elif p.x < 30.0 and p.role != ROLE_GK:
                out[(team, p.pid)] = LIntent(tx=p.x, ty=p.y, speed=0.4)
            else:
                out[(team, p.pid)] = LIntent(tx=p.x, ty=p.y, speed=0.5)
        return out


class DefensiveBot(OpponentController):
    kind = "defensive"

    def decide(self, st: PlanState, team: str) -> dict[tuple[str, str], LIntent]:
        out: dict[tuple[str, str], LIntent] = {}
        own_goal_x = 60.0 if team == "us" else 0.0
        bx, by = st.ball.x, st.ball.y
        field_players = [p for p in st.players if p.team == team and p.role != ROLE_GK]
        for p in st.players:
            if p.team != team:
                continue
            if p.role == ROLE_GK:
                out[(team, p.pid)] = LIntent(tx=p.x, ty=p.y, speed=0.5)
                continue
            # Tackle if opponent has ball and is close
            if st.ball.possessing_team and st.ball.possessing_team != team:
                holder = st.player(st.ball.possessing_team, st.ball.possessing_player or "")
                if holder and geom.distance(p.x, p.y, holder.x, holder.y) <= 1.5:
                    out[(team, p.pid)] = LIntent(tx=holder.x, ty=holder.y, speed=1.0, act="tackle")
                    continue
            if st.ball.possessing_team == team and st.ball.possessing_player == p.pid:
                out[(team, p.pid)] = LIntent(tx=own_goal_x, ty=p.y, speed=0.4, act="clear", action_target=(own_goal_x * 0.5, p.y), power=0.9)
            else:
                goal_side = geom.clamp(bx + (own_goal_x - bx) * 0.35, 8.0, 52.0)
                out[(team, p.pid)] = LIntent(tx=goal_side if p.x > bx else p.x, ty=geom.clamp(p.y, 4.0, 36.0), speed=0.7)
        return out


class RandomBot(OpponentController):
    kind = "random"

    def decide(self, st: PlanState, team: str) -> dict[tuple[str, str], LIntent]:
        out: dict[tuple[str, str], LIntent] = {}
        for p in st.players:
            if p.team != team:
                continue
            tx = self.rng.uniform(0.0, 60.0)
            ty = self.rng.uniform(0.0, 40.0)
            act = "none"
            at = None
            if st.ball.possessing_team == team and st.ball.possessing_player == p.pid:
                act = self.rng.choice(("pass", "shoot", "clear"))
                at = (self.rng.uniform(0.0, 60.0), self.rng.uniform(0.0, 40.0))
            out[(team, p.pid)] = LIntent(tx=tx, ty=ty, speed=self.rng.uniform(0.3, 1.0), act=act, action_target=at, power=self.rng.uniform(0.3, 1.0))
        return out


class CounterBot(OpponentController):
    """Fast-transition archetype: deep block, then an all-out vertical sprint
    once possession is won (mirrors the reference-strikers killer breakaway)."""

    kind = "counter"

    def decide(self, st: PlanState, team: str) -> dict[tuple[str, str], LIntent]:
        out: dict[tuple[str, str], LIntent] = {}
        goal_x = 60.0 if team == "us" else 0.0
        mid_x = 30.0 if team == "us" else 30.0
        bx, by = st.ball.x, st.ball.y
        field_players = [p for p in st.players if p.team == team and p.role != ROLE_GK]
        for p in st.players:
            if p.team != team:
                continue
            if p.role == ROLE_GK:
                out[(team, p.pid)] = LIntent(tx=p.x, ty=p.y, speed=0.5)
                continue
            holder = st.ball.possessing_team == team and st.ball.possessing_player == p.pid
            if holder:
                # Shoot if close, else sprint forward
                if abs(p.x - goal_x) <= 15.0:
                    out[(team, p.pid)] = LIntent(tx=goal_x, ty=by, speed=0.5, act="shoot", action_target=(goal_x, 18.0 if by >= 20.0 else 22.0), power=0.9)
                else:
                    out[(team, p.pid)] = LIntent(tx=goal_x, ty=by, speed=1.0)
                continue
            if st.ball.possessing_team == team:
                # Transition: everyone sprints forward, not to the ball.
                out[(team, p.pid)] = LIntent(tx=goal_x, ty=geom.clamp(by + (p.y - by) * 0.4, 4.0, 36.0), speed=1.0)
                continue
            # No possession: goal-side block around the midfield line.
            # Tackle if ball carrier comes close
            if st.ball.possessing_team and st.ball.possessing_team != team:
                holder = st.player(st.ball.possessing_team, st.ball.possessing_player or "")
                if holder and geom.distance(p.x, p.y, holder.x, holder.y) <= 1.5:
                    out[(team, p.pid)] = LIntent(tx=holder.x, ty=holder.y, speed=1.0, act="tackle")
                    continue
            tx = mid_x if p.x < mid_x else p.x
            ty = geom.clamp(p.y + (by - p.y) * 0.1, 4.0, 36.0)
            out[(team, p.pid)] = LIntent(tx=tx, ty=ty, speed=0.7)
        return out


class ParkBusBot(OpponentController):
    """Low-block archetype: four outfields sit in a tight zonal line, only
    clearing safely. Starves breakaway lanes and punishes risk-taking."""

    kind = "parkbus"

    def decide(self, st: PlanState, team: str) -> dict[tuple[str, str], LIntent]:
        out: dict[tuple[str, str], LIntent] = {}
        own_goal_x = 60.0 if team == "us" else 0.0
        line_x = 18.0 if team == "us" else 42.0
        zone_y = {0: 10.0, 1: 17.0, 2: 23.0, 3: 31.0}
        bx, by = st.ball.x, st.ball.y
        field = [q for q in st.outfield(team) if q.role != ROLE_GK]
        for idx, p in enumerate(sorted(field, key=lambda q: q.pid)):
            if p.role == ROLE_GK:
                out[(team, p.pid)] = LIntent(tx=p.x, ty=p.y, speed=0.5)
                continue
            if st.ball.possessing_team == team and st.ball.possessing_player == p.pid:
                target = (own_goal_x * 0.6, 34.0 if idx % 2 == 0 else 6.0)
                out[(team, p.pid)] = LIntent(tx=target[0], ty=target[1], speed=0.5, act="clear", action_target=target, power=0.9)
                continue
            ty = zone_y[idx % 4]
            # Hold a compact zonal line; a slight lane shift toward the ball.
            ty = geom.clamp(ty + (by - ty) * 0.15, 4.0, 36.0)
            out[(team, p.pid)] = LIntent(tx=line_x, ty=ty, speed=0.6)
        return out


class WallBot(OpponentController):
    """Wall-heavy archetype: pins possession to a touchline and advances via
    short near-wall passes to force wall-play training for the defender."""

    kind = "wall"

    def decide(self, st: PlanState, team: str) -> dict[tuple[str, str], LIntent]:
        out: dict[tuple[str, str], LIntent] = {}
        goal_x = 60.0 if team == "us" else 0.0
        bx, by = st.ball.x, st.ball.y
        lane = 3.5 if by < 20.0 else 36.5
        field_players = [p for p in st.players if p.team == team and p.role != ROLE_GK]
        for p in st.players:
            if p.team != team:
                continue
            if p.role == ROLE_GK:
                out[(team, p.pid)] = LIntent(tx=p.x, ty=p.y, speed=0.5)
                continue
            # Tackle if ball carrier comes close
            if st.ball.possessing_team and st.ball.possessing_team != team:
                holder = st.player(st.ball.possessing_team, st.ball.possessing_player or "")
                if holder and geom.distance(p.x, p.y, holder.x, holder.y) <= 1.5:
                    out[(team, p.pid)] = LIntent(tx=holder.x, ty=holder.y, speed=1.0, act="tackle")
                    continue
            if st.ball.possessing_team == team and st.ball.possessing_player == p.pid:
                mates = [q for q in st.outfield(team) if q.pid != p.pid]
                # Advance along the near wall; shoot inside the last 18 m.
                if goal_x - p.x <= 18.0:
                    out[(team, p.pid)] = LIntent(tx=p.x, ty=p.y, speed=0.4, act="shoot", action_target=(goal_x, 17.0 if p.y >= 20.0 else 23.0), power=0.95)
                    continue
                if mates:
                    t = max(mates, key=lambda q: q.x)
                    tx = geom.clamp(t.x, 0.0, 60.0)
                    out[(team, p.pid)] = LIntent(tx=t.x, ty=t.y, speed=0.5, act="pass", action_target=(tx, t.y), power=0.55)
                    out[(team, t.pid)] = LIntent(tx=tx, ty=t.y, speed=0.8)
                continue
            # Approach the ball hugging the chosen touchline.
            if geom.distance(p.x, p.y, bx, by) < 6.0:
                out[(team, p.pid)] = LIntent(tx=bx, ty=geom.clamp(lane if abs(by - lane) < 6.0 else by, 2.0, 38.0), speed=0.8)
            else:
                out[(team, p.pid)] = LIntent(tx=p.x, ty=geom.clamp(by if abs(by - lane) < 6.0 else lane, 2.0, 38.0), speed=0.6)
        return out


class OpponentLike(Protocol):
    """What `play_match` actually requires of a registered opponent.

    Structural on purpose: the arena registers the parameterised zoo and the
    engine probe registers a thin adapter around a dynamically executed module,
    and neither subclasses `OpponentController` even though both satisfy it.
    """

    def decide(
        self, st: PlanState, team: str
    ) -> dict[tuple[str, str], LIntent]: ...


# Any `rng -> controller` factory, not only a class: `play_match` only ever
# calls it. Typing it as `type[OpponentController]` made those registrations a
# type error while working fine at runtime.
OPPONENTS: dict[str, Callable[[random.Random], OpponentLike]] = {
    "possession": PossessionBot,
    "press": PressBot,
    "direct": DirectBot,
    "defensive": DefensiveBot,
    "random": RandomBot,
    "counter": CounterBot,
    "parkbus": ParkBusBot,
    "wall": WallBot,
}


def plan_to_obs(st: PlanState, game_id: str, sequence: int) -> dict:
    """Convert a PlanState into an engine-shaped observation dict so the real
    RuntimeManager and reward functions can be reused verbatim in simulation."""
    def player(team: str, pid: str, role: str, x: float, y: float, facing: float) -> dict:
        p = st.player(team, pid)
        # RULES.md: a player in cooldown, staggered, grounded or knocked down
        # cannot act. Reporting `canAct: true` unconditionally made every
        # kick-happy policy (including the opponent zoo) re-request a kick for
        # the whole 0.3 s cooldown, and the engine silently dropped them.
        can_act = True if p is None else (p.cooldown <= 0.0 and p.grounded <= 0.0 and p.can_act)
        return {
            "id": pid,
            "role": role,
            "position": {"x": x, "y": y},
            "velocity": {"x": 0.0, "y": 0.0},
            "facingRadians": facing,
            "canAct": can_act,
        }

    us = [player("us", p.pid, p.role, p.x, p.y, p.facing) for p in st.players if p.team == "us"]
    them = [player("them", p.pid, p.role, p.x, p.y, p.facing) for p in st.players if p.team == "them"]
    obs = {
        "protocolVersion": "1.0",
        "gameId": game_id,
        "sequence": sequence,
        "simulationTick": sequence,
        "applyAtTick": sequence,
        "timeRemainingSeconds": max(0.0, 300.0 - st.time),
        "phase": "openPlay",
        "score": {"us": st.score_us, "them": st.score_them},
        "ball": {
            "position": {"x": st.ball.x, "y": st.ball.y},
            "velocity": {"x": st.ball.vx, "y": st.ball.vy},
            "possessingTeam": st.ball.possessing_team,
            "possessedBy": st.ball.possessing_player,
        },
        "us": us,
        "them": them,
    }
    return obs


def plan_reward(st: PlanState, weights: dict[str, float], events: list[str]) -> float:
    """Shaped reward over a PlanState through the shared evaluate.reward."""
    obs = plan_to_obs(st, "sim", 0)
    state = GameState.from_observation(obs)
    world = WorldModel.build(state)
    prev: dict[str, bool] = {
        "goal": any(e == "goal" for e in events),
        "goal_conceded": any(e == "goal_conceded" for e in events),
        "dangerous_turnover": any(e == "turnover:lost" for e in events),
    }
    return reward(state, world, weights, prev)


def play_match(
    manager: RuntimeManager | None,
    opponent_kind: str,
    rng: random.Random,
    decisions: int = DEFAULT_DECISIONS,
    collect_trace: bool = False,
    collect_obs: bool = False,
    them_manager: RuntimeManager | None = None,
) -> SimResult:
    """Play the runtime team vs a scripted opponent inside the LightEngine.

    `manager` is required when using the real runtime brain for our side; for
    pure MCTS / policy-distillation runs pass `None` (only the opponent plays).

    `them_manager`, when given, replaces the scripted opponent with a second
    runtime brain that controls the away side — used for Hall-of-Fame
    head-to-head regression tests (§36). `opponent_kind` is ignored then.
    """
    engine = LightEngine(seed=rng.randint(0, 2**31 - 1))
    st = make_state(
        [(pid, role, x, y) for pid, role, x, y in BASE_LINEUP],
        mirrored_lineup(),
        ball=(30.0, 20.0),
        possess=("us", "st"),
    )
    opponent = None if them_manager is not None else OPPONENTS[opponent_kind](rng)

    result = SimResult()
    our_poss_ticks = 0
    total_ticks = 0
    for step in range(decisions):
        st.time += DECISION_INTERVAL
        total_ticks += 1
        if st.ball.possessing_team == "us":
            our_poss_ticks += 1

        merged: dict[tuple[str, str], LIntent] = {}

        def _collect(team: str, decision: dict) -> None:
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

        if manager is not None:
            obs = plan_to_obs(st, "sim", step)
            _collect("us", manager.decide(obs))
        if them_manager is not None:
            obs_them = plan_to_obs(st, "sim", step)
            # Give the away brain the "us" viewpoint (its own coords already
            # match its perspective in the normalized observation).
            obs_them["us"], obs_them["them"] = obs_them["them"], obs_them["us"]
            _collect("them", them_manager.decide(obs_them))
        elif opponent is not None:
            merged.update(opponent.decide(st, "them"))

        prev_poss = st.ball.possessing_team
        prev_poss_player = st.ball.possessing_player
        prev_events = list(st.events)
        st = engine.step(st, merged)
        result.events.extend(st.events)
        for e in st.events:
            if e == "goal":
                result.goals += 1
            elif e == "goal_conceded":
                result.goals_conceded += 1
            elif e == "kick:shoot":
                result.shots += 1
            elif e == "kick:shoot_conceded":
                result.shots_conceded += 1
            elif e.startswith("kick:"):
                # Map internal kick events to external names
                result.passes += 1
            elif e == "tackle:win":
                # Opponent won a tackle -> our pass was intercepted
                pass
        # Pass completion: if ball was loose and now we possess it, the pass arrived
        if prev_poss is None and st.ball.possessing_team == "us":
            result.completed_passes += 1
        result.rewards += plan_reward(st, REWARD_DEFAULTS, st.events)
        if collect_trace:
            result.trace.append(st)
        if collect_obs and manager is not None and step % 10 == 0:
            pass  # decision observations already routed through manager

    result.score_us = st.score_us
    result.score_them = st.score_them
    result.possession_ours = our_poss_ticks / max(1, total_ticks)
    result.rewards = float(result.rewards)
    return result