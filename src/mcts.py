"""MCTSPlanner — Monte Carlo Tree Search over tactical TeamActions.

Implements the first-experiment scope from AISTRATEGI §61: a small action
abstraction (pass / shoot / dribble / wall_pass / press), a 2 s horizon, UCT
with a tactical prior, rollouts through the internal LightEngine, and shaped
rewards. Offline this feeds policy distillation; at runtime it stays optional
and budget-limited (AISTRATEGI §26–30, §40).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from . import geom
from .config import DECISION_INTERVAL
from .geom import OPP_GOAL_X
from .light import LIntent, LightEngine, PlanState, make_state
from .physics import pick_shot_target
from .policy import PlayerIntent, PolicyInput
from .state import GameState
from .wall import is_near_wall

# Tactical action labels (AISTRATEGI §28 subset used by the first experiment).
ACTION_POLICY = "POLICY"
ACTION_SHOOT = "SHOOT"
ACTION_PASS_FORWARD = "PASS_FORWARD"
ACTION_PASS_WIDE = "PASS_WIDE"
ACTION_PASS_SUPPORT = "PASS_SUPPORT"
ACTION_DRIBBLE_FORWARD = "DRIBBLE_FORWARD"
ACTION_WALL_PASS = "WALL_PASS"


@dataclass
class Candidate:
    label: str
    intents: dict[str, PlayerIntent]
    prior: float


@dataclass
class _Node:
    label: str
    prior: float
    n: int = 0
    w: float = 0.0
    children: dict[str, "_Node"] = field(default_factory=dict)


def _to_plan(inp: PolicyInput) -> PlanState:
    state: GameState = inp.state
    us = [(p.id, p.role, p.x, p.y) for p in state.us]
    them = [(p.id, p.role, p.x, p.y) for p in state.them]
    ball = (state.ball.x, state.ball.y)
    bv = (state.ball.vx, state.ball.vy)
    possess = None
    if state.ball.possessing_team is not None:
        possess = (state.ball.possessing_team, state.ball.possessing_player or "")
    plan = make_state(us, them, ball=ball, ball_v=bv, possess=possess, score=(state.score_us, state.score_them))
    return plan


def _to_light(intents: dict[str, PlayerIntent], team: str) -> dict[tuple[str, str], LIntent]:
    out: dict[tuple[str, str], LIntent] = {}
    for pid, it in intents.items():
        out[(team, pid)] = LIntent(
            tx=it.tx,
            ty=it.ty,
            speed=it.speed,
            act=it.action_type,
            action_target=it.action_target,
            power=it.action_power,
        )
    return out


class MCTSPlanner:
    def __init__(
        self,
        genome: dict[str, float],
        weights: dict[str, float],
        rng: random.Random,
        iterations: int = 40,
        horizon: float = 2.0,
        cpuct: float = 1.4,
    ):
        self.genome = genome
        self.weights = weights
        self.rng = rng
        self.iterations = iterations
        self.horizon = horizon
        self.cpuct = cpuct
        self.engine = LightEngine(seed=47)

    def choose(self, inp: PolicyInput, base_intents: dict[str, PlayerIntent]) -> dict[str, PlayerIntent] | None:
        """Run a shallow UCT tree whose root children are tactical candidates.
        Returns the best intent set, or None when there is nothing to decide
        over (caller keeps its base intents)."""
        candidates = self._candidates(inp, base_intents)
        if len(candidates) <= 1:
            return None

        base = _to_plan(inp)
        root = _Node("root", 1.0)
        for label, c in candidates.items():
            root.children[label] = _Node(label=label, prior=max(c.prior, 0.01))

        visits: dict[str, int] = {label: 0 for label in candidates}
        values: dict[str, float] = {label: 0.0 for label in candidates}
        total = 0.0
        for _ in range(self.iterations):
            total += 1.0
            label = max(root.children, key=lambda lb: self._ucb(root, root.children[lb], total, visits))
            visits[label] += 1
            values[label] += self._rollout(base, candidates[label].intents)

        best_label = max(candidates, key=lambda lb: values[lb] / max(1, visits[lb]))
        return candidates[best_label].intents

    def _ucb(self, root: _Node, node: _Node, total: float, visits: dict[str, int]) -> float:
        n = visits[node.label]
        if n == 0:
            return 1e9 + node.prior
        # Mean value is stored separately (values dict); here we only need the
        # exploration bonus to break ties across unvisited/local optima.
        return node.prior * self.cpuct * math.sqrt(math.log(total + 1.0) / (n + 1.0))

    def _rollout(self, state: PlanState, our_intents: dict[str, PlayerIntent]) -> float:
        st = state
        our_light = _to_light(our_intents, "us")
        them_light = self._opponent_intents(st)
        total_reward = 0.0
        gamma = 1.0
        steps = max(1, int(round(self.horizon / DECISION_INTERVAL)))
        for _ in range(steps):
            merged = dict(our_light)
            merged.update(them_light)
            st = self.engine.step(st, merged)
            total_reward += gamma * self._reward(st)
            gamma *= 0.97
            if any(e.startswith("goal") for e in st.events):
                break
        return total_reward

    def _reward(self, st: PlanState) -> float:
        w = self.weights
        r = 0.0
        for e in st.events:
            if e == "goal":
                r += w["goal"]
            elif e == "goal_conceded":
                r += w["goal_conceded"]
            elif e == "kick:shoot":
                r += w["shot_quality"] * 0.05
            elif e in ("tackle:win", "slap:down"):
                r += w["counterpress_recovery"] * 0.5
        if st.ball.possessing_team == "us":
            r += (st.ball.x - 30.0) * w["territorial_progression"]
        return r

    def _opponent_intents(self, st: PlanState) -> dict[tuple[str, str], LIntent]:
        # Nearest opponent presses the ball; the rest hold a compact shape.
        out: dict[tuple[str, str], LIntent] = {}
        bx, by = st.ball.x, st.ball.y
        for p in st.outfield("them"):
            if geom.distance(p.x, p.y, bx, by) < 6.0:
                out[("them", p.pid)] = LIntent(tx=bx, ty=by, speed=0.9)
            else:
                tx = geom.clamp(p.x + (bx - p.x) * 0.3, 0.0, OPP_GOAL_X * 0.5)
                ty = geom.clamp(p.y + (by - p.y) * 0.3, 0.0, 40.0)
                out[("them", p.pid)] = LIntent(tx=tx, ty=ty, speed=0.6)
        gk = st.gk("them")
        if gk is not None:
            # Opponent GK defends x=60, their defensive fifth starts at x=48
            # Move GK toward center of their defensive fifth (x=54)
            target_gkx = 54.0
            gkx = gk.x + (target_gkx - gk.x) * 0.5
            out[("them", gk.pid)] = LIntent(tx=gkx, ty=gk.y, speed=0.6)
        return out

    def _candidates(self, inp: PolicyInput, base_intents: dict[str, PlayerIntent]) -> dict[str, Candidate]:
        cand: dict[str, Candidate] = {}
        cand[ACTION_POLICY] = Candidate(ACTION_POLICY, dict(base_intents), 0.6)

        possessor = inp.state.our_possessor()
        if possessor is None:
            return cand

        # Forward-pass variants onto the most advanced receivers.
        receivers = _rank_receivers(inp, possessor)
        for idx in range(min(2, len(receivers))):
            pid = receivers[idx][0]
            label = f"{ACTION_PASS_FORWARD}_{pid}"
            cand[label] = Candidate(label, self._with_pass(inp, base_intents, possessor.id, pid), 0.5 - idx * 0.1)

        # Wide pass.
        widers = [p for p in inp.state.outfield_us() if p.id != possessor.id and (p.y <= 8.0 or p.y >= 32.0)]
        if widers:
            w0 = widers[0].id
            cand[ACTION_PASS_WIDE] = Candidate(ACTION_PASS_WIDE, self._with_pass(inp, base_intents, possessor.id, w0), 0.4)

        # Keep driving forward.
        cand[ACTION_DRIBBLE_FORWARD] = Candidate(ACTION_DRIBBLE_FORWARD, self._with_dribble(inp, base_intents, possessor.id), 0.45)

        # Immediate shot opportunity.
        if OPP_GOAL_X - possessor.x < 20.0:
            cand[ACTION_SHOOT] = Candidate(ACTION_SHOOT, self._with_shot(inp, base_intents, possessor.id), 0.35)

        # Wall pass near touchlines.
        if is_near_wall(possessor.x, possessor.y, margin=6.0):
            cand[ACTION_WALL_PASS] = Candidate(ACTION_WALL_PASS, self._with_wall(inp, base_intents, possessor.id), 0.4)
        return cand

    # -- candidate builders ------------------------------------------------
    @staticmethod
    def _with_pass(inp: PolicyInput, base: dict[str, PlayerIntent], src: str, dst: str) -> dict[str, PlayerIntent]:
        out = dict(base)
        src_it = out.get(src)
        dst_p = next((p for p in inp.state.outfield_us() if p.id == dst), None)
        if src_it is not None and dst_p is not None:
            tx, ty = dst_p.x, dst_p.y
            out[src] = PlayerIntent(src, src_it.tx, src_it.ty, 0.4, tx, ty, "pass", (tx, ty), 0.5)
            out[dst] = PlayerIntent(dst, dst_p.x, dst_p.y, 0.7, inp.state.ball.x, inp.state.ball.y)
        return out

    @staticmethod
    def _with_dribble(inp: PolicyInput, base: dict[str, PlayerIntent], src: str) -> dict[str, PlayerIntent]:
        out = dict(base)
        it = out.get(src)
        p = next((q for q in inp.state.outfield_us() if q.id == src), None)
        if it is not None and p is not None:
            tx = min(p.x + 6.0, 58.0)
            out[src] = PlayerIntent(src, tx, p.y, 0.8, tx, p.y)
        return out

    @staticmethod
    def _with_shot(inp: PolicyInput, base: dict[str, PlayerIntent], src: str) -> dict[str, PlayerIntent]:
        out = dict(base)
        it = out.get(src)
        gk = inp.state.goalkeeper_them()
        gkx = gk.x if gk else 58.0
        gky = gk.y if gk else 20.0
        if it is not None:
            t = pick_shot_target(inp.state.ball.x, inp.state.ball.y, gkx, gky)
            out[src] = PlayerIntent(src, it.tx, it.ty, 0.4, t.x, t.y, "shoot", (t.x, t.y), t.power)
        return out

    @staticmethod
    def _with_wall(inp: PolicyInput, base: dict[str, PlayerIntent], src: str) -> dict[str, PlayerIntent]:
        """Wall pass: play a firm pass into the wall ahead and follow it up."""
        out = dict(base)
        it = out.get(src)
        p = next((q for q in inp.state.outfield_us() if q.id == src), None)
        if it is not None and p is not None:
            from .wall import direction_to_wall
            dx, dy = direction_to_wall(p.x, p.y)
            tx = geom.clamp(p.x + dx * 4.0, 0.1, OPP_GOAL_X - 0.1)
            ty = geom.clamp(p.y + dy * 4.0, 0.1, 39.9)
            out[src] = PlayerIntent(src, tx, ty, 0.8, tx, ty, "pass", (tx, ty), 0.6)
        return out


def _rank_receivers(inp: PolicyInput, possessor) -> list[tuple[str, float]]:
    """(player-id, forward x) sorted by progression for candidate generation."""
    rows = [(t.id, t.x) for t in inp.state.outfield_us() if t.id != possessor.id]
    rows.sort(key=lambda r: r[1], reverse=True)
    return rows