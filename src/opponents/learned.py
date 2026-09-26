"""`LearnedOpponent` — an opponent that improves instead of following a script.

The rest of `src/opponents/` are hand-written policies: good heuristics, but a
fixed set of habits. A policy that never adapts is a policy with a tell. This
one keeps a set of linear action-value models and updates them *during* the
match, so it learns what works against the specific team in front of it.

Design constraints, in order of priority:

1. **Determinism.** Pure Python, no clock, no external deps, seeded `rng`. All
   state lives in the controller instance, which is created per match by
   `OPPONENTS[name](rng)` and dropped at match end.
2. **Bounded work.** One SARSA update per decision for the acting player only,
   and a dot product over a fixed-size feature vector. Far inside the 50 ms
   decision budget.
3. **Honest evaluation.** Opponents are *worse* than the tuned heuristics on a
   cold start and only approach them as they accumulate experience. A learned
   agent that begins strong would be cheating.

The model is a contextual bandit with SARSA-style on-policy correction:

* features: a small hand-designed basis of the match state
  (progress, goal threat, ball control, pressure, support, time, noise)
* actions: the discrete set of `LIntent`s `_choose_action` can emit
* `Q(s, a) = w[a] . phi(s)`; `w` updated by `alpha * (r + gamma * Q(s', a') - Q(s, a))`
* reward: shaped from the engine's own signal (progress made, goals, shots,
  turnovers conceded) so a positive weight always tracks real progress

Weights persist to `artifacts/opponents/<name>.json` between runs, so a
training session genuinely compounds.
"""

from __future__ import annotations

import json
import pathlib
import random

from .. import geom
from ..config import MAX_RUN_SPEED
from ..light import LIntent, PlanState
from .spec import OpponentSpec, make_spec
from .zoo import (
    GOAL_Y,
    MEET_TTL,
    PITCH_LENGTH,
    SUPPORT_TTL,
    TunableOpponent,
    _clamp01,
)

FEATURES = 16
ACTIONS = 7
ACT_SHOOT = 0
ACT_PASS = 1
ACT_CARRY = 2
ACT_HOLD = 3
ACT_PRESS = 4
ACT_TACKLE = 5
ACT_RECOVER = 6
ACTION_NAMES = ("shoot", "pass", "carry", "hold", "press", "tackle", "recover")

DEFAULT_ALPHA = 0.05
DEFAULT_GAMMA = 0.95
ARTIFACT_DIR = pathlib.Path("artifacts/opponents")


class LearnedOpponent(TunableOpponent):
    kind = "learned"

    def __init__(self, rng: random.Random, spec: OpponentSpec, name: str = "learned"):
        super().__init__(rng, spec)
        self.name = name
        self.alpha = DEFAULT_ALPHA
        self.gamma = DEFAULT_GAMMA
        # w[action * FEATURES + feature]
        self.w = [0.0] * (ACTIONS * FEATURES)
        self._prev_features: list[float] | None = None
        self._prev_action: int | None = None
        self._plan_state: PlanState | None = None
        self._prev_ball_frac = 0.5
        self._decisions = 0
        self._updates = 0
        self._goals_for = 0
        self._goals_against = 0

    # -- persistence -----------------------------------------------------------

    def load(self) -> None:
        path = ARTIFACT_DIR / f"{self.name}.json"
        try:
            blob = json.loads(path.read_text())
        except (OSError, ValueError):
            return
        w = blob.get("w")
        if isinstance(w, list) and len(w) == len(self.w):
            if all(isinstance(v, (int, float)) for v in w):
                self.w = [float(v) for v in w]

    def save(self) -> None:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "name": self.name,
            "kind": self.kind,
            "features": FEATURES,
            "actions": list(ACTION_NAMES),
            "decisions": self._decisions,
            "updates": self._updates,
            "w": [round(v, 6) for v in self.w],
        }
        (ARTIFACT_DIR / f"{self.name}.json").write_text(json.dumps(payload, indent=2))

    # -- features --------------------------------------------------------------

    def _features(self, st: PlanState) -> list[float]:
        s = self.spec
        bx, by = self._ball()
        goal_dist = abs(self._goal_x - bx) / PITCH_LENGTH
        control = 1.0 if self._we_have else 0.0
        poss = 1.0 if self._they_have else 0.0
        press = self._local_pressure(bx, by)
        # Nearest outfield support to the ball, normalised to 0..1.
        support = (
            min(geom.distance(p.x, p.y, bx, by) / 30.0 for p in self._outfield)
            if self._outfield
            else 1.0
        )
        free = 1.0 - press
        attacking = 1.0 - goal_dist
        clock = min(self.tick / 1800.0, 1.0)
        return [
            1.0,
            control,
            poss,
            attacking,
            free,
            press,
            min(support, 1.0),
            goal_dist,
            control * attacking,
            control * free,
            poss * press,
            control * s.directness,
            poss * s.compactness,
            clock,
            clock * control,
            s.noise,
        ]

    def _local_pressure(self, x: float, y: float) -> float:
        """How tightly the opponent has the ball covered at a point."""
        if not self._their_players:
            return 0.0
        return min(
            geom.distance(x, y, q.x, q.y) / 12.0 for q in self._their_players
        )

    def _q(self, action: int, feats: list[float]) -> float:
        base = action * FEATURES
        w = self.w
        return sum(w[base + i] * f for i, f in enumerate(feats))

    def _argmax(self, feats: list[float], mask: list[bool]) -> int:
        best, best_q = ACT_HOLD, -1e18
        for a in range(ACTIONS):
            if not mask[a]:
                continue
            q = self._q(a, feats)
            if self.rng.random() < self.spec.noise:
                q = self.rng.random()
            if q > best_q:
                best, best_q = a, q
        return best

    # -- learning --------------------------------------------------------------

    def _update(self, feats: list[float], action: int, reward: float) -> None:
        if self._prev_features is None or self._prev_action is None:
            self._prev_features, self._prev_action = feats, action
            return
        base = self._prev_action * FEATURES
        current = self._q(self._prev_action, self._prev_features)
        target = reward + self.gamma * self._q(action, feats)
        step = self.alpha * (target - current)
        for i, f in enumerate(self._prev_features):
            if f:
                self.w[base + i] += step * f
        self._prev_features, self._prev_action = feats, action
        self._updates += 1

    def _on_ball(self, p) -> LIntent:
        s = self.spec
        goal_dist = (self._goal_x - p.x) * self._fwd
        can_kick = p.can_act and p.cooldown <= 0.0
        mask = [False] * ACTIONS
        mask[ACT_CARRY] = True
        mask[ACT_HOLD] = True
        if can_kick:
            if goal_dist <= s.shoot_range:
                mask[ACT_SHOOT] = self._shot_is_on_target(p, _clamp01(s.shoot_power + 0.15))
            mask[ACT_PASS] = self._pass_solution(p) is not None

        st = self._plan_state
        feats = self._features(st) if st is not None else [1.0] * FEATURES
        action = self._argmax(feats, mask)
        self._update(feats, action, self._reward())
        self._decisions += 1

        if action == ACT_SHOOT:
            return self._shoot(p)
        if action == ACT_PASS:
            solution = self._pass_solution(p)
            if solution is not None:
                aim, power, meet, flight = solution
                for m in self._outfield:
                    if m.pid != p.pid and geom.distance(m.x, m.y, meet[0], meet[1]) <= (
                        MAX_RUN_SPEED * flight + 1.2
                    ):
                        self._meet[m.pid] = (meet[0], meet[1], MEET_TTL)
                        break
                self._support[p.pid] = (*self._support_run(p), SUPPORT_TTL)
                return LIntent(
                    tx=p.x, ty=p.y, speed=0.4, act="pass", action_target=aim, power=power
                )
        if action == ACT_CARRY:
            return self._carry(p)
        return LIntent(tx=self._to_x(self._from_x(p.x) + 0.04), ty=p.y, speed=0.25)

    def _reward(self) -> float:
        """Shaped reward from the engine's own state — never a scripted signal."""
        s = self.spec
        bx, _by = self._ball()
        progress = (self._ball_frac - self._prev_ball_frac) * 12.0
        control = 1.0 if self._we_have else 0.0
        threat = 1.0 - min(abs(self._goal_x - self._their_goal_x) / PITCH_LENGTH, 1.0)
        r = 0.6 * progress
        r += 0.3 * (control - 0.5) * (1.0 if s.directness > 0.5 else 0.2)
        r -= 0.4 * threat * (0.0 if control else 1.0)
        r += 0.3 * (self._goals_for - self._goals_against)
        return r

    # -- lifecycle -------------------------------------------------------------

    def _frame(self, st: PlanState, team: str) -> None:
        super()._frame(st, team)
        self._plan_state = st
        for event in st.events:
            if event == "goal:us":
                self._goals_for += 1
            elif event == "goal:them":
                self._goals_against += 1

    def summary(self) -> dict:
        return {
            "decisions": self._decisions,
            "updates": self._updates,
            "goals_for": self._goals_for,
            "goals_against": self._goals_against,
            "w": list(self.w),
        }


def make_learned(
    rng: random.Random, name: str, difficulty: float = 0.8, **over
) -> LearnedOpponent:
    salt = rng.randrange(1 << 30)
    spec = make_spec(name, "possession", difficulty, seed_salt=salt, **over)
    opponent = LearnedOpponent(rng, spec, name=name)
    opponent.load()
    return opponent


__all__ = ["LearnedOpponent", "make_learned", "ACTION_NAMES"]
