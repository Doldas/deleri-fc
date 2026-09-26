"""`LureOpponent` — deception: show a gap, then attack the one you made.

A straightforward opponent is beaten by a good policy because the policy can
*read* it. This class exists to break that: it deliberately puts a visible
weakness in front of the opponent in order to move the opponent's players
somewhere else, and then punishes the space that movement left behind.

Two feints, both driven by an online estimate of where the opponent's players
actually are:

`invite-then-burst`
    While the opponent has the ball, the shape is skewed to leave one channel
    looking open. The opponent reads that as a free lane and commits players
    into it. The moment possession is won, every runner goes to the *other*
    channel, the shooting range is stretched, and the burst is over in about
    two seconds.

`slow-build`
    Near its own goal the side plays safe and lets the press come, then plays
    long into the space behind it.

Every feint is deliberately fallible: the side the lure reads is chosen from
the opponent's observed commitment, but with probability `noise` it commits to
the wrong channel anyway. A lure that is always right is just another habit for
the opponent's model to learn.
"""

from __future__ import annotations

import random

from .. import geom
from ..light import LIntent, PlanState
from .spec import OpponentSpec, make_spec
from .zoo import GOAL_Y, PITCH_LENGTH, TunableOpponent, _clamp01, _lerp

ROLE_GK = "goalkeeper"
COMMIT_RADIUS = 8.0  # how close counts as "committed to the ball"
COMMIT_ALPHA = 0.25  # EWMA rate for the commitment estimate


class LureOpponent(TunableOpponent):
    kind = "lure"

    def __init__(self, rng: random.Random, spec: OpponentSpec):
        super().__init__(rng, spec)
        # EWMA of how many opponent outfielders are within COMMIT_RADIUS of the
        # ball on each half of the pitch. Index 0 is y < GOAL_Y, index 1 is above.
        self._commit = [0.5, 0.5]
        self._invite_side = self.rng.choice((-1.0, 1.0))
        self._invite_timer = 0.0
        self._burst_side = 0.0
        self._burst_timer = 0.0
        self._soft_press_timer = 0.0
        self.feints_run = 0
        self.feints_scored = 0
        self.feints_conceded = 0

    # -- state machine ---------------------------------------------------------

    def _update_deception(self, st: PlanState) -> None:
        s = self.spec
        bx, by = self._ball()
        low = high = 0
        for q in self._their_players:
            if q.role == ROLE_GK:
                continue
            if geom.distance(q.x, q.y, bx, by) <= COMMIT_RADIUS:
                if q.y < GOAL_Y:
                    low += 1
                else:
                    high += 1
        self._commit[0] = (1.0 - COMMIT_ALPHA) * self._commit[0] + COMMIT_ALPHA * low
        self._commit[1] = (1.0 - COMMIT_ALPHA) * self._commit[1] + COMMIT_ALPHA * high

        self._invite_timer -= 0.1
        self._burst_timer = max(0.0, self._burst_timer - 0.1)
        self._soft_press_timer = max(0.0, self._soft_press_timer - 0.1)

        if self._we_have and self._burst_timer <= 0.0:
            self._burst_side = self._weak_channel()
            if self._burst_side != 0.0:
                # The lure has paid off: two seconds of committed runners into
                # the channel the opponent just emptied.
                self._burst_timer = 1.6 + 2.4 * s.deception
                self.feints_run += 1
        elif not self._we_have and self._invite_timer <= 0.0:
            self._invite_side = self._weak_channel()
            if self._invite_side == 0.0:
                self._invite_side = self.rng.choice((-1.0, 1.0))
            # Occasionally stand off the press entirely and let them out.
            if self.rng.random() < 0.5 * s.deception:
                self._soft_press_timer = 0.8 + 1.6 * self.rng.random()
            self._invite_timer = 2.5 + 3.5 * self.rng.random()

    def _weak_channel(self) -> float:
        """Side of the pitch the opponent has left thinnest near the ball.

        Returns -1.0 (low y), +1.0 (high y) or 0.0 when it cannot tell.
        `noise` makes the read deliberately fallible.
        """
        lo, hi = self._commit
        if abs(lo - hi) < 0.15:
            return 0.0
        read = -1.0 if lo < hi else 1.0
        if self.rng.random() < self.spec.noise * 1.5:
            return -read  # misjudged: this is the "lure the lure" randomness
        return read

    def _frame(self, st: PlanState, team: str) -> None:
        super()._frame(st, team)
        self._update_deception(st)

    # -- overrides -------------------------------------------------------------

    def _plan_press(self, bx: float, by: float) -> set[str]:
        pressers = super()._plan_press(bx, by)
        if self._soft_press_timer > 0.0 and not self._we_have and len(pressers) > 1:
            # Stand off: one player only, so the opponent commits and leaves gaps.
            order = sorted(
                self._outfield, key=lambda p: (geom.distance(p.x, p.y, bx, by), p.pid)
            )
            pressers = {order[0].pid}
        return pressers

    def _shape_target(self, p, idx: int) -> tuple[float, float]:
        tx, ty = super()._shape_target(p, idx)
        if self._we_have and self._burst_timer <= 0.0:
            return tx, ty
        # Defending: visibly vacate the invite channel so the opponent reads it
        # as the safe way to play and moves players into it.
        side = self._invite_side
        on_invite_side = (ty >= GOAL_Y) if side > 0 else (ty < GOAL_Y)
        if idx in (1, 2) and on_invite_side:
            pull = 0.55 * self.spec.deception
            ty = _lerp(ty, GOAL_Y + side * 11.0, pull)
            tx = _lerp(tx, self._to_x(self._ball_frac - 0.06), pull)
        return tx, ty

    def _pass_bias(self, cx: float, cy: float) -> float:
        """Per-candidate bonus/penalty applied inside `_pass_solution`."""
        if self._we_have and self._burst_timer > 0.0 and self._burst_side != 0.0:
            return 12.0 * self._burst_side * (1.0 if cy >= GOAL_Y else -1.0)
        if self._we_have and self._ball_frac < 0.35:
            # Slow build: do not play the ball forward into a settled press.
            return -8.0 if cx > self._ball_x() else 0.0
        return 0.0

    def _ball_x(self) -> float:
        return self._ball()[0]

    def _on_ball(self, p) -> LIntent:
        s = self.spec
        if self._burst_timer > 0.0 and self._burst_side != 0.0:
            goal_dist = (self._goal_x - p.x) * self._fwd
            # Burst mode shoots from further out than a normal side would.
            if goal_dist <= s.shoot_range * 1.7:
                power = _clamp01(s.shoot_power + 0.15)
                if self._shot_is_on_target(p, power) and self.rng.random() < 0.75:
                    return self._shoot(p)
            channel_y = clamp_channel(GOAL_Y + self._burst_side * 9.0)
            return LIntent(
                tx=self._to_x(self._from_x(p.x) + 0.14),
                ty=_lerp(p.y, channel_y, 0.45),
                speed=1.0,
            )
        return super()._on_ball(p)


def clamp_channel(y: float) -> float:
    return 3.0 if y < 3.0 else 37.0 if y > 37.0 else y


def make_lure(
    rng: random.Random, name: str, difficulty: float = 0.9, **over
) -> LureOpponent:
    salt = rng.randrange(1 << 30)
    return LureOpponent(rng, make_spec(name, "lure", difficulty, seed_salt=salt, **over))


__all__ = ["LureOpponent", "make_lure"]
