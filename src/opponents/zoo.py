"""`TunableOpponent` — one generic, spec-driven opponent controller.

Every synthetic opponent in the zoo is this class plus an `OpponentSpec`. That
keeps the behaviour space continuous (a "slightly less pressy high_press side"
is just a different spec) instead of a hand-written class per archetype, and it
makes cross-family variance cheap to generate and to regression-test.

The controller is a hand-built football AI, not a random walker:

* it maintains a block whose height, compactness and width come from the spec;
* it decides *who presses* from a trigger value, so a high-press side commits
  three players and a low-block side commits one;
* it **leads the ball** — every chaser aims at the point where the ball will
  stop being fast enough to control, which is 15-45 m down its flight path;
* it searches pass options over (aim direction x power) and only plays the ones
  a teammate can physically reach, instead of aiming at a receiver's feet;
* it marks goal-side, sprints on transitions, distributes before the engine
  forces a 20 m/s keeper distribution, sweeps with an aggressive keeper, and
  injects spec-scaled execution noise so no two seeds play identically.

The movement/decision backbone is also reused by `LureOpponent` and
`LearnedOpponent`, which override only the parts they are specialised for.

Not imported by the decision path; offline evaluation only (AGENTS.md §7).
"""

from __future__ import annotations

import math
import random

from .. import geom
from ..config import (
    BALL_CONTROL_MAX_SPEED,
    BALL_DECAY,
    DECISION_INTERVAL,
    MAX_RUN_SPEED,
    kick_speed,
    power_for,
)
from ..light import LIntent, LPlayer, PlanState
from .spec import (
    ANCHOR_DEPTH,
    ANCHOR_WIDTH,
    OpponentSpec,
    clamp_x,
    clamp_y,
    make_spec,
)

ROLE_GK = "goalkeeper"
PITCH_LENGTH = geom.PITCH_LENGTH
PITCH_WIDTH = geom.PITCH_WIDTH
GOAL_Y = geom.GOAL_CENTER_Y
REACTION_INTERVAL = DECISION_INTERVAL
# BALL_DECAY is per engine tick; the simulator advances 6 ticks per decision.
BALL_DECAY_TICK = BALL_DECAY ** 6.0
MEET_TTL = 1.6  # seconds a pass recipient stays on its meeting point
SUPPORT_TTL = 2.4  # seconds a passer keeps making its supporting run


def _clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


class TunableOpponent:
    """Spec-driven opponent. Subclasses add behaviour, never re-implement it."""

    kind = "tunable"

    def __init__(self, rng: random.Random, spec: OpponentSpec):
        self.rng = rng
        self.spec = spec
        self.kind = spec.family
        self.tick = 0
        # Cached per-match frame, refreshed once per decide() call.
        self._fwd = 1.0
        self._own_goal_x = 0.0
        self._goal_x = PITCH_LENGTH
        self._their_goal_x = 0.0
        self._ball_frac = 0.5
        self._ball_v = (0.0, 0.0)
        self._we_have = False
        self._they_have = False
        self._loose = True
        self._pressers: set[str] = set()
        self._their_players: list[LPlayer] = []
        self._outfield: list[LPlayer] = []
        # Per-player transient plan: (target_x, target_y, ttl).
        self._meet: dict[str, tuple[float, float, float]] = {}
        self._support: dict[str, tuple[float, float, float]] = {}
        # Family-specific persistent state.
        self._load_side = self.rng.choice((-1.0, 1.0))
        self._load_timer = 0.0
        self._wall_side = self.rng.choice((-1.0, 1.0))
        self._wall_timer = 0.0
        self._lag_ball = (PITCH_LENGTH / 2.0, PITCH_WIDTH / 2.0)

    # -- frame helpers ---------------------------------------------------------

    def _frame(self, st: PlanState, team: str) -> None:
        """Normalise the pitch into 'we attack toward +x' coordinates."""
        s = self.spec
        self.tick += 1
        self._fwd = 1.0 if team == "us" else -1.0
        self._own_goal_x = 0.0 if team == "us" else PITCH_LENGTH
        self._goal_x = PITCH_LENGTH if team == "us" else 0.0
        self._their_goal_x = 0.0 if team == "us" else PITCH_LENGTH
        self._outfield = sorted(
            (p for p in st.players if p.team == team and p.role != ROLE_GK),
            key=lambda p: p.pid,
        )
        self._their_players = [p for p in st.players if p.team != team]
        self._we_have = st.ball.possessing_team == team
        self._they_have = st.ball.possessing_team is not None and not self._we_have
        self._loose = st.ball.possessing_team is None
        self._ball_frac = _clamp01(
            (st.ball.x - self._own_goal_x) * self._fwd / PITCH_LENGTH
        )
        # Side-switching behaviour for the overload family.
        if s.family == "overload":
            self._load_timer -= REACTION_INTERVAL
            if self._load_timer <= 0.0:
                self._load_side = -self._load_side
                self._load_timer = 4.0 + 4.0 * self.rng.random()
        # Reaction lag: a laggy side keeps chasing a stale ball position.
        if s.press_delay > 0.0:
            if self.tick % max(1, int(round(s.press_delay / REACTION_INTERVAL))) == 0:
                self._lag_ball = (st.ball.x, st.ball.y)
        else:
            self._lag_ball = (st.ball.x, st.ball.y)
        # Transient plans decay with the clock.
        for table in (self._meet, self._support):
            for pid in [k for k, v in table.items() if v[2] <= 0.0]:
                del table[pid]
            for pid, v in list(table.items()):
                table[pid] = (v[0], v[1], v[2] - REACTION_INTERVAL)

    def _to_x(self, frac: float) -> float:
        return clamp_x(self._own_goal_x + self._fwd * _clamp01(frac) * PITCH_LENGTH)

    def _from_x(self, x: float) -> float:
        return (x - self._own_goal_x) * self._fwd / PITCH_LENGTH

    def _to_y(self, frac: float) -> float:
        return clamp_y(frac * PITCH_WIDTH)

    def _from_y(self, y: float) -> float:
        return y / PITCH_WIDTH

    def _ball(self) -> tuple[float, float]:
        """Ball position, stale by `press_delay` for a laggy side."""
        return float(self._lag_ball[0]), float(self._lag_ball[1])

    # -- press allocation ------------------------------------------------------

    def _press_urgency(self) -> float:
        """How badly this side wants the ball right now (0..1).

        Peaks around the attacking third and midfield and dies off deep in its
        own goalmouth, which is what makes a trigger-based press a trigger-based
        press rather than a mob.
        """
        target = 0.55 + 0.12 * self.spec.press_trigger
        return _clamp01(1.25 - abs(self._ball_frac - target) * 1.9)

    def _plan_press(self, bx: float, by: float) -> set[str]:
        s = self.spec
        if self._we_have or not self._outfield:
            return set()
        urgency = self._press_urgency()
        n = 1 + int(round(_clamp01(s.press_intensity) * 2.0))
        if urgency < 0.85 - 0.75 * s.press_trigger:
            n = 1  # out of zone: only the nearest player steps out
        if self._loose:
            # A ball nobody owns is worth an extra runner regardless of the
            # press trigger, otherwise both sides just let it roll.
            n = min(len(self._outfield), n + 1)
        order = sorted(
            self._outfield, key=lambda p: (geom.distance(p.x, p.y, bx, by), p.pid)
        )
        chosen = {p.pid for p in order[:n]}
        if not self._we_have and s.counterpress > 0.7 and self._loose:
            chosen |= {p.pid for p in order[: min(len(order), n + 1)]}
        return chosen

    # -- positioning -----------------------------------------------------------

    def _block_height(self) -> float:
        """Attack-direction fraction the defensive block sits at."""
        s = self.spec
        h = s.line_height
        if self._we_have:
            h += 0.08 + 0.26 * s.directness
        elif self._loose:
            h += 0.04 + 0.10 * s.press_trigger
        pull = 0.30 + 0.55 * s.compactness
        return _clamp01(self._ball_frac + (h - self._ball_frac) * pull)

    def _shape_width(self, idx: int) -> float:
        """Target width fraction (0..1) for slot `idx`."""
        s = self.spec
        w = ANCHOR_WIDTH[idx]
        w = 0.5 + (w - 0.5) * (0.45 + 0.60 * s.width)
        by_frac = self._from_y(self._ball()[1])
        attract = 0.30 * s.compactness + (0.22 * s.marking if not self._we_have else 0.0)
        w = _lerp(w, by_frac, _clamp01(attract))
        if s.wall_usage > 0.0 and self._we_have:
            # Wall play: drift the whole shape onto the chosen touchline so
            # passes stay near the boards and cutbacks are available.
            self._wall_timer -= REACTION_INTERVAL
            if self._wall_timer <= 0.0:
                self._wall_side = -self._wall_side
                self._wall_timer = 8.0 + 6.0 * self.rng.random()
            w = _lerp(w, 0.06 if self._wall_side < 0 else 0.94, 0.55 * s.wall_usage)
        if s.family == "overload":
            w = _lerp(w, 0.5 + 0.34 * self._load_side, 0.45)
        return _clamp01(w)

    def _shape_target(self, p, idx: int) -> tuple[float, float]:
        s = self.spec
        block = self._block_height()
        depth = ANCHOR_DEPTH[idx]
        if self._we_have:
            if idx in (1, 2):
                # Wide runners push beyond the ball for crosses.
                block = max(block, self._ball_frac + 0.06 + 0.16 * s.support_depth)
            block = max(block, depth - 0.04)
        elif self._loose:
            block = _clamp01(block + 0.03)
        tx = self._to_x(block)
        ty = self._to_y(self._shape_width(idx))
        if not self._we_have and s.marking > 0.0 and self._their_players:
            mark = self._marking_point(p)
            if mark is not None:
                tx = _lerp(tx, mark[0], 0.5 * s.marking)
                ty = _lerp(ty, mark[1], 0.5 * s.marking)
        return tx, ty

    def _marking_point(self, p) -> tuple[float, float] | None:
        """Goal-side of the nearest opponent this player is responsible for."""
        bx, by = self._ball()
        cands = [
            q
            for q in self._their_players
            if q.role != ROLE_GK and geom.distance(q.x, q.y, bx, by) > 3.0
        ]
        if not cands:
            cands = [q for q in self._their_players if q.role != ROLE_GK]
        if not cands:
            return None
        q = min(cands, key=lambda r: geom.distance(p.x, p.y, r.x, r.y))
        gx = self._own_goal_x + self._fwd * 3.5
        return (_lerp(q.x, gx, 0.3), _lerp(q.y, GOAL_Y, 0.2))

    # -- threat assessment -----------------------------------------------------

    def _pressure(self, p) -> float:
        """0..1 how surrounded `p` is by opponents."""
        if not self._their_players:
            return 0.0
        near = sum(
            1 for q in self._their_players if geom.distance(p.x, p.y, q.x, q.y) <= 4.0
        )
        return _clamp01(near / 3.0)

    def _lane_risk(self, ax: float, ay: float, bx: float, by: float) -> float:
        """0..1 how contested the pass lane a->b is."""
        best = 1e9
        for q in self._their_players:
            d = math.sqrt(geom.seg_point_distance_sq(q.x, q.y, ax, ay, bx, by))
            if d < best:
                best = d
        if best > 1e8:
            return 0.0
        return _clamp01(1.0 - best / 6.0)

    def _shot_risk(self, p) -> float:
        return self._lane_risk(p.x, p.y, self._goal_x, GOAL_Y)

    def _best_shot_y(self, p) -> float:
        """Pick the goal-mouth y with the most clearance, biased by aim error."""
        best_y, best_clear = GOAL_Y, -1.0
        for y in (17.0, 18.5, 20.0, 21.5, 23.0):
            clear = 1e9
            for q in self._their_players:
                if q.role == ROLE_GK:
                    continue
                clear = min(clear, geom.distance(q.x, q.y, self._goal_x, y))
            if self.spec.noise > 0.0:
                clear += self.rng.gauss(0.0, 1.5 * self.spec.noise)
            if clear > best_clear:
                best_clear, best_y = clear, y
        return best_y

    # -- kick simulation -------------------------------------------------------

    def _simulate_kick(
        self, fx: float, fy: float, ax: float, ay: float, power: float, max_steps: int = 46
    ) -> tuple[list[tuple[float, float]], int, bool]:
        """Integrate a kick under engine decay and wall bounces.

        Returns `(path, control_index, scored)` where `path[i]` is the ball
        position after decision `i`, `control_index` is the first index at which
        a player could actually take possession (speed <= BALL_CONTROL_MAX_SPEED)
        and `scored` says whether it crossed a goal plane inside the mouth.

        This matters more than it looks: the slowest possible kick still runs
        about 15 m and stays uncontrollable for ~1.8 s, so a "pass to his feet"
        is a ball sailing past the receiver. Every on-ball decision has to be
        built around where the ball will *stop being fast*, not where the
        receiver is standing.
        """
        d = geom.distance(fx, fy, ax, ay)
        if d <= 1e-6:
            return [(fx, fy)], 0, False
        v0 = kick_speed(_clamp01(power))
        vx = (ax - fx) / d * v0
        vy = (ay - fy) / d * v0
        x, y = fx, fy
        path: list[tuple[float, float]] = []
        control_index = -1
        for _ in range(max_steps):
            nx = x + vx * REACTION_INTERVAL
            ny = y + vy * REACTION_INTERVAL
            hit: str | None = None
            if nx <= 0.0 or nx >= PITCH_LENGTH:
                if geom.GOAL_LOW_Y <= ny <= geom.GOAL_HIGH_Y:
                    path.append((nx, ny))
                    return path, len(path) - 1, True
                nx = clamp_x(nx)
                hit = "left" if nx <= 0.0 else "right"
            if ny <= 0.0 or ny >= PITCH_WIDTH:
                ny = clamp_y(ny)
                hit = hit or ("bottom" if ny <= 0.0 else "top")
            if hit is not None:
                vx, vy = geom.wall_bounce(vx, vy, hit)
            x, y = nx, ny
            vx *= BALL_DECAY_TICK
            vy *= BALL_DECAY_TICK
            path.append((x, y))
            if control_index < 0 and math.hypot(vx, vy) <= BALL_CONTROL_MAX_SPEED:
                control_index = len(path) - 1
        return path, control_index, False

    def _collection_point(self, p) -> tuple[float, float]:
        """Where the currently loose ball will become collectable."""
        bx, by = self._ball()
        if not self._loose:
            return bx, by
        vx, vy = self._ball_v
        if math.hypot(vx, vy) < 0.2:
            return bx, by
        d = math.hypot(vx, vy)
        path, control_index, _ = self._simulate_kick(
            bx, by, bx + vx / d * 10.0, by + vy / d * 10.0, power_for(_clamp01(d / 26.0))
        )
        if control_index >= 0:
            return path[control_index]
        return path[-1] if path else (bx, by)

    def _intercept_point(self, p) -> tuple[float, float]:
        """Where the loose ball will stop being fast, if we can beat them to it."""
        return self._collection_point(p)

    # -- behaviour -------------------------------------------------------------

    def _pass_solution(self, p) -> tuple[tuple[float, float], float, tuple[float, float], float] | None:
        """Search aim direction x power for the best pass `p` can actually play.

        Returns `(aim, power, meet, flight_time)` where `meet` is where the
        receiver must stand to collect it, or None if no pass is worth making.
        Because the reachable landing spots are so sparse (a kick is 15-45 m),
        the honest way to pick a pass is to enumerate landing spots and ask
        "which of my teammates can be there first, and is the lane clean?".
        """
        s = self.spec
        powers = (0.0, 0.15, 0.3, 0.5) if s.directness > 0.5 else (0.0, 0.25, 0.5)
        best: tuple | None = None
        best_score = -1e9
        for power in powers:
            for step in range(10):
                ang = step * (2.0 * math.pi / 10.0) + 0.31
                ax = clamp_x(p.x + math.cos(ang) * 24.0)
                ay = clamp_y(p.y + math.sin(ang) * 24.0)
                if geom.distance(p.x, p.y, ax, ay) < 4.0:
                    continue  # no room to play a pass in that direction
                path, control_index, scored = self._simulate_kick(p.x, p.y, ax, ay, power)
                if control_index < 0:
                    continue
                cx, cy = path[control_index]
                flight = (control_index + 1) * REACTION_INTERVAL
                if self._their_goal_x <= 0.0 and cx <= 1.0 and geom.GOAL_LOW_Y <= cy <= geom.GOAL_HIGH_Y:
                    # This pass is a goal: take it.
                    return ((ax, ay), power, (cx, cy), flight)
                lane = self._lane_risk(p.x, p.y, cx, cy)
                progress = (cx - p.x) * self._fwd
                if progress < -2.0:
                    continue  # never hand the ball backwards just to be safe
                for m in self._outfield:
                    if m.pid == p.pid or m.grounded > 0.0:
                        continue
                    need = geom.distance(m.x, m.y, cx, cy)
                    reach = MAX_RUN_SPEED * flight + 0.9
                    if need > reach:
                        continue
                    # A pass into a crowded landing spot is a turnover, not a
                    # pass. Require real daylight where the ball will arrive.
                    clearance = min(
                        (geom.distance(cx, cy, q.x, q.y) for q in self._their_players),
                        default=99.0,
                    )
                    if clearance < 4.0:
                        continue
                    score = (
                        2.0 * progress
                        + 0.8 * min(clearance, 12.0)
                        - 0.5 * need
                        + 0.6 * (1.0 - self._pressure(m))
                        - 0.5 * s.risk * lane
                        + self._pass_bias(cx, cy)
                    )
                    if score > best_score:
                        best_score = score
                        best = ((ax, ay), power, (cx, cy), flight)
        return best

    def _pass_bias(self, cx: float, cy: float) -> float:
        """Extra score for a pass landing at (cx, cy).

        Subclasses use this to steer the solver: `LureOpponent` pulls passes
        into the channel it just emptied, and the slow-build feint refuses to
        play forward. Returning 0.0 keeps the neutral scoring above.
        """
        return 0.0

    def _shoot(self, p) -> LIntent:
        s = self.spec
        aim_y = self._best_shot_y(p)
        return LIntent(
            tx=p.x,
            ty=p.y,
            speed=0.3,
            act="shoot",
            action_target=(self._goal_x, aim_y),
            power=_clamp01(s.shoot_power + 0.15),
        )

    def _shot_is_on_target(self, p, power: float) -> bool:
        """Only shoot when the straight kick actually crosses the goal mouth."""
        for aim_y in (17.4, 18.8, 20.0, 21.2, 22.6):
            _, _, scored = self._simulate_kick(p.x, p.y, self._goal_x, aim_y, power)
            if scored:
                return True
        return False

    def _support_run(self, p) -> tuple[float, float]:
        """Where a passer goes after releasing the ball: a clear lane, forward."""
        s = self.spec
        ahead = self._ball_frac + 0.08 + 0.18 * s.support_depth
        ty = self._to_y(self._shape_width(self._slot_of(p)))
        if s.wall_usage > 0.0:
            ty = _lerp(ty, self._to_y(0.08 if self._wall_side < 0 else 0.92), 0.4 * s.wall_usage)
        return self._to_x(ahead), ty

    def _slot_of(self, p) -> int:
        for i, q in enumerate(self._outfield):
            if q.pid == p.pid:
                return i
        return 0

    def _on_ball(self, p) -> LIntent:
        s = self.spec
        goal_dist = (self._goal_x - p.x) * self._fwd
        can_act = p.can_act and p.cooldown <= 0.0
        if can_act and goal_dist <= s.shoot_range:
            # A straight kick at the mouth is essentially unstoppable in this
            # engine, so range is the whole story: if the geometry reaches the
            # goal, shoot rather than pass. A carried ball can never score, so
            # standing on the goal line holding it is not an option.
            power = _clamp01(s.shoot_power + 0.15)
            if self._shot_is_on_target(p, power):
                if goal_dist <= 8.0 or self.rng.random() < 0.35 + 0.65 * s.shot_ambition:
                    return self._shoot(p)
        solution = self._pass_solution(p)
        if solution is not None:
            aim, power, meet, flight = solution
            # Hand the receiver its collection point so it goes and picks it up.
            for m in self._outfield:
                if m.pid == p.pid:
                    continue
                if geom.distance(m.x, m.y, meet[0], meet[1]) <= MAX_RUN_SPEED * flight + 1.2:
                    self._meet[m.pid] = (meet[0], meet[1], MEET_TTL)
                    break
            self._support[p.pid] = (*self._support_run(p), SUPPORT_TTL)
            return LIntent(tx=p.x, ty=p.y, speed=0.4, act="pass", action_target=aim, power=power)
        # No safe forward pass: carry the ball. A carried ball stays glued to
        # the carrier at 6.4 m/s, so carrying is the only movement that
        # reliably keeps possession *and* walks it into shooting range.
        return self._carry(p)

    def _carry(self, p) -> LIntent:
        """Dribble toward the goal, steering around whoever is closest."""
        s = self.spec
        goal_dist = (self._goal_x - p.x) * self._fwd
        goal_y = GOAL_Y
        # Aim at the goal, then bend away from the nearest opponent so a
        # covering defender does not simply cut the carry off.
        if self._their_players:
            blocker = min(self._their_players, key=lambda q: geom.distance(p.x, p.y, q.x, q.y))
            away = -1.0 if (p.y - blocker.y) < 0.0 else 1.0
            if geom.distance(p.x, p.y, blocker.x, blocker.y) < 6.0:
                goal_y = clamp_y(GOAL_Y + away * (4.0 + 6.0 * s.risk))
        lead = 0.10 + 0.10 * s.directness
        # Never carry past the shooting range: a carried ball stays glued to the
        # carrier and cannot score, so dribbling into the goalmouth just parks
        # the ball on the line.
        if goal_dist <= s.shoot_range * 0.8:
            return LIntent(tx=self._to_x(self._from_x(p.x)), ty=_lerp(p.y, GOAL_Y, 0.2), speed=0.6)
        return LIntent(
            tx=self._to_x(self._from_x(p.x) + lead),
            ty=_lerp(p.y, goal_y, 0.35),
            speed=_clamp01(0.75 + 0.25 * s.tempo),
        )

    def _tackle_attempt(self, p, holder) -> bool:
        s = self.spec
        if p.role == ROLE_GK or p.grounded > 0.0 or p.cooldown > 0.0:
            return False
        if holder is not None and holder.team != p.team:
            if geom.distance(p.x, p.y, holder.x, holder.y) > 1.25:
                return False
            return self.rng.random() < 0.35 + 0.65 * s.tackling
        return self._slap_attempt(p)

    def _slap_attempt(self, p) -> bool:
        s = self.spec
        if p.role == ROLE_GK or p.grounded > 0.0 or p.cooldown > 0.0:
            return False
        for q in self._their_players:
            if geom.distance(p.x, p.y, q.x, q.y) <= 1.15 and self.rng.random() < 0.12 * s.physical:
                return True
        return False

    def _gk_intent(self, gk, st: PlanState) -> LIntent:
        s = self.spec
        bx, by = self._ball()
        tx = self._own_goal_x + self._fwd * 3.0
        ty = clamp_y(_lerp(by, GOAL_Y, 0.35))
        if self._we_have and st.ball.possessing_player == gk.pid:
            # The engine force-feeds a 20 m/s distribution after 1.25 s, which
            # is uncontrollable and usually runs out of play. Get rid of it
            # first, on our terms, with a pass a teammate can actually collect.
            solution = self._pass_solution(gk)
            if solution is not None:
                aim, power, meet, _flight = solution
                for m in self._outfield:
                    if geom.distance(m.x, m.y, meet[0], meet[1]) <= MAX_RUN_SPEED * _flight + 1.2:
                        self._meet[m.pid] = (meet[0], meet[1], MEET_TTL)
                        break
                return LIntent(tx=gk.x, ty=gk.y, speed=0.3, act="pass", action_target=aim, power=power)
            return LIntent(tx=self._to_x(0.3), ty=clamp_y(by), speed=0.7)
        if not self._we_have and self._from_x(bx) < 0.45:
            # Sweeper: leave the line to smother loose ball in our own third.
            reach = 4.0 + 9.0 * s.gk_aggression
            tx = _lerp(tx, self._own_goal_x + self._fwd * reach, 0.6 * s.gk_aggression)
            ty = _lerp(ty, by, 0.45 * s.gk_aggression)
        return LIntent(tx=tx, ty=ty, speed=_clamp01(s.gk_speed))

    # -- noise -----------------------------------------------------------------

    def _distort(self, x: float, y: float) -> tuple[float, float]:
        n = self.spec.noise
        if n <= 0.0:
            return x, y
        return (
            clamp_x(x + self.rng.gauss(0.0, 2.4 * n)),
            clamp_y(y + self.rng.gauss(0.0, 2.4 * n)),
        )

    def _hesitates(self) -> bool:
        """True when the side fumbles the intent (low `discipline`)."""
        return self.rng.random() > self.spec.discipline

    # -- main entry point ------------------------------------------------------

    def decide(self, st: PlanState, team: str) -> dict[tuple[str, str], LIntent]:
        self._ball_v = (st.ball.vx, st.ball.vy)
        self._frame(st, team)
        if not self._outfield:
            return {}
        s = self.spec
        bx, by = self._ball()
        self._pressers = self._plan_press(bx, by)
        holder = None
        if self._they_have:
            holder = st.player("them" if team == "us" else "us", st.ball.possessing_player or "")

        out: dict[tuple[str, str], LIntent] = {}
        for idx, p in enumerate(self._outfield):
            out[(team, p.pid)] = self._player_intent(p, idx, st, holder, bx, by)
        gk = st.gk(team)
        if gk is not None:
            out[(team, gk.pid)] = self._gk_intent(gk, st)
        return out

    def _player_intent(self, p, idx, st, holder, bx, by) -> LIntent:
        s = self.spec
        # Fresh possession: everyone leans forward immediately.
        transition = 0.06 * s.transition_speed if self._we_have else 0.0

        if p.pid == st.ball.possessing_player and self._we_have:
            if self._hesitates():
                return LIntent(tx=p.x, ty=p.y, speed=0.5)
            return self._on_ball(p)

        # Pass recipient: run onto the ball and hold the meeting point so the
        # pass is actually collected instead of sailing past a moving target.
        plan = self._meet.get(p.pid)
        if plan is not None and not self._we_have:
            plan = None  # someone else has the ball; the plan is void
        if plan is not None:
            tx, ty = plan[0], plan[1]
            speed = 0.95
            if self._slap_attempt(p):
                tx, ty, speed = p.x, p.y, 0.6
            return LIntent(tx=clamp_x(tx), ty=clamp_y(ty), speed=speed)

        if p.pid in self._pressers:
            tx, ty = self._intercept_point(p)
            speed = _clamp01(s.tempo)
            act = "none"
            if self._tackle_attempt(p, holder):
                act = "tackle"
            elif self._slap_attempt(p):
                act = "slap"
            tx, ty = self._distort(tx, ty)
            if self._hesitates():
                speed *= 0.55
            return LIntent(tx=tx, ty=ty, speed=speed, act=act)

        # Passer making its supporting run after releasing the ball.
        run = self._support.get(p.pid)
        if run is not None and self._we_have:
            return LIntent(tx=clamp_x(run[0]), ty=clamp_y(run[1]), speed=0.8)

        tx, ty = self._shape_target(p, idx)
        tx = clamp_x(tx + self._fwd * transition * PITCH_LENGTH)
        speed = 0.45 + 0.25 * s.tempo
        tx, ty = self._distort(tx, ty)
        if self._hesitates():
            speed *= 0.5
        return LIntent(tx=tx, ty=ty, speed=_clamp01(speed))


def make_tunable(
    rng: random.Random, name: str, family: str, difficulty: float = 0.8, **over
) -> TunableOpponent:
    """Factory: build a configured `TunableOpponent` for a family."""
    salt = rng.randrange(1 << 30)
    return TunableOpponent(rng, make_spec(name, family, difficulty, seed_salt=salt, **over))


__all__ = ["TunableOpponent", "make_tunable"]
