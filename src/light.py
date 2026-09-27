"""LightEngine — compact deterministic forward model of the Babylon engine.

Used by MCTS rollouts and the self-play simulator. It mirrors the rules in
RULES.md but is NOT authoritative — the real engine always wins on conflicts
(AGENTS.md §2, AISTRATEGI §52).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

from . import geom
from .config import (
    ACTION_COOLDOWN,
    BALL_CONTROL_MAX_SPEED,
    BALL_CONTROL_RADIUS,
    BALL_DECAY,
    CONTROLLED_BALL_AHEAD,
    DECISION_INTERVAL,
    GK_AUTO_SPEED,
    GK_HOLD_AUTO,
    GK_REACH,
    MAX_DRIBBLE_SPEED,
    MAX_RUN_SPEED,
    POSSESSION_PROTECTION,
    SLAP_CLEAN,
    SLAP_COOLDOWN,
    SLAP_FACING_DEG,
    SLAP_KNOCKDOWN,
    SLAP_MAX,
    SLIDE_GROUNDED,
    TACKLE_CLOSE,
    TACKLE_MAX,
    kick_speed,
)
from .geom import GOAL_HIGH_Y, GOAL_LOW_Y, PITCH_LENGTH, PITCH_WIDTH


@dataclass
class LPlayer:
    team: str
    pid: str
    role: str
    x: float
    y: float
    facing: float = 0.0
    can_act: bool = True
    cooldown: float = 0.0
    grounded: float = 0.0
    vx: float = 0.0
    vy: float = 0.0


@dataclass
class LBall:
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    possessing_team: str | None = None
    possessing_player: str | None = None
    protection: float = 0.0
    gk_hold: float = 0.0


@dataclass
class PlanState:
    time: float = 0.0
    score_us: int = 0
    score_them: int = 0
    ball: LBall = field(default_factory=lambda: LBall())
    players: list[LPlayer] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    # Restart formation: (team, pid, role, x, y) per player, used to reset both
    # teams to their formation after a goal (RULES.md §127-131).
    base_positions: list[tuple[str, str, str, float, float]] = field(default_factory=list)

    def player(self, team: str, pid: str) -> LPlayer | None:
        for p in self.players:
            if p.team == team and p.pid == pid:
                return p
        return None

    def outfield(self, team: str) -> list[LPlayer]:
        return [p for p in self.players if p.team == team and p.role != "goalkeeper"]

    def gk(self, team: str) -> LPlayer | None:
        for p in self.players:
            if p.team == team and p.role == "goalkeeper":
                return p
        return None


@dataclass
class LIntent:
    tx: float
    ty: float
    speed: float
    act: str = "none"
    action_target: tuple[float, float] | None = None
    power: float | None = None


def make_state(
    us: list[tuple[str, str, float, float]],
    them: list[tuple[str, str, float, float]],
    ball: tuple[float, float] = (30.0, 20.0),
    ball_v: tuple[float, float] = (0.0, 0.0),
    possess: tuple[str, str | None] | None = None,
    score: tuple[int, int] = (0, 0),
) -> PlanState:
    """Construct a PlanState from compact player tuples (team-local id, role,
    x, y). `possess` is (team, player-local-id) or None."""
    players = [
        LPlayer(team=team, pid=pid, role=role, x=x, y=y, facing=0.0)
        for team, rows in (("us", us), ("them", them))
        for pid, role, x, y in rows
    ]
    st = PlanState(
        score_us=score[0],
        score_them=score[1],
        ball=LBall(x=ball[0], y=ball[1], vx=ball_v[0], vy=ball_v[1]),
        players=players,
        base_positions=[
            (team, pid, role, x, y)
            for team, rows in (("us", us), ("them", them))
            for pid, role, x, y in rows
        ],
    )
    if possess is not None:
        st.ball.possessing_team = possess[0]
        st.ball.possessing_player = possess[1]
        holder = st.player(possess[0], possess[1] or "")
        if holder is not None:
            st.ball.x = holder.x + CONTROLLED_BALL_AHEAD
            st.ball.y = holder.y
    return st


class LightEngine:
    """Advances a PlanState by one decision interval (default 0.1 s)."""

    def __init__(self, seed: int = 0):
        self._rng = __import__("random").Random(seed)
        self.dt = DECISION_INTERVAL

    def step(self, state: PlanState, intents: dict[tuple[str, str], LIntent]) -> PlanState:
        st = replace(state)
        st.players = [replace(p) for p in state.players]
        st.ball = replace(state.ball)
        st.events = []

        tick_fraction = self.dt * 60.0
        decay = BALL_DECAY ** tick_fraction

        st.time += self.dt
        st.ball.protection = max(0.0, state.ball.protection - self.dt)

        # 1) Movement & facing.
        for p in st.players:
            intent = intents.get((p.team, p.pid))
            p.cooldown = max(0.0, p.cooldown - self.dt)
            p.grounded = max(0.0, p.grounded - self.dt)
            if p.grounded > 0.0:
                p.can_act = False
                continue
            # Recover from a knockdown. Without this, `can_act` latches false
            # for the rest of the match after a single slap: grounded decays to
            # zero but the flag is only ever cleared by a goal restart, so a
            # knocked-down player can run but never pass or shoot again.
            p.can_act = True
            if intent is None:
                continue
            # Turn immediately toward target.
            p.facing = geom.angle_to(p.x, p.y, intent.tx, intent.ty)
            possessing = st.ball.possessing_team == p.team and st.ball.possessing_player == p.pid
            top = MAX_DRIBBLE_SPEED if possessing else MAX_RUN_SPEED
            speed = min(1.0, max(0.0, intent.speed)) * top
            d = speed * self.dt
            dx = intent.tx - p.x
            dy = intent.ty - p.y
            norm = math.hypot(dx, dy)
            if norm > 1e-6:
                step_dist = min(d, norm)
                p.x += dx / norm * step_dist
                p.y += dy / norm * step_dist
            p.vx = (dx / norm if norm > 1e-6 else 0.0) * speed
            p.vy = (dy / norm if norm > 1e-6 else 0.0) * speed

        # 2) Ball-holder dribbling update (controlled ball follows player).
        if st.ball.possessing_team is not None:
            holder = st.player(st.ball.possessing_team, st.ball.possessing_player or "")
            if holder is not None:
                st.ball.x = holder.x + CONTROLLED_BALL_AHEAD * math.cos(holder.facing)
                st.ball.y = holder.y + CONTROLLED_BALL_AHEAD * math.sin(holder.facing)

        # 3) Actions.
        self._apply_actions(st, intents)

        # 4) Free-ball integration.
        # Integrate at engine tick rate (60 Hz) so fast balls don't skip the goal line,
        # but apply velocity decay only once per decision tick (matching original physics).
        substeps = max(1, int(self.dt * 60.0))
        dt_sub = self.dt / substeps
        for _ in range(substeps):
            goal_scored = self._integrate_ball_substep(st, dt_sub)
            if goal_scored:
                break
        # Apply velocity decay once per decision tick (matching original physics)
        decay = BALL_DECAY ** tick_fraction
        st.ball.vx *= decay
        st.ball.vy *= decay

        # 5) Auto GK handling.
        self._gk_handling(st)

        # 6) Engine-forced GK distribution after 1.25 s hold (RULES.md §90).
        self._gk_auto_distribute(st)

        return st

    def _integrate_ball_substep(self, st: PlanState, dt_sub: float) -> bool:
        """Integrate one engine sub-tick. Returns True if a goal was scored."""
        if st.ball.possessing_team is not None:
            return False
        st.ball.x += st.ball.vx * dt_sub
        st.ball.y += st.ball.vy * dt_sub
        hit: str | None = None
        if st.ball.x <= 0.0:
            if GOAL_LOW_Y <= st.ball.y <= GOAL_HIGH_Y:
                st.score_them += 1
                st.events.append("goal_conceded")
                st.ball.x = 0.0
                st.ball.possessing_team = None
                st.ball.possessing_player = None
                st.ball.vx = st.ball.vy = 0.0
                self._kickoff_reset(st, "us")
                return True
            st.ball.x = 0.0
            hit = "left"
        elif st.ball.x >= PITCH_LENGTH:
            if GOAL_LOW_Y <= st.ball.y <= GOAL_HIGH_Y:
                st.score_us += 1
                st.events.append("goal")
                st.ball.x = PITCH_LENGTH
                st.ball.possessing_team = None
                st.ball.possessing_player = None
                st.ball.vx = st.ball.vy = 0.0
                self._kickoff_reset(st, "them")
                return True
            st.ball.x = PITCH_LENGTH
            hit = "right"
        if st.ball.y <= 0.0:
            st.ball.y = 0.0
            hit = "bottom"
        elif st.ball.y >= PITCH_WIDTH:
            st.ball.y = PITCH_WIDTH
            hit = "top"
        if hit is not None:
            st.ball.vx, st.ball.vy = geom.wall_bounce(st.ball.vx, st.ball.vy, hit)
        self._loose_ball_control(st)
        return False

    def _apply_actions(self, st: PlanState, intents: dict[tuple[str, str], LIntent]) -> None:
        between = []  # (team, pid, distance)
        for (team, pid), intent in intents.items():
            p = st.player(team, pid)
            assert p is not None
            if p.grounded > 0.0 or p.cooldown > 0.0:
                continue
            if intent.act in ("pass", "shoot", "clear"):
                if st.ball.possessing_team == team and st.ball.possessing_player == pid:
                    target = intent.action_target
                    if target is not None and p.can_act:
                        power = 0.5 if intent.power is None else intent.power
                        v0 = kick_speed(max(0.0, min(1.0, power)))
                        dist = geom.distance(
                            p.x, p.y, target[0], target[1]
                        )
                        if dist > 1e-6:
                            st.ball.vx = (target[0] - p.x) / dist * v0
                            st.ball.vy = (target[1] - p.y) / dist * v0
                        st.ball.possessing_team = None
                        st.ball.possessing_player = None
                        st.ball.protection = 0.0
                        p.cooldown = ACTION_COOLDOWN
                        st.events.append("kick:" + intent.act)
            elif intent.act == "tackle" and team != st.ball.possessing_team and st.ball.possessing_team is not None:
                if p.role == "goalkeeper":
                    continue
                if st.ball.protection > 0.0:
                    continue
                holder = st.player(st.ball.possessing_team, st.ball.possessing_player or "")
                if holder is None:
                    continue
                d = geom.distance(p.x, p.y, holder.x, holder.y)
                if d <= TACKLE_MAX:
                    if d < TACKLE_CLOSE:
                        st.ball.possessing_team = team
                        st.ball.possessing_player = pid
                        st.ball.protection = POSSESSION_PROTECTION
                        st.events.append("tackle:win")
                    else:
                        # Slide: knock loose at 10 m/s away from the tackler.
                        ang = geom.angle_to(holder.x, holder.y, p.x, p.y)
                        st.ball.possessing_team = None
                        st.ball.possessing_player = None
                        st.ball.vx = math.cos(ang) * 10.0
                        st.ball.vy = math.sin(ang) * 10.0
                        p.grounded = SLIDE_GROUNDED
                        st.events.append("tackle:slide")
                    p.cooldown = ACTION_COOLDOWN
                    holder.cooldown = ACTION_COOLDOWN
            elif intent.act == "slap":
                if p.role == "goalkeeper":
                    continue
                px, py, pf = p.x, p.y, p.facing
                # Pick nearest opponent in range.
                candidates = [p2 for p2 in st.players if p2.team != team and p2.grounded <= 0.0 and geom.distance(px, py, p2.x, p2.y) <= SLAP_MAX]
                if candidates:
                    close = sorted(candidates, key=lambda p2: (geom.distance(px, py, p2.x, p2.y), p2.pid))
                    q = close[0]
                    d = geom.distance(px, py, q.x, q.y)
                    clean = d <= SLAP_CLEAN and geom.facing_diff(pf, q.x, q.y, px, py) <= math.radians(SLAP_FACING_DEG)
                    if clean:
                        # Knockdown + spill ball if carried.
                        q.grounded = SLAP_KNOCKDOWN
                        if st.ball.possessing_team == q.team and st.ball.possessing_player == q.pid:
                            st.ball.possessing_team = None
                            st.ball.possessing_player = None
                            st.ball.vx = math.cos(pf) * 8.0
                            st.ball.vy = math.sin(pf) * 8.0
                        st.events.append("slap:down")
                    elif not (q.grounded > 0.0):
                        q.x, q.y = geom.clamp_point(q.x + (q.x - px) / max(d, 1e-6) * 0.25, q.y + (q.y - py) / max(d, 1e-6) * 0.25)
                        st.events.append("slap:stagger")
                    p.cooldown = SLAP_COOLDOWN

    def _integrate_ball(self, st: PlanState, decay: float) -> None:
        if st.ball.possessing_team is not None:
            return
        steps = 1
        for _ in range(steps):
            st.ball.x += st.ball.vx * self.dt
            st.ball.y += st.ball.vy * self.dt
            hit: str | None = None
            if st.ball.x <= 0.0:
                if GOAL_LOW_Y <= st.ball.y <= GOAL_HIGH_Y:
                    st.score_them += 1
                    st.events.append("goal_conceded")
                    st.ball.x = 0.0
                    st.ball.possessing_team = None
                    st.ball.possessing_player = None
                    st.ball.vx = st.ball.vy = 0.0
                    self._kickoff_reset(st, "us")
                    return
                st.ball.x = 0.0
                hit = "left"
            elif st.ball.x >= PITCH_LENGTH:
                if GOAL_LOW_Y <= st.ball.y <= GOAL_HIGH_Y:
                    st.score_us += 1
                    st.events.append("goal")
                    st.ball.x = PITCH_LENGTH
                    st.ball.possessing_team = None
                    st.ball.possessing_player = None
                    st.ball.vx = st.ball.vy = 0.0
                    self._kickoff_reset(st, "them")
                    return
                st.ball.x = PITCH_LENGTH
                hit = "right"
            if st.ball.y <= 0.0:
                st.ball.y = 0.0
                hit = "bottom"
            elif st.ball.y >= PITCH_WIDTH:
                st.ball.y = PITCH_WIDTH
                hit = "top"
            if hit is not None:
                st.ball.vx, st.ball.vy = geom.wall_bounce(st.ball.vx, st.ball.vy, hit)
            st.ball.vx *= decay
            st.ball.vy *= decay
        self._loose_ball_control(st)

    def _kickoff_reset(self, st: PlanState, kickoff_team: str) -> None:
        """Engine restart after a goal (RULES.md §127-131).

        Players reset to their formation; the conceding team's striker receives
        the ball at the centre spot. The two-second restart hold is a
        presentation phase in the engine and consumes no active time, so the
        model resumes play immediately at the next decision.
        """
        for p in st.players:
            p.vx = p.vy = 0.0
            p.facing = 0.0
            p.can_act = True
            p.cooldown = 0.0
            p.grounded = 0.0
        for team, pid, role, x, y in st.base_positions:
            p = st.player(team, pid)
            if p is not None:
                p.x, p.y = x, y
        kicker = st.player(kickoff_team, "st")
        if kicker is None:
            out = st.outfield(kickoff_team)
            kicker = out[0] if out else None
        st.ball.x, st.ball.y = PITCH_LENGTH / 2.0, PITCH_WIDTH / 2.0
        st.ball.vx = st.ball.vy = 0.0
        st.ball.protection = POSSESSION_PROTECTION
        if kicker is not None:
            st.ball.possessing_team = kickoff_team
            st.ball.possessing_player = kicker.pid
        st.events.append("kickoff")

    def _loose_ball_control(self, st: PlanState) -> None:
        speed = math.hypot(st.ball.vx, st.ball.vy)
        eligible = [
            p
            for p in st.players
            if p.grounded <= 0.0
            and p.cooldown <= 0.0
            and p.role != "goalkeeper"
            and speed <= BALL_CONTROL_MAX_SPEED
            and geom.distance(p.x, p.y, st.ball.x, st.ball.y) <= BALL_CONTROL_RADIUS
        ]
        if not eligible:
            return
        winner = min(eligible, key=lambda p: (geom.distance(p.x, p.y, st.ball.x, st.ball.y), p.pid))
        st.ball.possessing_team = winner.team
        st.ball.possessing_player = winner.pid
        st.ball.protection = POSSESSION_PROTECTION

    def _gk_handling(self, st: PlanState) -> None:
        for team in ("us", "them"):
            gk = st.gk(team)
            if gk is None:
                continue
            own_fifth = gk.x <= PITCH_LENGTH * 0.2
            if not own_fifth:
                continue
            if st.ball.possessing_team == team:
                continue
            if st.ball.possessing_team == ("them" if team == "us" else "us"):
                if geom.distance(gk.x, gk.y, st.ball.x, st.ball.y) <= GK_REACH:
                    st.ball.possessing_team = team
                    st.ball.possessing_player = gk.pid
                    st.events.append("gk:catch")
                    return
            # Loose ball: swept-path check approximated by endpoint distance.
            if st.ball.possessing_team is None and geom.distance(gk.x, gk.y, st.ball.x, st.ball.y) <= GK_REACH:
                st.ball.possessing_team = team
                st.ball.possessing_player = gk.pid
                st.events.append("gk:catch")

    def _gk_auto_distribute(self, st: PlanState) -> None:
        """Engine-forced distribution: a goalkeeper that still holds the ball
        after GK_HOLD_AUTO sends it at GK_AUTO_SPEED toward its widest
        outfield teammate and picks up a 0.5 s cooldown (RULES.md §90)."""
        owner_gk = None
        if st.ball.possessing_team in ("us", "them") and st.ball.possessing_player:
            gk = st.gk(st.ball.possessing_team)
            if gk is not None and st.ball.possessing_player == gk.pid:
                owner_gk = gk
        if owner_gk is None:
            st.ball.gk_hold = 0.0
            return
        st.ball.gk_hold += self.dt
        if st.ball.gk_hold < GK_HOLD_AUTO:
            return
        out = st.outfield(owner_gk.team)
        wide = [q for q in out if q.y <= PITCH_WIDTH * 0.25 or q.y >= PITCH_WIDTH * 0.75]
        target = max(wide or out, key=lambda q: (q.x, q.pid)) if out else None
        if target is not None:
            dx = target.x - owner_gk.x
            dy = target.y - owner_gk.y
            nd = math.hypot(dx, dy)
            if nd > 1e-6:
                st.ball.vx = dx / nd * GK_AUTO_SPEED
                st.ball.vy = dy / nd * GK_AUTO_SPEED
            st.ball.possessing_team = None
            st.ball.possessing_player = None
            st.ball.protection = 0.0
            st.ball.gk_hold = 0.0
            owner_gk.cooldown = max(owner_gk.cooldown, 0.5)
            st.events.append("gk:distribution")