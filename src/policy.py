"""RuntimePolicy — the distilled, fast, deterministic team controller.

This is the tournament runtime brain: observe → world → tactics → policy →
role assignment → intents. It encodes the elite-football priors (width, depth,
support, rest defence, pressing triggers, counterpress, wall usage) as
deterministic scoring functions. No neural network, no LLM, no MCTS at this
layer by default (MCTS is an optional override for critical states).

Everything respects the engine contract (RULES.md §3): only the possessor may
pass/shoot/clear, outfields tackle/slap, the goalkeeper never tackles, and all
numbers are finite and ranged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import geom
from .config import (
    BALL_CONTROL_MAX_SPEED,
    BALL_CONTROL_SAFE_SPEED,
    KICK_MIN_SPEED,
    MAX_RUN_SPEED,
    SLAP_CLEAN,
    SLAP_FACING_DEG,
    TACKLE_CLOSE,
    TACKLE_MAX,
    RuntimeConfig,
    shooting_distance,
)
from .geom import GOAL_CENTER_Y, GOAL_HIGH_Y, GOAL_LOW_Y, OPP_GOAL_X, PITCH_LENGTH, PITCH_WIDTH


# Default distance for nearest opponent fallback (20m = center of pitch width)

def _natural_role(player_id: str) -> str:
    """Determine a player's natural role from their ID.
    
    This is used for shooting decisions to prevent defenders who move up
    from being treated as strikers just because they're in attacking positions.
    """
    if player_id.startswith("gk"):
        return "goalkeeper"
    if player_id.startswith("cd") or player_id.startswith("cb") or player_id.startswith("df") or player_id.startswith("db") or player_id.startswith("dm"):
        return "defender"
    if player_id.startswith("st") or player_id.startswith("cf") or player_id.startswith("fw"):
        return "striker"
    if player_id.startswith("am") or player_id.startswith("cm") or player_id.startswith("lm") or player_id.startswith("rm") or player_id.startswith("wm") or player_id.startswith("mm"):
        return "midfielder"
    if player_id.startswith("wb") or player_id.startswith("lw") or player_id.startswith("rw"):
        return "winger"
    # Default to defender for unknown IDs
    return "defender"


def _is_natural_striker(inp, p) -> bool:
    """Check if player is a natural striker (by ID), not just current dynamic role."""
    natural_role = _natural_role(p.id)
    dynamic_role = inp.roles.get(p.id, "defender")
    # Consider striker if natural role is striker, or dynamic role is striker and in attacking position
    return natural_role == "striker" or (dynamic_role == "striker" and p.x > 30.0)


def go_20() -> float:
    return 20.0


# --- Loose-ball physics (RULES.md is authoritative) -------------------------
# A free ball is only collectable while it travels at no more than 5 m/s, and
# it loses 0.8% of its speed every 60 Hz tick. Every "pass" action leaves the
# ball at KICK_MIN_SPEED (12 m/s) or more, and players run at 8 m/s, so a pass
# is a ball that travels roughly 15 m before anybody is even allowed to touch
# it. Consequences that drive the whole policy:
#   * carrying the ball (dribble, 6.4 m/s, ball glued 0.65 m in front) is the
#     only way to reliably keep possession;
#   * a pass is only worth it when a teammate is already close to the landing
#     spot, open, and moving forward;
#   * winning a loose ball back means meeting it where it has finally slowed
#     below the control limit, not sprinting at its current position.
BALL_DECAY_PER_TICK = 0.992
TICKS_PER_SECOND = 60.0
CONTROL_SAFE_SPEED = BALL_CONTROL_SAFE_SPEED


# --- Goalkeeper positioning -------------------------------------------------
# RULES.md "Goalkeeper handling and diving": automatic handling only applies
# inside the first 20% of the pitch from the keeper's own goal line, so there is
# nothing to gain by going past it and a real risk of a ball running through.
GK_DEFENSIVE_FIFTH_LIMIT = 0.2 * PITCH_LENGTH
# How far toward goal centre the keeper stands from the ball's y, as a fraction
# of the ball-to-centre gap. 1.0 would put him on the goal line and 0.0 would
# leave him level with the ball; 0.62 keeps him goal-side of the midpoint so a
# shot straight down the middle is still covered while a wide crosser's angle is
# narrowed. opponents/counter-elite uses the same 0.62.
GK_BALL_GOAL_BIAS = 0.62
# Default sweeper aggressiveness, matching the opponent's default gk_aggression.
GK_AGGRESSION_DEFAULT = 0.4
# How far to blend the keeper's y toward the ball's predicted crossing point
# when the ball is travelling at our goal. Tuned by A/B, not by intuition.
GK_INTERCEPT_BLEND = 0.75
# How far off his own goal line the keeper may come to meet an incoming ball.
# RULES.md only allows him to control a free ball inside the defensive fifth,
# so there is nothing to gain past 20% of the pitch.
GK_INCOMING_MAX_X = GK_DEFENSIVE_FIFTH_LIMIT - 1.0
# Headings the carrier may probe, in degrees off straight at their goal.
# 0 is straight on; the rest fan out to both flanks so a wall across the
# middle can be walked around instead of into.
DRIBBLE_PROBE_ANGLES = (-70.0, -50.0, -34.0, -20.0, -9.0, 0.0, 9.0, 20.0, 34.0, 50.0, 70.0)
# How close an opponent has to be to a probe line to count as blocking it.
DRIBBLE_LANE_RADIUS = 3.5
# Metres of forward ground given up per unit of blockage. High enough that a
# genuinely open goal is never traded for a wide detour.
DRIBBLE_BLOCKED_PENALTY = 7.0

# --- Low Block Detection ----------------------------------------------------
# A low block is a deep, compact defensive structure where opponent's defensive
# line is deep (x > 40) and they have 3+ players behind the ball in a narrow band.
# This triggers specific attacking patterns: width, crosses, cutbacks, switches.
LOW_BLOCK_DEEP_LINE_THRESHOLD = 40.0
LOW_BLOCK_COMPACT_WIDTH = 22.0
LOW_BLOCK_MIN_DEEP_PLAYERS = 3

# --- High Press Detection ---------------------------------------------------
# High press: 3+ opponent players in our half (x < 30) actively pressing.
HIGH_PRESS_MIN_PLAYERS = 3
HIGH_PRESS_ZONE_THRESHOLD = 30.0


def ball_travel_before_control(speed: float) -> float:
    """Metres a free ball rolls before it slows to a collectable speed."""
    if speed <= CONTROL_SAFE_SPEED:
        return 0.0
    decay = BALL_DECAY_PER_TICK
    ticks = math.log(CONTROL_SAFE_SPEED / speed) / math.log(decay)
    return speed * (1.0 - decay ** ticks) / (1.0 - decay) / TICKS_PER_SECOND


def loose_ball_meeting_point(
    x: float, y: float, vx: float, vy: float
) -> tuple[float, float, float]:
    """Where, and after how long, a loose ball can finally be controlled.

    Returns (x, y, seconds). The point is clamped to what a runner travelling
    at MAX_RUN_SPEED can actually reach before the ball gets there, so an
    unreachable ball yields the best interception spot instead of a wild one.
    """
    speed = math.hypot(vx, vy)
    if speed <= CONTROL_SAFE_SPEED or speed < 1e-6:
        return x, y, 0.0
    decay = BALL_DECAY_PER_TICK
    ticks = math.log(CONTROL_SAFE_SPEED / speed) / math.log(decay)
    seconds = max(0.0, ticks) / TICKS_PER_SECOND
    ux, uy = vx / speed, vy / speed
    reachable = MAX_RUN_SPEED * seconds
    dist = min(ball_travel_before_control(speed), reachable)
    return x + ux * dist, y + uy * dist, seconds


# A pass is released at KICK_MIN_SPEED, so it always rolls this far before it
# may be touched. A receiver standing further away than this can never get it.
MAX_COLLECTABLE_PASS = ball_travel_before_control(KICK_MIN_SPEED) + 1.0

from .physics import (
    pass_collection_point,
    pass_lane_clear,
    pick_shot_target,
    plan_lead_pass,
    shot_beats_keeper,
    shot_lane_clear,
    shot_open_angle,
    wall_pass_target,
    wall_shot_target,
)
from .opponent import OpponentModel
from .state import Ball, GameState, Player, WorldModel
from .tactics import (
    ROLE_DEFENDER,
    ROLE_STRIKER,
    ROLE_WIDE_LEFT,
    ROLE_WIDE_RIGHT,
    PressPlan,
    TacticalState,
)
from .wall import is_near_wall


@dataclass
class PlayerIntent:
    pid: str
    tx: float
    ty: float
    speed: float
    face_x: float
    face_y: float
    action_type: str = "none"
    action_target: tuple[float, float] | None = None
    action_power: float | None = None

    def to_wire(self) -> dict:
        intent: dict = {
            "playerId": self.pid,
            "move": {"target": {"x": _fin(self.tx), "y": _fin(self.ty)}, "speed": _fin(self.speed)},
            "face": {"x": _fin(self.face_x), "y": _fin(self.face_y)},
            "action": {"type": self.action_type},
        }
        if self.action_target is not None:
            intent["action"]["target"] = {
                "x": _fin(self.action_target[0]),
                "y": _fin(self.action_target[1]),
            }
        else:
            intent["action"]["target"] = None
        if self.action_power is not None:
            intent["action"]["power"] = _fin(self.action_power)
        else:
            intent["action"]["power"] = None
        return intent


def _fin(v: float) -> float:
    if not math.isfinite(v):
        return 0.0
    return v


@dataclass
class PolicyInput:
    state: GameState
    world: WorldModel
    config: RuntimeConfig
    tactical_state: TacticalState
    press_plan: PressPlan
    roles: dict[str, str]
    their_possession_ticks: int = 0  # how many decisions they've held the ball
    opp: OpponentModel | None = None  # online opponent profile (§38), may be absent
    time_remaining: float | None = None  # seconds left; falls back to state if None
    score_us: int | None = None
    score_them: int | None = None
    counter_press_active: bool = False  # trigger immediate high press after winning ball
    match_duration: float = 60.0  # total match duration in seconds

    def their_possession_protected(self) -> bool:
        return self.their_possession_ticks < 3  # 0.25 s protection ≈ 2-3 decisions

    @property
    def tr(self) -> float:
        if self.time_remaining is not None:
            return max(0.0, self.time_remaining)
        return max(0.0, self.state.time_remaining)

    @property
    def su(self) -> int:
        if self.score_us is not None:
            return self.score_us
        return self.state.score_us

    @property
    def st_th(self) -> int:
        if self.score_them is not None:
            return self.score_them
        return self.state.score_them

    def late(self) -> bool:
        # Late = last 20% of match duration. Track initial time to infer match duration.
        # If we don't have initial time, fall back to absolute 120s.
        match_duration = getattr(self, '_match_duration', None)
        if match_duration is None:
            return self.tr <= 120.0
        return self.tr <= match_duration * 0.2

    def context(self) -> str:
        return match_context(self.tr, self.su, self.st_th)


@dataclass
class PolicyLog:
    possessions_decided: list[dict] = field(default_factory=list)

    def record(self, entry: dict) -> None:
        if len(self.possessions_decided) < 64:
            self.possessions_decided.append(entry)


def _most_advanced(state: GameState, team: str) -> Player | None:
    members = state.outfield_us() if team == "us" else state.outfield_them()
    if not members:
        return None
    return max(members, key=lambda q: q.x)


def _is_low_block(state: GameState) -> bool:
    """Detect if opponent is in a low block (compact defense deep in their half)."""
    them = state.outfield_them()
    if not them:
        return False
    # Low block: opponent's formation is compact and deep
    # At least 2 players very deep (x > 40), OR all players behind midfield (x > 30)
    deep_count = sum(1 for p in them if p.x > 40.0)
    mid_count = sum(1 for p in them if p.x > 30.0)
    # Also check if they're compact (small spread)
    if len(them) >= 3:
        xs = [p.x for p in them]
        spread = max(xs) - min(xs)
        compact = spread < 20.0
        return (deep_count >= 2 or mid_count >= 4) and compact
    return deep_count >= 2 or mid_count >= 4


def _detect_low_block(state: GameState) -> dict:
    """Detailed low block analysis for adaptive attacking."""
    them = state.outfield_them()
    if not them:
        return {"is_low_block": False}
    
    deep_players = [p for p in them if p.x > LOW_BLOCK_DEEP_LINE_THRESHOLD]
    mid_players = [p for p in them if p.x > 30.0]
    
    # Check compactness
    if len(them) >= 3:
        xs = [p.x for p in them]
        ys = [p.y for p in them]
        x_spread = max(xs) - min(xs)
        y_spread = max(ys) - min(ys)
        compact = x_spread < LOW_BLOCK_COMPACT_WIDTH and y_spread < 28.0
    else:
        compact = False
    
    is_lb = (len(deep_players) >= LOW_BLOCK_MIN_DEEP_PLAYERS or len(mid_players) >= 4) and compact
    
    # Determine which flank is weaker (fewer defenders)
    left_defenders = sum(1 for p in them if p.y < 20.0 and p.x > 30.0)
    right_defenders = sum(1 for p in them if p.y > 20.0 and p.x > 30.0)
    weak_flank = "left" if left_defenders < right_defenders else "right"
    
    # Find gaps between defenders
    left_gap = 20.0 - min((p.y for p in them if p.y < 20.0 and p.x > 30.0), default=0.0)
    right_gap = max((p.y for p in them if p.y > 20.0 and p.x > 30.0), default=40.0) - 20.0
    
    return {
        "is_low_block": is_lb,
        "deep_count": len(deep_players),
        "compact": compact,
        "weak_flank": weak_flank,
        "left_gap": left_gap,
        "right_gap": right_gap,
        "defensive_line": max((p.x for p in them), default=60.0),
    }


def _is_high_press(state: GameState) -> bool:
    """Detect if opponent is high pressing."""
    them = state.outfield_them()
    if not them:
        return False
    pressers = sum(1 for p in them if p.x < HIGH_PRESS_ZONE_THRESHOLD)
    return pressers >= HIGH_PRESS_MIN_PLAYERS


# Multiplicative parameter adjustments per match context (§32). Values > 1
# raise the parameter, < 1 lower it. Only keys present here are affected.
_CONTEXT_MODS: dict[str, dict[str, float]] = {
    "leading_late": {
        "passing_risk": 0.6,
        "verticality": 0.8,
        "shooting_threshold": 1.15,
        "wall_shot_threshold": 1.1,
        "depth": 0.9,
        "support_distance": 1.15,
        "width": 0.9,
        "defensive_line": 0.9,
    },
    "tied_late": {
        "passing_risk": 1.05,
        "shooting_threshold": 0.95,
    },
    "trailing_late": {
        "passing_risk": 1.3,
        "verticality": 1.25,
        "shooting_threshold": 0.8,
        "wall_shot_threshold": 0.85,
        "depth": 1.1,
        "support_distance": 0.85,
        "width": 1.1,
        "defensive_line": 1.1,
    },
}


def match_context(
    time_remaining: float | None,
    score_us: int | None,
    score_them: int | None,
) -> str:
    """§32 score/time context: leading/trailing/tied, late or not. Late means
    <= 120 s remaining."""
    tr = max(0.0, time_remaining or 0.0)
    su = int(score_us or 0)
    st_th = int(score_them or 0)
    late = tr <= 120.0
    diff = su - st_th
    if late:
        if diff > 0:
            return "leading_late"
        if diff < 0:
            return "trailing_late"
        return "tied_late"
    if diff > 0:
        return "leading"
    if diff < 0:
        return "trailing"
    return "tied"


class PolicyController:
    log: PolicyLog

    def __init__(self) -> None:
        self.log = PolicyLog()

    def _ctx_mods(self, inp: PolicyInput) -> dict[str, float]:
        return _CONTEXT_MODS.get(inp.context(), {})

    def _cfg(self, inp: PolicyInput, key: str, default: float) -> float:
        """Parameter read with §32 context modulation applied.
        
        Reads from inp.config.genome dict, not from config attributes.
        """
        v = inp.config.genome.get(key, default)
        mods = self._ctx_mods(inp)
        return v * mods.get(key, 1.0)

    def decide(self, inp: PolicyInput) -> dict[str, PlayerIntent]:
        state: GameState = inp.state
        intents: dict[str, PlayerIntent] = {}
        ours = state.outfield_us()

        if not ours:
            return intents

        if inp.tactical_state == TacticalState.KICKOFF:
            return self._kickoff_intents(inp)

        possessor = state.our_possessor()
        for p in ours:
            if possessor is not None and p.id == possessor.id:
                intent = self._decide_possessor(inp, p)
            else:
                intent = self._decide_off_ball(inp, p, possessor)
            intents[p.id] = intent

        gk = state.goalkeeper_us()
        if gk is not None:
            intents[gk.id] = self._decide_goalkeeper(inp, gk)

        self._enforce_press_actions(inp, intents)
        self._resolve_ball_carrier_conflicts(inp, intents)
        return intents

    # ------------------------------------------------------------------ #
    # Goalkeeper - Elite Level
    # ------------------------------------------------------------------ #
    def _decide_goalkeeper(self, inp: PolicyInput, gk: Player) -> PlayerIntent:
        state = inp.state
        ball = state.ball
        allows_act = gk.can_act

        if ball.possessing_team == "us" and ball.possessing_player == gk.id:
            if allows_act:
                return self._decide_gk_possession(inp, gk)
            # Hold; engine distributes automatically after 1.25 s.
            return PlayerIntent(gk.id, gk.x, gk.y, 0.0, ball.x, ball.y, "none")

        # Elite GK positioning: track ball between posts with advanced anticipation
        ty = self._gk_target_y(inp, gk, ball)

        # A ball travelling at our goal outranks every other case. Note this is
        # checked *before* the possession branch: once a shot is struck the ball
        # is loose, so `possessing_team` is "us"/"them"/"null" rather than
        # "them", and a handler placed inside the "they have it" branch would
        # never see the shot it exists to save. Measured: with the check nested
        # under `possessing_team == "them"` the keeper conceded 7.0 a match;
        # hoisted above it, 0.55.
        lead_y = self._gk_intercept_y(inp, gk, ball)
        if lead_y is not None:
            return self._gk_incoming(gk, ball, lead_y)

        if ball.possessing_team == "them":
            # Analyze threat level
            their_possessor = state.their_possessor()
            threat = self._assess_threat(inp, state, ball, their_possessor)

            if threat["is_1v1"] and allows_act and their_possessor is not None:
                # 1v1 situation: come out aggressively, narrow angle
                return self._gk_one_v_one(inp, gk, ball, their_possessor, threat)
            elif threat["is_through_ball"] and allows_act:
                # Through ball anticipated: sweep up
                return self._gk_sweep(inp, gk, ball, threat)
            elif threat["is_cross"] and allows_act:
                # Cross coming: position for claim/punch
                return self._gk_cross(inp, gk, ball, threat)
            else:
                # Standard positioning with angle play
                return self._gk_standard_position(inp, gk, ball, threat, ty)
        elif ball.x > 14.0:
            # Ball in their half: high start position for sweeps
            tx = 2.5
            speed = 0.65 if abs(ball.y - gk.y) > 1.0 else 0.3
        else:
            # Ball in our half but not possessed by us
            tx = geom.clamp(ball.x - 1.5, 1.0, 5.0)
            speed = 0.65 if abs(ball.y - gk.y) > 1.0 else 0.3
        return PlayerIntent(gk.id, tx, ty, speed, ball.x, ball.y, "none")

    def _gk_target_y(self, inp: PolicyInput, gk: Player, ball: Ball) -> float:
        """Position the keeper on the bisector between the ball and goal centre.

        The old version clamped the ball's y into the goal mouth (17.6..22.4)
        and then nudged it by 1.5 m when the ball was wide. That confines the
        keeper to a 4.8 m band in the middle of a 6 m goal, so any cross or
        shot from a wide angle simply goes round him, and the anticipation term
        only ever moved him 1.5 m when the ball was already in the corner.

        Standing on the bisector instead means he always sees the shot along
        the shortest line from the ball to the goal, and the 0.62 factor keeps
        him goal-side of the midpoint so a fast shot down the middle is still
        covered. For a ball at y=2 that puts him at y=8.8 rather than 17.6,
        which is what actually narrows the angle from a wide crosser.

        Modelled on the keeper positioning in opponents/counter-elite, whose
        `_gk_plan` uses `_lerp(by, GOAL_CENTER_Y, 0.62)`.
        """
        state = inp.state
        ty = ball.y + (GOAL_CENTER_Y - ball.y) * GK_BALL_GOAL_BIAS
        ty = geom.clamp(ty, GOAL_LOW_Y - 6.0, GOAL_HIGH_Y + 6.0)

        # Still shade toward a runner in the box: a breakaway beats a pure
        # bisector, and 0.22 is enough to cover the near post without giving
        # up the far one.
        if ball.possessing_team == "them":
            striker = state.their_possessor()
            if striker is not None and striker.x > 40.0:
                run_y = geom.clamp(striker.y, GOAL_LOW_Y + 0.5, GOAL_HIGH_Y - 0.5)
                ty = ty * 0.78 + run_y * 0.22

        return ty

    def _gk_incoming(self, gk: Player, ball: Ball, lead_y: float) -> PlayerIntent:
        """A ball is travelling at our goal: stand on its path and sprint.

        The old behaviour was to hold the bisector at `0.7` speed, easing to
        `0.4` once the ball came within 2 m -- i.e. the keeper slowed down
        exactly as the ball closed on him. Traced against
        reference-strikers, his lateral speed was 0.7 m/s where 8 m/s was
        available, and he finished 1.76 m off a ball he can control at 1.65 m:
        a miss by 11 cm, seven times out of seven.

        He has the whole flight to close the gap: 8 ticks inside his own
        defensive fifth at ~1.1 m of travel per tick is 3.4 m of lateral
        movement, far more than the ~1.7 m this needs. So stand on the
        predicted crossing point and run at full speed.
        """
        depth = ball.x * (0.22 + 0.16 * GK_AGGRESSION_DEFAULT)
        tx = geom.clamp(depth, 2.5, GK_INCOMING_MAX_X)
        return PlayerIntent(gk.id, tx, lead_y, 1.0, ball.x, ball.y, "none")

    def _gk_intercept_y(self, inp: PolicyInput, gk: Player, ball: Ball) -> float | None:
        """Where the ball will cross the keeper's own line, not where it is.

        RULES.md lets the keeper take a free ball "of any speed" whose swept
        path passes within 1.65 m, but only inside his defensive fifth. The
        bisector in `_gk_target_y` tracks where the ball *is*, and a shot
        drifts on the way in: against reference-strikers the ball crossed from
        y=19.6 at x=19 to y=17.4 by x=3.3, so the keeper standing on the
        bisector at y=19.7 was 2.3 m off the ball's path -- 0.65 m outside his
        reach -- and the shot went in. He scored 7 of 7.

        Leading the ball fixes it: solve for the time the ball reaches the
        keeper's x, then stand on the y it will be at by then. He has the whole
        flight (0.5-0.9 s, 4-7 m at 8 m/s) to get there, which is far more than
        the ~1.3 m of correction this needs.
        """
        if ball.vx >= -0.5 or gk.x > GK_DEFENSIVE_FIFTH_LIMIT:
            return None  # not coming at us, or we may not leave the fifth
        t = (gk.x - ball.x) / ball.vx
        if t <= 0.0 or t > 1.5:
            return None
        y_at_gk = ball.y + ball.vy * t
        return geom.clamp(y_at_gk, GOAL_LOW_Y - 3.0, GOAL_HIGH_Y + 3.0)

    def _assess_threat(self, inp: PolicyInput, state: GameState, ball: Ball, possessor: Player | None) -> dict:
        """Analyze the current attacking threat."""
        threat = {
            "is_1v1": False,
            "is_through_ball": False,
            "is_cross": False,
            "is_long_shot": False,
            "ball_x": ball.x,
            "ball_y": ball.y,
            "possessor": possessor,
            "closest_defender_dist": float('inf'),
        }
        
        if possessor is None:
            return threat
        
        # Find closest defender to the ball
        for q in state.outfield_us():
            if q.id == "gk":
                continue
            d = geom.distance(q.x, q.y, ball.x, ball.y)
            if d < threat["closest_defender_dist"]:
                threat["closest_defender_dist"] = d
        
        # 1v1: attacker past last defender, in our half, no cover
        if ball.x < 18.0 and threat["closest_defender_dist"] > 6.0:
            threat["is_1v1"] = True
        
        # Through ball: ball played behind defense, attacker running onto it
        if ball.vx < -2.0 and ball.x < 25.0:  # Ball moving toward our goal
            for q in state.outfield_them():
                if q.x > ball.x and q.x - ball.x > 5.0:
                    threat["is_through_ball"] = True
                    break
        
        # Cross: ball wide and moving toward goal line
        if (ball.y < 8.0 or ball.y > 32.0) and ball.x > 40.0:
            threat["is_cross"] = True
        
        # Long shot: attacker in shooting range with time
        if ball.x > 30.0 and ball.x < 45.0 and threat["closest_defender_dist"] > 4.0:
            threat["is_long_shot"] = True
        
        return threat

    def _gk_one_v_one(self, inp: PolicyInput, gk: Player, ball: Ball, possessor: Player, threat: dict) -> PlayerIntent:
        """Aggressive 1v1: come out, make yourself big, force shooter wide."""
        # Optimal position: 3-4m off line, cutting angle
        tx = max(3.0, min(6.0, ball.x - 3.0))
        
        # Shade toward near post to cut angle
        if ball.y < 20.0:
            ty = GOAL_LOW_Y + 1.5
        else:
            ty = GOAL_HIGH_Y - 1.5
        
        # Sprint out at full speed
        speed = 1.0
        
        # Face the ball
        face_x, face_y = ball.x, ball.y
        
        # If very close and ball not moving fast, consider dive/smother
        dist = geom.distance(gk.x, gk.y, ball.x, ball.y)
        if dist < 2.5 and ball.vx > -1.0:
            # Stay tall, make body big
            pass
        
        return PlayerIntent(gk.id, tx, ty, speed, face_x, face_y, "none")

    def _gk_sweep(self, inp: PolicyInput, gk: Player, ball: Ball, threat: dict) -> PlayerIntent:
        """Sweep up through balls: intercept before attacker reaches it."""
        # Predict where ball will be collectable
        from .policy import loose_ball_meeting_point
        mx, my, secs = loose_ball_meeting_point(ball.x, ball.y, ball.vx, ball.vy)
        
        # Only sweep if we can get there before attacker
        gk_dist = geom.distance(gk.x, gk.y, mx, my)
        gk_time = gk_dist / (MAX_RUN_SPEED * 0.8)  # GK slower than outfield
        
        if secs > gk_time + 0.3:  # We have time
            tx = geom.clamp(mx, 2.0, 12.0)
            ty = geom.clamp(my, GOAL_LOW_Y + 1.0, GOAL_HIGH_Y - 1.0)
            speed = 1.0
        else:
            # Can't get there: drop back to standard position
            return self._gk_standard_position(inp, gk, ball, threat, self._gk_target_y(inp, gk, ball))
        
        return PlayerIntent(gk.id, tx, ty, speed, ball.x, ball.y, "none")

    def _gk_cross(self, inp: PolicyInput, gk: Player, ball: Ball, threat: dict) -> PlayerIntent:
        """Position for cross: central, ready to claim or punch."""
        # Stand in the "corridor" between 6-yard box and penalty spot
        tx = geom.clamp(6.0, 3.0, 10.0)
        
        # Shade to the side the cross comes from
        if ball.y < 20.0:
            ty = GOAL_CENTER_Y - 2.0
        else:
            ty = GOAL_CENTER_Y + 2.0
        
        speed = 0.8
        face_x, face_y = ball.x, ball.y
        
        return PlayerIntent(gk.id, tx, ty, speed, face_x, face_y, "none")

    def _gk_standard_position(self, inp: PolicyInput, gk: Player, ball: Ball, threat: dict, ty: float) -> PlayerIntent:
        """Advance up the pitch with the ball, but never leave the defensive fifth.

        The old version sat at x=2.0-2.5 for the whole match regardless of where
        the ball was, only stepping up to 5.0 when the ball was inside our
        defensive third. Standing on the goal line is the worst place for a
        keeper: the further out he is, the narrower the angle an attacker has to
        hit, and at 8 m/s he can recover the ground.

        opponents/counter-elite scales his depth with the ball instead
        (`bx * (0.22 + 0.16 * aggression)`), clamped to stay inside the
        defensive fifth, which is the same idea. RULES.md "Goalkeeper handling
        and diving": he can only control a free ball inside the first 20% from
        his own goal line, so going past that buys nothing and risks a ball
        passing him.
        """
        # Come off the line in proportion to how far up the pitch the ball is,
        # so a solo run from 50 m is met well outside the six-yard box.
        depth = ball.x * (0.22 + 0.16 * GK_AGGRESSION_DEFAULT)
        tx = geom.clamp(depth, 1.5, GK_DEFENSIVE_FIFTH_LIMIT - 1.0)

        # Beat them to a dying ball in our own area, but only if no outfielder
        # is closer: otherwise a defender is already on it and coming out just
        # opens a gap behind him.
        if ball.vx < 0.0 and ball.x < GK_DEFENSIVE_FIFTH_LIMIT:
            best_d, _ = self._nearest_competitor(inp, ball.x, ball.y)
            mine = geom.distance(gk.x, gk.y, ball.x, ball.y)
            if mine <= best_d:
                tx = geom.clamp(ball.x, 1.5, GK_DEFENSIVE_FIFTH_LIMIT - 1.0)
                ty = geom.clamp(ball.y, GOAL_LOW_Y - 6.0, GOAL_HIGH_Y + 6.0)

        speed = 0.7 if abs(ball.y - gk.y) > 2.0 else 0.4

        return PlayerIntent(gk.id, tx, ty, speed, ball.x, ball.y, "none")

    def _nearest_competitor(self, inp: PolicyInput, x: float, y: float) -> tuple[float, str]:
        """Distance to the nearest outfielder who could beat us to a ball.

        The keeper is excluded: he is asking whether anyone is closer than he
        is, so counting himself would always answer yes.
        """
        best_d, who = float("inf"), ""
        for q in inp.state.outfield_us():
            d = geom.distance(q.x, q.y, x, y)
            if d < best_d:
                best_d, who = d, q.id
        return best_d, who

    def _decide_gk_possession(self, inp: PolicyInput, gk: Player) -> PlayerIntent:
        state = inp.state
        ball = state.ball
        teammates = state.outfield_us()
        
        # Quick counter-attack check: if opponent is pushed up and we have a fast outlet
        their_deepest = min((q.x for q in state.outfield_them()), default=60.0)
        if their_deepest > 35.0:  # Opponent high line
            # Look for striker/winger making a run behind
            for t in teammates:
                role = inp.roles.get(t.id, ROLE_DEFENDER)
                if role == ROLE_STRIKER and t.x > 40.0:
                    # Check if pass lane is clear
                    opponents_pos = [(o.x, o.y) for o in state.outfield_them()]
                    if pass_lane_clear(gk.x, gk.y, t.x, t.y, opponents_pos, margin=0.15):
                        plan = self._safe_pass_plan(inp, gk, t)
                        intent = PlayerIntent(gk.id, plan[0], plan[1], 0.5, ball.x, ball.y, "pass", (plan[0], plan[1]), plan[2])
                        return self._veto_own_goal(inp, gk, intent)
        
        # Standard distribution: find best open teammate.
        #
        # Deliberately scored on the space around the teammate's *current*
        # position, not on the ball's landing point. The outfield version in
        # _collectable_pass has to aim at the collection point, because a pass
        # aimed at someone's feet sails 15.6 m over their head. The keeper is
        # the opposite case, and aiming at the collection point was measured
        # and reverted: against the reference side it collapsed our attack
        # from 7.0 to 2.5 goals per match (280-0 -> 101-0 over 40 matches)
        # while adding only 3.4 points of win rate against Vanguard.
        #
        # A keeper pass is a long ball by nature and a forward can run onto
        # one. Solving for the collection point instead lands the ball ~15.6 m
        # beyond the receiver, usually in the opponent's half where we have
        # nobody, so we trade a chance to build an attack for a ball we lose.
        # Nothing orders a teammate to go and collect it (the receiver
        # meet-order gap in docs/SCORING.md), so it is simply lost. The
        # sideways hoof below is left as it was for the same reason: it is
        # ugly, but it is not the thing that broke the attack.
        best = None
        best_score = -1e9
        for t in teammates:
            plan = self._safe_pass_plan(inp, gk, t)
            land_x, land_y = t.x, t.y
            open_d = min(
                (geom.distance(land_x, land_y, o.x, o.y) for o in state.outfield_them()),
                default=go_20(),
            )
            progress = land_x - gk.x
            travel = geom.distance(gk.x, gk.y, land_x, land_y)

            # Prefer forward passes, but also value wide options for switching play
            if land_x < gk.x + 2.0:
                progress *= 0.5

            # Bonus for wingers when we want to switch play
            role = inp.roles.get(t.id, ROLE_DEFENDER)
            wide_bonus = 0.0
            if role in (ROLE_WIDE_LEFT, ROLE_WIDE_RIGHT):
                wide_bonus = 3.0 if land_x > 25.0 else 1.0

            # Penalize if the landing spot is marked tightly
            mark_penalty = max(0.0, 5.0 - open_d)

            # Pass lane clearance to the landing spot, not to the receiver.
            opponents_pos = [(o.x, o.y) for o in state.outfield_them()]
            lane_clear = pass_lane_clear(gk.x, gk.y, land_x, land_y, opponents_pos, margin=0.12)
            if not lane_clear and travel > 15.0:
                continue  # Don't force long passes through traffic

            score = progress * 1.5 + open_d * 1.5 - travel * 0.2 + wide_bonus - mark_penalty
            if score > best_score:
                best_score = score
                best = (t, plan)

        if best is not None and best_score > -3.0:
            _, plan = best
            intent = PlayerIntent(gk.id, plan[0], plan[1], 0.5, ball.x, ball.y, "pass", (plan[0], plan[1]), plan[2])
            return self._veto_own_goal(inp, gk, intent)
        
        # No good short pass: try wall pass for switching play
        for t in teammates:
            role = inp.roles.get(t.id, ROLE_DEFENDER)
            if role in (ROLE_WIDE_LEFT, ROLE_WIDE_RIGHT) and t.x > 20.0:
                wall_pass = wall_pass_target(gk.x, gk.y, t.x, t.y)
                if wall_pass:
                    cx, cy, power = wall_pass
                    # Verify wall pass lane
                    opponents_pos = [(o.x, o.y) for o in state.outfield_them()]
                    if pass_lane_clear(gk.x, gk.y, cx, cy, opponents_pos, margin=0.15):
                        intent = PlayerIntent(gk.id, cx, cy, 0.5, ball.x, ball.y, "pass", (cx, cy), power)
                        return self._veto_own_goal(inp, gk, intent)
        
        # Last resort: hoof it long into space, away from their keeper and away
        # from the nearest man. The old code instead nudged the ball 4 m to one
        # side at power 0.3, which travels nowhere useful and is intercepted
        # from the front. Try both flanks and keep whichever the opponents are
        # furthest from, rather than always playing down the middle. Same idea
        # as the hoof in opponents/counter-elite's `_gk_plan`.
        ty = 8.0 if ball.y >= 20.0 else 32.0
        tx = 4.0
        intent = PlayerIntent(gk.id, tx, ty, 0.5, OPP_GOAL_X, 20.0, "pass", (tx, ty), 0.3)
        return self._veto_own_goal(inp, gk, intent)

    def _veto_own_goal(self, inp: PolicyInput, p: Player, intent: PlayerIntent) -> PlayerIntent:
        """Block any action that would send the ball toward our own goal (x=0).

        The engine's "clear" action and some passes can erroneously target our
        own goal line (x=0). Veto any action_target with x < ball.x - 2 when
        ball is in our half, or any target x < 10 (deep in our box).
        """
        ball = inp.state.ball
        at = intent.action_target
        if at is None:
            return intent
        target_x = at[0]
        # If target is behind the ball and ball is in our half, or deep in our box: veto
        if (target_x < ball.x - 2.0 and ball.x < 30.0) or target_x < 10.0:
            # Replace with safe dribble forward
            tx, ty, speed = self._dribble_target(inp, p)
            return PlayerIntent(p.id, tx, ty, speed, tx, ty)
        return intent

    # ------------------------------------------------------------------ #
    # Possessor - Goal-First Decision Tree
    # Priority: Shoot → Cross → Cutback → Through → Wall → Switch → Pullback → OneTwo → ThirdMan → GKBypass → Carry → SafePass
    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    # Possessor - Action-Value Decision
    # ===================================================================
    # Evaluates all attack options by expected value and picks the best.
    # Replaces the sequential if-else chain with decision-theoretic approach.
    # ===================================================================
    def _decide_possessor(self, inp: PolicyInput, p: Player) -> PlayerIntent:
        if not p.can_act:
            # Dribble-forward intent still applies.
            tx, ty, speed = self._dribble_target(inp, p)
            return PlayerIntent(p.id, tx, ty, speed, tx, ty)

        # Use action-value evaluation for all attack decisions
        return self._evaluate_attack_actions(inp, p)
    def _evaluate_attack_actions(self, inp: PolicyInput, p: Player) -> PlayerIntent:
        """Evaluate all attack options and return the best one by expected value."""
        state = inp.state
        world = inp.world
        ball = state.ball
        role = inp.roles.get(p.id, ROLE_DEFENDER)
        
        # Get opponent GK position once for all evaluations
        gk = state.goalkeeper_them()
        gkx = gk.x if gk else -1.0
        gky = gk.y if gk else 20.0
        
        candidates: list[tuple[float, PlayerIntent, str]] = []  # (value, intent, reason)
        
# ---- 1. SHOOT ----
        # Compute shot directly using pick_shot_target (bypasses _shot_choice GK HELL blocking)
        gk = state.goalkeeper_them()
        gkx = gk.x if gk else -1.0
        gky = gk.y if gk else 20.0
        dist_goal = OPP_GOAL_X - p.x
        max_dist = shooting_distance(self._cfg(inp, "shooting_threshold", 0.5))
        
        # Use natural striker check (based on player ID) to prevent defenders from shooting
        is_nat_striker = _is_natural_striker(inp, p)
        
        if dist_goal <= max_dist:
            target = pick_shot_target(p.x, p.y, gkx, gky)
            beats = shot_beats_keeper(p.x, p.y, gkx, gky, target.y, dist_goal, target.power)
            # Only shoot if: (1) beats keeper, OR (2) natural striker inside box
            is_finisher = _is_natural_striker(inp, p)
            inside_box = dist_goal <= 11.0
            is_finisher = _is_natural_striker(inp, p)
            if not beats and not (is_finisher and inside_box):
                shoot_value = 0.0  # Don't shoot
            else:
                # Base value: higher if beats keeper
                base_value = 0.8 if beats else 0.15
                # Minimum floor for natural strikers inside box
                min_shoot_value = 10.0 if (_is_natural_striker(inp, p) and inside_box) else 0.0
                shoot_value = max(0.0, base_value * 100.0 - dist_goal * 0.4)
                if shoot_value > 0:
                    intent = PlayerIntent(p.id, p.x, p.y, 0.4, OPP_GOAL_X, GOAL_CENTER_Y, "shoot", (target.x, target.y), target.power)
                    candidates.append((shoot_value, intent, "shoot"))
        
        # ---- 1b. TEST THE KEEPER (long shot to pull keeper out / create rebound) ----
        # Only strikers inside the box should test the keeper from distance
        inside_box = dist_goal <= 11.0
        is_finisher = _is_natural_striker(inp, p)
        if _is_natural_striker(inp, p) and dist_goal <= max_dist * 1.5 and dist_goal > 15.0:
            # Long shot to test keeper / create rebound (strikers only, and only if inside box or beats keeper)
            test_target = pick_shot_target(p.x, p.y, gkx, gky)
            test_beats = shot_beats_keeper(p.x, p.y, gkx, gky, test_target.y, dist_goal, test_target.power)
            if test_beats or inside_box:
                test_value = 12.0  # Higher value for testing keeper
                intent = PlayerIntent(p.id, p.x, p.y, 0.4, OPP_GOAL_X, GOAL_CENTER_Y, "shoot", (test_target.x, test_target.y), test_target.power)
                candidates.append((12.0, intent, "test_keeper"))
        
        # ---- 1b. REBOUND SETUP ----
        # Shoot at keeper's body to create rebound (strikers only)
        gk = state.goalkeeper_them()
        if _is_natural_striker(inp, p) and gk is not None and dist_goal < 20.0 and p.can_act:
            gkx, gky = gk.x, gk.y
            # Aim at keeper's body (center of goal)
            tx, ty = OPP_GOAL_X, GOAL_CENTER_Y
            power = 0.85
            opp_pos = [(o.x, o.y) for o in state.outfield_them()]
            if pass_lane_clear(p.x, p.y, tx, ty, opp_pos, margin=0.15):
                rebound_value = 45.0
                intent = PlayerIntent(p.id, p.x, p.y, 0.4, tx, ty, "shoot", (tx, ty), power)
                candidates.append((rebound_value, intent, "rebound_setup"))
        
        # ---- 1b. SECOND BALL ANTICIPATION ----
        # If ball is loose in dangerous area, anticipate where it goes
        ball = state.ball
        if ball.possessing_team is None and ball.x > 40.0:
            mx, my, secs = loose_ball_meeting_point(ball.x, ball.y, ball.vx, ball.vy)
            mx = geom.clamp(mx, 0.5, PITCH_LENGTH - 0.5)
            my = geom.clamp(my, 0.5, PITCH_WIDTH - 0.5)
            dist = geom.distance(p.x, p.y, mx, my)
            reach_time = dist / (MAX_RUN_SPEED * 0.9)
            if reach_time < secs - 0.2:
                second_ball_value = 35.0
                intent = PlayerIntent(p.id, mx, my, 1.0, mx, my, "none")
                candidates.append((second_ball_value, intent, "second_ball"))
        
        # ---- 2. CROSS FROM WING ----
        if role in (ROLE_WIDE_LEFT, ROLE_WIDE_RIGHT) and p.x > 45.0:
            cross = self._cross_choice(inp, p)
            if cross is not None:
                tx, ty, power = cross
                # Cross value depends on striker position and box occupancy
                cross_value = 25.0  # Base value for creating chance
                intent = PlayerIntent(p.id, p.x, p.y, 0.4, cross[0], cross[1], "pass", (cross[0], cross[1]), cross[2])
                candidates.append((cross_value, intent, "cross"))
        
        # ---- 3. CUTBACK FROM BYLINE ----
        cutback = self._cutback_choice(inp, p)
        if cutback is not None:
            # Cutback creates high-quality chance at edge of box
            cutback_value = 35.0
            intent = PlayerIntent(p.id, p.x, p.y, 0.4, cutback[0], cutback[1], "pass", (cutback[0], cutback[1]), cutback[2])
            candidates.append((cutback_value, intent, "cutback"))
        
        # ---- 4. THROUGH BALL ----
        through = self._through_ball_choice(inp, p)
        if through is not None:
            through_value = 40.0  # High value - breaks defensive line
            intent = PlayerIntent(p.id, p.x, p.y, 0.4, through[0], through[1], "pass", (through[0], through[1]), through[2])
            candidates.append((through_value, intent, "through_ball"))
        
        # ---- 5. WALL PASS ----
        wall = self._wall_pass_choice(inp, p)
        if wall is not None:
            wall_value = 30.0  # Good for breaking lines
            intent = PlayerIntent(p.id, p.x, p.y, 0.4, wall[0], wall[1], "pass", (wall[0], wall[1]), wall[2])
            candidates.append((wall_value, intent, "wall_pass"))
        
        # ---- 6. SWITCH PLAY ----
        switch = self._switch_play_choice(inp, p)
        if switch is not None:
            switch_value = 20.0  # Opens up weak side
            intent = PlayerIntent(p.id, p.x, p.y, 0.4, switch[0], switch[1], "pass", (switch[0], switch[1]), switch[2])
            candidates.append((switch_value, intent, "switch_play"))
        
        # ---- 7. PULL BACK ----
        pullback = self._pullback_choice(inp, p)
        if pullback is not None:
            pullback_value = 28.0  # Creates shot from edge of box
            intent = PlayerIntent(p.id, p.x, p.y, 0.4, pullback[0], pullback[1], "pass", (pullback[0], pullback[1]), pullback[2])
            candidates.append((pullback_value, intent, "pullback"))
        
        # ---- 8. ONE-TWO (High Press Escape) ----
        onetwo = self._onetwo_choice(inp, p)
        if onetwo is not None:
            onetwo_value = 22.0  # Escapes press, maintains possession
            intent = PlayerIntent(p.id, p.x, p.y, 0.4, onetwo[0], onetwo[1], "pass", (onetwo[0], onetwo[1]), onetwo[2])
            candidates.append((onetwo_value, intent, "high_press_onetwo"))
        
        # ---- 9. THIRD MAN RUN ----
        third = self._third_man_choice(inp, p)
        if third is not None:
            third_value = 25.0  # Breaks press with forward run
            intent = PlayerIntent(p.id, p.x, p.y, 0.4, third[0], third[1], "pass", (third[0], third[1]), third[2])
            candidates.append((third_value, intent, "high_press_third_man"))
        
        # ---- 10. GK BYPASS (High Press) ----
        if inp.roles.get(p.id) == "goalkeeper":
            bypass = self._gk_bypass_choice(inp, p)
            if bypass is not None:
                bypass_value = 30.0  # Direct counter-attack
                intent = PlayerIntent(p.id, p.x, p.y, 0.4, bypass[0], bypass[1], "pass", (bypass[0], bypass[1]), bypass[2])
                candidates.append((bypass_value, intent, "high_press_gk_bypass"))
        
        # ---- 11. REBOUND SETUP ----
        rebound = self._rebound_setup_choice(inp, p)
        if rebound is not None:
            rebound_value = 45.0  # Very high value - creates chaos in box
            intent = PlayerIntent(p.id, p.x, p.y, 0.4, rebound[0], rebound[1], "shoot", (rebound[0], rebound[1]), rebound[2])
            candidates.append((rebound_value, intent, "rebound_setup"))
        
        # ---- 12. SECOND BALL / REBOUND ANTICIPATION ----
        second_ball = self._second_ball_choice(inp, p)
        if second_ball is not None:
            second_ball_value = 35.0  # Anticipating loose ball in dangerous area
            intent = PlayerIntent(p.id, p.x, p.y, 0.4, second_ball[0], second_ball[1], "pass", (second_ball[0], second_ball[1]), second_ball[2])
            candidates.append((second_ball_value, intent, "second_ball"))
        
        # ---- 13. WALL SHOT ----
        if is_near_wall(p.x, p.y, margin=6.0) and (OPP_GOAL_X - p.x) > 8.0:
            wall_shot = wall_shot_target(p.x, p.y, gkx if 'gkx' in dir() else -1.0, gky if 'gky' in dir() else 20.0)
            if wall_shot:
                wx, wy, power = wall_shot
                # Verify both legs clear
                opp_pos = [(o.x, o.y) for o in state.outfield_them()]
                if pass_lane_clear(p.x, p.y, wx, wy, opp_pos, margin=0.12):
                    if pass_lane_clear(wx, wy, OPP_GOAL_X, wy, opp_pos, margin=0.12):
                        wall_shot_value = 28.0
                        intent = PlayerIntent(p.id, p.x, p.y, 0.4, wx, wy, "shoot", (wx, wy), power)
                        candidates.append((wall_shot_value, intent, "wall_shot"))
        
        # ---- 14. CARRY (Dribble) ----
        tx, ty, speed = self._dribble_target(inp, p)
        carry_value = max(5.0, (OPP_GOAL_X - p.x) * 0.1)  # Progress toward goal
        carry_intent = PlayerIntent(p.id, tx, ty, speed, tx, ty)
        candidates.append((carry_value, carry_intent, "carry_forward"))
        
        # ---- 15. SAFE PASS (last resort) ----
        safe_pass = self._collectable_pass(inp, p)
        if safe_pass is not None:
            best, tx, ty, power = safe_pass
            pass_value = 10.0  # Safe but low value
            intent = PlayerIntent(p.id, p.x, p.y, 0.4, tx, ty, "pass", (tx, ty), power)
            candidates.append((pass_value, intent, f"safe_pass_to_{best}"))
        
        # Select best candidate
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            best_value, best_intent, reason = candidates[0]
            log_entry = {"state": str(inp.tactical_state), "player": p.id, "action": best_intent.action_type, "reason": reason, "value": best_value}
            self.log.record(log_entry)
            return self._veto_own_goal(inp, p, best_intent)
        
        # Fallback
        tx, ty, speed = self._dribble_target(inp, p)
        log_entry = {"state": str(inp.tactical_state), "player": p.id, "action": "dribble", "reason": "fallback_carry"}
        self.log.record(log_entry)
        return PlayerIntent(p.id, tx, ty, speed, tx, ty)
    
    def _cross_choice(self, inp: PolicyInput, p: Player) -> tuple[float, float, float] | None:
        """Cross from wing to striker in box."""
        state = inp.state
        role = inp.roles.get(p.id, ROLE_DEFENDER)
        if role not in (ROLE_WIDE_LEFT, ROLE_WIDE_RIGHT) or p.x < 45.0:
            return None
        
        striker = None
        for t in state.outfield_us():
            if inp.roles.get(t.id) == ROLE_STRIKER:
                striker = t
                break
        if striker and striker.x > 38.0:
            tx = geom.clamp(striker.x + 3.0, 46.0, 54.0)
            ty = striker.y
            plan = plan_lead_pass(p.x, p.y, tx, ty, striker.vx, striker.vy, velocity_weight=0.2)
            if plan.target_x > 0:
                collect_x, collect_y, power, _ = pass_collection_point(
                    p.x, p.y, plan.target_x, plan.target_y, 0.0
                )
                miss = geom.distance(collect_x, collect_y, striker.x, striker.y)
                if miss < 7.0:
                    opp_pos = [(o.x, o.y) for o in state.outfield_them()]
                    if pass_lane_clear(p.x, p.y, collect_x, collect_y, opp_pos, margin=0.12):
                        return (collect_x, collect_y, power)
        return None
    
    def _rebound_setup_choice(self, inp: PolicyInput, p: Player) -> tuple[float, float, float] | None:
        """Set up a rebound opportunity by shooting at keeper's hands/post."""
        state = inp.state
        gk = state.goalkeeper_them()
        if gk is None:
            return None
        
        # Shoot at keeper's body to create rebound
        dist_goal = OPP_GOAL_X - p.x
        if dist_goal < 18.0 and p.can_act:
            gkx, gky = gk.x, gk.y
            # Aim at keeper's body (center of goal)
            tx, ty = OPP_GOAL_X, GOAL_CENTER_Y
            power = 0.85
            # Check if lane is reasonably clear
            opp_pos = [(o.x, o.y) for o in state.outfield_them()]
            if pass_lane_clear(p.x, p.y, tx, ty, opp_pos, margin=0.15):
                return (tx, ty, power)
        return None
    
    def _second_ball_choice(self, inp: PolicyInput, p: Player) -> tuple[float, float, float] | None:
        """Anticipate second ball / rebound in dangerous area."""
        state = inp.state
        ball = state.ball
        
        # If ball is loose in dangerous area, anticipate where it goes
        if ball.possessing_team is None and ball.x > 40.0:
            # Predict where ball will be collectable
            mx, my, secs = loose_ball_meeting_point(ball.x, ball.y, ball.vx, ball.vy)
            mx = geom.clamp(mx, 0.5, PITCH_LENGTH - 0.5)
            my = geom.clamp(my, 0.5, PITCH_WIDTH - 0.5)
            
            # If we can get there first, go for it
            dist = geom.distance(p.x, p.y, mx, my)
            reach_time = dist / (MAX_RUN_SPEED * 0.9)
            if reach_time < secs - 0.2:
                # Position to receive second ball
                return (mx, my, 0.0)  # 0.0 power = move only
        return None
    def _cutback_choice(self, inp: PolicyInput, p: Player) -> tuple[float, float, float] | None:
        """Cutback from byline when no clear forward path exists."""
        state = inp.state
        role = inp.roles.get(p.id, ROLE_DEFENDER)
        if role not in (ROLE_WIDE_LEFT, ROLE_WIDE_RIGHT) or p.x < 46.0:
            return None
        
        # Check if we have a clear forward path (shot, cross, or dribble)
        if self._shot_choice(inp, p) is not None:
            return None
        
        striker = None
        for t in state.outfield_us():
            if inp.roles.get(t.id) == ROLE_STRIKER:
                striker = t
                break
        if striker and striker.x > 38.0:
            opp_pos = [(o.x, o.y) for o in state.outfield_them()]
            if pass_lane_clear(p.x, p.y, striker.x, striker.y, opp_pos, margin=0.12):
                return None
        
        # Check dribble path
        dribble_target = self._dribble_target(inp, p)
        if dribble_target[0] > p.x + 3.0:
            opp_pos = state.outfield_them()
            lane_clear = True
            for o in opp_pos:
                if geom.seg_point_distance_sq(o.x, o.y, p.x, p.y, dribble_target[0], dribble_target[1]) < 2.0 * 2.0:
                    lane_clear = False
                    break
            if lane_clear:
                return None
        
        # No clear forward path -> cutback to edge of box
        for t in state.outfield_us():
            r = inp.roles.get(t.id, ROLE_DEFENDER)
            if r in (ROLE_DEFENDER, ROLE_WIDE_LEFT, ROLE_WIDE_RIGHT) and t.x > 30.0 and t.x < 42.0:
                opp_pos = [(o.x, o.y) for o in state.outfield_them()]
                if pass_lane_clear(p.x, p.y, t.x, t.y, opp_pos, margin=0.12):
                    plan = plan_lead_pass(p.x, p.y, t.x, t.y, t.vx, t.vy, velocity_weight=0.2)
                    if plan.target_x > 0:
                        collect_x, collect_y, power, _ = pass_collection_point(
                            p.x, p.y, plan.target_x, plan.target_y, 0.0
                        )
                        miss = geom.distance(collect_x, collect_y, t.x, t.y)
                        if miss < 7.0 and pass_lane_clear(p.x, p.y, collect_x, collect_y, opp_pos, margin=0.12):
                            return (collect_x, collect_y, power)
        return None

    # -- 4. THROUGH BALL -----------------------------------------------
    def _through_ball_choice(self, inp: PolicyInput, p: Player) -> tuple[float, float, float] | None:
        """Through ball to runner behind high defensive line."""
        state = inp.state
        ball = state.ball
        
        # Only attempt when opponent has high line
        their_deepest = min((q.x for q in state.outfield_them()), default=60.0)
        their_highest = max((q.x for q in state.outfield_them()), default=0.0)
        high_line = their_highest > 35.0
        if not high_line or ball.x < 30.0:
            return None
        
        # Find runner making run behind defence
        for t in state.outfield_us():
            r = inp.roles.get(t.id, ROLE_DEFENDER)
            if r not in (ROLE_STRIKER, ROLE_WIDE_LEFT, ROLE_WIDE_RIGHT):
                continue
            if t.x < their_highest + 2.0:  # Not yet behind line
                continue
            
            # Check if pass lane is clear to the space ahead of runner
            target_x = min(t.x + 5.0, 58.0)
            target_y = t.y
            opp_pos = [(o.x, o.y) for o in state.outfield_them()]
            if pass_lane_clear(p.x, p.y, target_x, target_y, opp_pos, margin=0.15):
                plan = plan_lead_pass(p.x, p.y, target_x, target_y, t.vx, t.vy, velocity_weight=0.25)
                if plan.target_x > 0:
                    collect_x, collect_y, power, _ = pass_collection_point(
                        p.x, p.y, plan.target_x, plan.target_y, 0.0
                    )
                    miss = geom.distance(collect_x, collect_y, t.x, t.y)
                    if miss < 7.0 and pass_lane_clear(p.x, p.y, collect_x, collect_y, opp_pos, margin=0.15):
                        return (collect_x, collect_y, power)
        return None

    # -- 6. SWITCH PLAY ------------------------------------------------
    def _switch_play_choice(self, inp: PolicyInput, p: Player) -> tuple[float, float, float] | None:
        """Quick switch to opposite flank when ball is on one side."""
        state = inp.state
        ball = state.ball
        role = inp.roles.get(p.id, ROLE_DEFENDER)
        
        # Only attempt in attacking half with ball on flank
        if ball.x < 35.0:
            return None
        
        # Find opposite winger
        target_role = ROLE_WIDE_RIGHT if role == ROLE_WIDE_LEFT else ROLE_WIDE_LEFT
        if role not in (ROLE_WIDE_LEFT, ROLE_WIDE_RIGHT):
            # Central player - switch to weaker flank
            lb_info = _detect_low_block(state)
            target_role = ROLE_WIDE_LEFT if lb_info.get("weak_flank") == "left" else ROLE_WIDE_RIGHT
        
        for t in state.outfield_us():
            if inp.roles.get(t.id) == target_role and t.x > 25.0:
                opp_pos = [(o.x, o.y) for o in state.outfield_them()]
                if pass_lane_clear(p.x, p.y, t.x, t.y, opp_pos, margin=0.15):
                    plan = plan_lead_pass(p.x, p.y, t.x, t.y, t.vx, t.vy, velocity_weight=0.25)
                    if plan.target_x > 0:
                        collect_x, collect_y, power, _ = pass_collection_point(
                            p.x, p.y, plan.target_x, plan.target_y, 0.0
                        )
                        travel = geom.distance(p.x, p.y, collect_x, collect_y)
                        if travel > 12.0:  # Must be collectable distance
                            miss = geom.distance(collect_x, collect_y, t.x, t.y)
                            if miss < 7.0:
                                return (collect_x, collect_y, power)
        return None

    # -- 7. PULL BACK --------------------------------------------------
    def _pullback_choice(self, inp: PolicyInput, p: Player) -> tuple[float, float, float] | None:
        """Striker pulls back to edge of box for arriving winger/midfielder."""
        state = inp.state
        role = inp.roles.get(p.id, ROLE_DEFENDER)
        if role != ROLE_STRIKER or p.x < 38.0 or p.x > 46.0:
            return None
        
        # Check if we have a clear shot
        if self._shot_choice(inp, p) is not None:
            return None
        
        # Check dribble path
        dribble_target = self._dribble_target(inp, p)
        has_dribble_path = dribble_target[0] > p.x + 3.0
        if has_dribble_path:
            opp_pos = state.outfield_them()
            lane_clear = True
            for o in opp_pos:
                if geom.seg_point_distance_sq(o.x, o.y, p.x, p.y, dribble_target[0], dribble_target[1]) < 2.0 * 2.0:
                    lane_clear = False
                    break
            if lane_clear:
                return None  # Dribble forward instead
        
        for t in state.outfield_us():
            r = inp.roles.get(t.id, ROLE_DEFENDER)
            if r in (ROLE_WIDE_LEFT, ROLE_WIDE_RIGHT) and t.x > 35.0 and t.x < 44.0:
                opp_pos = [(o.x, o.y) for o in state.outfield_them()]
                if pass_lane_clear(p.x, p.y, t.x, t.y, opp_pos, margin=0.1):
                    plan = plan_lead_pass(p.x, p.y, t.x, t.y, t.vx, t.vy, velocity_weight=0.15)
                    if plan.target_x > 0:
                        collect_x, collect_y, power, _ = pass_collection_point(
                            p.x, p.y, plan.target_x, plan.target_y, 0.0
                        )
                        miss = geom.distance(collect_x, collect_y, t.x, t.y)
                        if miss < 7.0:
                            return (collect_x, collect_y, power)
        return None

    # -- 8. ONE-TWO (High Press Escape) --------------------------------
    def _onetwo_choice(self, inp: PolicyInput, p: Player) -> tuple[float, float, float] | None:
        """Quick one-two with nearby teammate under high press."""
        state = inp.state
        if not _is_high_press(state) or inp.world.pressure_on_ball < 0.4:
            return None
        
        nearby = []
        for t in state.outfield_us():
            if t.id != p.id:
                dist = geom.distance(p.x, p.y, t.x, t.y)
                if dist < 8.0 and t.can_act and t.x > p.x - 5.0:
                    nearby.append((dist, t))
        
        if nearby:
            nearby.sort(key=lambda x: x[0])
            t = nearby[0][1]
            opp_pos = [(o.x, o.y) for o in state.outfield_them()]
            if pass_lane_clear(p.x, p.y, t.x, t.y, opp_pos, margin=0.1):
                plan = plan_lead_pass(p.x, p.y, t.x, t.y, t.vx, t.vy, velocity_weight=0.4)
                if plan.target_x > 0:
                    dist = geom.distance(p.x, p.y, plan.target_x, plan.target_y)
                    power = min(1.0, max(0.0, dist / 33.3))
                    return (plan.target_x, plan.target_y, power)
        return None

    # -- 9. THIRD MAN RUN ----------------------------------------------
    def _third_man_choice(self, inp: PolicyInput, p: Player) -> tuple[float, float, float] | None:
        """Third man run - pass to player with space ahead."""
        state = inp.state
        if not _is_high_press(state) or inp.world.pressure_on_ball < 0.4:
            return None
        
        for t in state.outfield_us():
            if t.id == p.id:
                continue
            # Check if this teammate has space ahead
            space_ahead = True
            for o in state.outfield_them():
                if o.x > t.x and o.x - t.x < 6.0 and abs(o.y - t.y) < 5.0:
                    space_ahead = False
                    break
            if space_ahead and t.x > p.x and t.x < 40.0:
                opp_pos = [(o.x, o.y) for o in state.outfield_them()]
                if pass_lane_clear(p.x, p.y, t.x, t.y, opp_pos, margin=0.12):
                    plan = plan_lead_pass(p.x, p.y, t.x, t.y, t.vx, t.vy, velocity_weight=0.3)
                    if plan.target_x > 0:
                        dist = geom.distance(p.x, p.y, plan.target_x, plan.target_y)
                        power = min(1.0, max(0.0, dist / 33.3))
                        return (plan.target_x, plan.target_y, power)
        return None

    # -- 10. GK BYPASS (High Press) ------------------------------------
    def _gk_bypass_choice(self, inp: PolicyInput, p: Player) -> tuple[float, float, float] | None:
        """GK distribution bypass under high press."""
        state = inp.state
        if not _is_high_press(state):
            return None
        
        for t in state.outfield_us():
            r = inp.roles.get(t.id, ROLE_DEFENDER)
            if r in (ROLE_STRIKER, ROLE_WIDE_LEFT, ROLE_WIDE_RIGHT) and t.x > 30.0:
                opp_pos = [(o.x, o.y) for o in state.outfield_them()]
                if pass_lane_clear(p.x, p.y, t.x, t.y, opp_pos, margin=0.12):
                    plan = plan_lead_pass(p.x, p.y, t.x, t.y, t.vx, t.vy, velocity_weight=0.2)
                    if plan.target_x > 0:
                        dist = geom.distance(p.x, p.y, plan.target_x, plan.target_y)
                        power = min(1.0, max(0.0, dist / 33.3))
                        return (plan.target_x, plan.target_y, power)
        return None

    def _collectable_pass(
        self, inp: PolicyInput, p: Player
    ) -> tuple[str, float, float, float] | None:
        """Best pass whose receiver can realistically get to the ball.

        RULES.md: a free ball is collectable only below 5 m/s, and every pass
        action releases it at 12 m/s or more. The ball therefore always rolls
        at least ~15 m before anybody may touch it, which no one can chase
        down from rest. So a pass is only worth taking when the receiver is
        already near the landing spot, unmarked, with a clear lane, and the
        pass actually moves us forward. Otherwise we keep carrying the ball.
        """
        state = inp.state
        best: tuple[str, float, float, float] | None = None
        best_score = -1e9
        for rid, tx, ty, power in self._receivers(inp, p):
            t = next((q for q in state.outfield_us() if q.id == rid), None)
            if t is None or not t.can_act:
                continue
            travel = geom.distance(p.x, p.y, t.x, t.y)
            # The ball needs ~15 m of roll before it can be controlled, so a
            # receiver further away than that simply never gets the ball.
            if travel > MAX_COLLECTABLE_PASS:
                continue
            nearest = min(
                (geom.distance(t.x, t.y, o.x, o.y) for o in state.outfield_them()),
                default=go_20(),
            )
            if nearest < 5.0:
                continue
            if self._lane_risk(inp, p, t) > 0.35:
                continue
            progress = t.x - p.x
            if progress < -2.0:
                continue  # never hand the ball backwards just to be safe
            score = progress * 2.0 + nearest * 0.8 - travel * 0.5
            if score > best_score:
                best_score = score
                best = (rid, tx, ty, power)
        return best

    def _shot_choice(self, inp: PolicyInput, p: Player) -> tuple[float, float, float] | None:
        state = inp.state
        gk = state.goalkeeper_them()
        dist_goal = OPP_GOAL_X - p.x
        gkx = gk.x if gk is not None else -1.0
        gky = gk.y if gk is not None else 20.0
        max_dist = shooting_distance(self._cfg(inp, "shooting_threshold", 0.5))
        if dist_goal > max_dist:
            return None
        
        opponents = [q.pos for q in state.outfield_them() if q.x > p.x]
        inside_box = dist_goal <= 11.0

        # Wall shot: only when NOT in clear 1v1 (ball close to goal) and near wall
        # In 1v1, direct shot is better - wall shot adds unpredictability
        if is_near_wall(p.x, p.y, margin=6.0) and dist_goal > 8.0:
            wall_shot = wall_shot_target(p.x, p.y, gkx, gky)
            if wall_shot:
                wx, wy, power = wall_shot
                # Verify wall shot lane is clear
                opp_pos = [(o.x, o.y) for o in state.outfield_them()]
                if pass_lane_clear(p.x, p.y, wx, wy, opp_pos, margin=0.12):
                    # Check second leg (wall to goal) is also clear
                    if pass_lane_clear(wx, wy, OPP_GOAL_X, wy, opp_pos, margin=0.12):
                        return (wx, wy, power)
        
        # Primary trigger: the lane to the aimed corner is free. A single
        # marker on the lane blocks the shot; wide markers do not (§62).
        target = pick_shot_target(p.x, p.y, gkx, gky)
        lane_margin = 0.08 if inside_box else 0.15

        # Does this shot actually have a chance against an automatic keeper?
        #
        # Measured against Vanguard FC (elite): we took 35 shots from a mean
        # 15.1 m and every single one was saved, 26 of them by the centre-backs.
        # The engine makes this inevitable -- the goal is 6 m wide, so the widest
        # angle available against a keeper on the centre line is 3.0 m, his dive
        # reach is 1.65 m, and he keeps shifting across his goal at run speed
        # while the ball flies. Beyond roughly 7 m that shift alone eats the
        # whole angle, so a shot at a set keeper is a standing catch no matter
        # how clean the lane looks.
        #
        # So a free lane is not sufficient reason to shoot. The keeper has to be
        # genuinely beaten: dragged out of his defensive fifth, pulled well wide
        # by the ball, or beaten by a point-blank effort.
        beats_keeper = shot_beats_keeper(
            p.x, p.y, gkx, gky, target.y, dist_goal, target.power
        )
        
        # GK HELL ADAPTATION: Detect elite keepers (very central, quick reactions)
        # and adjust shooting strategy. Opponent GK defends x=60, so check near x=60.
        is_gk_hell = gk is not None and gkx > 50.0 and gkx < 58.0 and abs(gky - 20.0) < 3.0
        if is_gk_hell:
            # Against elite keepers: only shoot from very close range or extreme angles
            # Force the keeper to move by dribbling wide first
            if dist_goal > 8.0 and not inside_box:
                return None
            # Be more aggressive with shooting when keeper is pulled wide
            if abs(gky - 20.0) > 5.0:
                beats_keeper = True  # Keeper out of position = shoot
        
        # The striker is the designated finisher: once he is on the ball inside
        # the box he has nothing better to do, and a saved shot costs no more
        # than a hopeful pass into the same congestion. Defenders and wingers
        # are held to the real test, which is what stops the 26 centre-back
        # efforts from range that produced nothing.
        is_finisher = inp.roles.get(p.id) == ROLE_STRIKER
        if not beats_keeper and not (is_finisher and inside_box):
            return None

        if shot_lane_clear(p.x, p.y, target.x, target.y, opponents, margin=lane_margin):
            power = 0.92 if inside_box else 0.8
            return (target.x, target.y, power)
        
        # Shoot-on-sight in box: if we're in the box with any opening, shoot!
        # Low blocks leave small windows - don't wait for perfect lane.
        open_angle = shot_open_angle(p.x, p.y, opponents)
        if is_near_wall(p.x, p.y, margin=4.0):
            angle_limit = 0.10 if inside_box else 0.22
            if open_angle >= angle_limit:
                power = 0.92 if inside_box else 0.8
                return (target.x, target.y, power)
        
        # Desperation shot from distance when trailing late
        ctx = inp.context()
        if (
            ctx == "trailing_late"
            and dist_goal <= max_dist * 1.2
            and beats_keeper
        ):
            # Lower standards when chasing game
            if open_angle >= 0.05:
                power = 0.85
                return (target.x, target.y, power)
        
        return None

    # ------------------------------------------------------------------ #
    # Low Block Breaking: Cross, Cutback, Switch Play
    # ------------------------------------------------------------------ #
    def _low_block_attack(self, inp: PolicyInput, p: Player) -> tuple[float, float, float, str, str] | None:
        """Generate attacking options specifically for breaking low blocks.
        
        Returns (target_x, target_y, power, action_name, reason) or None.
        Low blocks are deep, compact defenses (defensive line > 40, 3+ deep players).
        We attack them with: wing play, crosses, cutbacks, quick switches.
        Uses pass_collection_point geometry so passes are actually collectable.
        """
        state = inp.state
        ball = state.ball
        role = inp.roles.get(p.id, ROLE_DEFENDER)
        
        lb_info = _detect_low_block(state)
        if not lb_info["is_low_block"]:
            return None
        
        # Only attempt these patterns in the attacking half
        if ball.x < 30.0:
            return None
        
        # 1. WINGER CROSS FROM BYLINE - use collection geometry
        if role in (ROLE_WIDE_LEFT, ROLE_WIDE_RIGHT) and p.x > 42.0:
            # Find striker in box
            striker = None
            for t in state.outfield_us():
                if inp.roles.get(t.id) == ROLE_STRIKER:
                    striker = t
                    break
            if striker and striker.x > 38.0:
                # Cross to striker - aim at collection point ahead of striker
                target_x = striker.x + 4.0  # aim ahead for run onto ball
                target_y = striker.y
                # Use collection geometry like _collectable_pass does
                plan = plan_lead_pass(p.x, p.y, target_x, target_y, striker.vx, striker.vy, velocity_weight=0.25)
                if plan.target_x > 0:
                    collect_x, collect_y, power, _ = pass_collection_point(
                        p.x, p.y, plan.target_x, plan.target_y, 0.0
                    )
                    # Check if striker can reach collection point
                    miss = geom.distance(collect_x, collect_y, striker.x, striker.y)
                    if miss < 7.0:
                        opp_pos = [(o.x, o.y) for o in state.outfield_them()]
                        if pass_lane_clear(p.x, p.y, collect_x, collect_y, opp_pos, margin=0.12):
                            return (collect_x, collect_y, power, "pass", "low_block_cross")
        
        # 2. CUTBACK FROM BYLINE - pass back to edge of box (only if no clear forward path)
        if role in (ROLE_WIDE_LEFT, ROLE_WIDE_RIGHT) and p.x > 46.0:
            # Check if we have a clear forward path (shot, cross, or dribble lane)
            has_forward_path = False
            if self._shot_choice(inp, p) is not None:
                has_forward_path = True
            # Also check if we can cross to striker
            striker = None
            for t in state.outfield_us():
                if inp.roles.get(t.id) == ROLE_STRIKER:
                    striker = t
                    break
            if striker and striker.x > 38.0:
                opp_pos = [(o.x, o.y) for o in state.outfield_them()]
                if pass_lane_clear(p.x, p.y, striker.x, striker.y, opp_pos, margin=0.12):
                    has_forward_path = True
            # Check if dribble path forward is open
            if not has_forward_path:
                dribble_target = self._dribble_target(inp, p)
                # dribble_target is (tx, ty, speed)
                if dribble_target[0] > p.x + 3.0:
                    # Check if lane to dribble target is clear
                    opp_pos = [(o.x, o.y) for o in state.outfield_them()]
                    lane_clear = True
                    for o in opp_pos:
                        if geom.seg_point_distance_sq(o.x, o.y, p.x, p.y, dribble_target[0], dribble_target[1]) < 2.0 * 2.0:
                            lane_clear = False
                            break
                    if lane_clear:
                        has_forward_path = True
            
            if not has_forward_path:
                for t in state.outfield_us():
                    r = inp.roles.get(t.id, ROLE_DEFENDER)
                    if r in (ROLE_DEFENDER, ROLE_WIDE_LEFT, ROLE_WIDE_RIGHT) and t.x > 30.0 and t.x < 42.0:
                        opp_pos = [(o.x, o.y) for o in state.outfield_them()]
                        if pass_lane_clear(p.x, p.y, t.x, t.y, opp_pos, margin=0.12):
                            plan = plan_lead_pass(p.x, p.y, t.x, t.y, t.vx, t.vy, velocity_weight=0.2)
                            if plan.target_x > 0:
                                collect_x, collect_y, power, _ = pass_collection_point(
                                    p.x, p.y, plan.target_x, plan.target_y, 0.0
                                )
                                miss = geom.distance(collect_x, collect_y, t.x, t.y)
                                if miss < 7.0 and pass_lane_clear(p.x, p.y, collect_x, collect_y, opp_pos, margin=0.12):
                                    return (collect_x, collect_y, power, "pass", "low_block_cutback")
        
        # 3. QUICK SWITCH PLAY - long diagonal to opposite flank
        if ball.x > 35.0:
            target_role = ROLE_WIDE_RIGHT if role == ROLE_WIDE_LEFT else ROLE_WIDE_LEFT
            if role not in (ROLE_WIDE_LEFT, ROLE_WIDE_RIGHT):
                weak_flank = lb_info["weak_flank"]
                target_role = ROLE_WIDE_LEFT if weak_flank == "left" else ROLE_WIDE_RIGHT
            
            for t in state.outfield_us():
                if inp.roles.get(t.id) == target_role and t.x > 25.0:
                    opp_pos = [(o.x, o.y) for o in state.outfield_them()]
                    if pass_lane_clear(p.x, p.y, t.x, t.y, opp_pos, margin=0.15):
                        plan = plan_lead_pass(p.x, p.y, t.x, t.y, t.vx, t.vy, velocity_weight=0.25)
                        if plan.target_x > 0:
                            collect_x, collect_y, power, _ = pass_collection_point(
                                p.x, p.y, plan.target_x, plan.target_y, 0.0
                            )
                            # Switch passes need to be long - check travel distance
                            travel = geom.distance(p.x, p.y, collect_x, collect_y)
                            if travel > 12.0:  # Must be collectable distance
                                miss = geom.distance(collect_x, collect_y, t.x, t.y)
                                if miss < 7.0:
                                    return (collect_x, collect_y, power, "pass", "low_block_switch")
        
# 4. PULL BACK TO EDGE OF BOX FOR SHOT (only if no clear forward path)
        if role == ROLE_STRIKER and p.x > 38.0 and p.x < 46.0:
            # Check if we have a clear shot
            if self._shot_choice(inp, p) is not None:
                return None  # Shoot instead
            # Check if we have a clear dribble path forward
            dribble_target = self._dribble_target(inp, p)
            has_dribble_path = dribble_target[0] > p.x + 3.0
            if has_dribble_path:
                opp_pos = [(o.x, o.y) for o in state.outfield_them()]
                lane_clear = True
                for o in opp_pos:
                    if geom.seg_point_distance_sq(o.x, o.y, p.x, p.y, dribble_target[0], dribble_target[1]) < 2.0 * 2.0:
                        lane_clear = False
                        break
                if lane_clear:
                    return None  # Dribble forward instead
            
            for t in state.outfield_us():
                r = inp.roles.get(t.id, ROLE_DEFENDER)
                if r in (ROLE_WIDE_LEFT, ROLE_WIDE_RIGHT) and t.x > 35.0 and t.x < 44.0:
                    opp_pos = [(o.x, o.y) for o in state.outfield_them()]
                    if pass_lane_clear(p.x, p.y, t.x, t.y, opp_pos, margin=0.1):
                        plan = plan_lead_pass(p.x, p.y, t.x, t.y, t.vx, t.vy, velocity_weight=0.15)
                        if plan.target_x > 0:
                            collect_x, collect_y, power, _ = pass_collection_point(
                                p.x, p.y, plan.target_x, plan.target_y, 0.0
                            )
                            miss = geom.distance(collect_x, collect_y, t.x, t.y)
                            if miss < 7.0:
                                return (collect_x, collect_y, power, "pass", "low_block_pullback")
        
        return None

    # ------------------------------------------------------------------ #
    # High Press Resistance: Quick One-Twos, Third Man, Bypass
    # ------------------------------------------------------------------ #
    def _high_press_escape(self, inp: PolicyInput, p: Player) -> tuple[float, float, float, str, str] | None:
        """Generate escape options when opponent is high pressing.
        
        Returns (target_x, target_y, power, action_name, reason) or None.
        High press: 3+ opponent players in our half (x < 30).
        We escape with: quick one-twos, third man runs, direct to striker, GK bypass.
        """
        state = inp.state
        ball = state.ball
        role = inp.roles.get(p.id, ROLE_DEFENDER)
        
        if not _is_high_press(state):
            return None
        
        # Only attempt when under pressure
        if inp.world.pressure_on_ball < 0.4:
            return None
        
        # 1. QUICK ONE-TWO WITH NEARBY TEAMMATE
        # Find closest teammate for a quick layoff (only if they're not behind us)
        nearby = []
        for t in state.outfield_us():
            if t.id != p.id:
                dist = geom.distance(p.x, p.y, t.x, t.y)
                if dist < 8.0 and t.can_act and t.x > p.x - 5.0:  # Not significantly behind
                    nearby.append((dist, t))
        
        if nearby:
            nearby.sort(key=lambda x: x[0])
            t = nearby[0][1]
            # Quick pass to feet, then they pass back or forward
            opp_pos = [(o.x, o.y) for o in state.outfield_them()]
            if pass_lane_clear(p.x, p.y, t.x, t.y, opp_pos, margin=0.1):
                plan = plan_lead_pass(p.x, p.y, t.x, t.y, t.vx, t.vy, velocity_weight=0.4)
                if plan.target_x > 0:
                    dist = geom.distance(p.x, p.y, plan.target_x, plan.target_y)
                    power = min(1.0, max(0.0, dist / 33.3))
                    return (plan.target_x, plan.target_y, power, "pass", "high_press_onetwo")
        
        # 2. THIRD MAN RUN - pass to player who has a forward runner
        for t in state.outfield_us():
            if t.id == p.id:
                continue
            # Check if this teammate has space ahead
            space_ahead = True
            for o in state.outfield_them():
                if o.x > t.x and o.x - t.x < 6.0 and abs(o.y - t.y) < 5.0:
                    space_ahead = False
                    break
            if space_ahead and t.x > p.x and t.x < 40.0:
                opp_pos = [(o.x, o.y) for o in state.outfield_them()]
                if pass_lane_clear(p.x, p.y, t.x, t.y, opp_pos, margin=0.12):
                    plan = plan_lead_pass(p.x, p.y, t.x, t.y, t.vx, t.vy, velocity_weight=0.3)
                    if plan.target_x > 0:
                        dist = geom.distance(p.x, p.y, plan.target_x, plan.target_y)
                        power = min(1.0, max(0.0, dist / 33.3))
                        return (plan.target_x, plan.target_y, power, "pass", "high_press_third_man")
        
        # 3. DIRECT BYPASS TO STRIKER (if they're high up)
        striker = None
        for t in state.outfield_us():
            if inp.roles.get(t.id) == ROLE_STRIKER:
                striker = t
                break
        
        if striker and striker.x > 35.0:
            # Check if we can play over the press
            opp_pos = [(o.x, o.y) for o in state.outfield_them()]
            # High ball over press - less lane risk for long ball
            if pass_lane_clear(p.x, p.y, striker.x, striker.y, opp_pos, margin=0.15):
                plan = plan_lead_pass(p.x, p.y, striker.x, striker.y, striker.vx, striker.vy, velocity_weight=0.2)
                if plan.target_x > 0:
                    dist = geom.distance(p.x, p.y, plan.target_x, plan.target_y)
                    power = min(1.0, max(0.0, dist / 33.3))
                    return (plan.target_x, plan.target_y, power, "pass", "high_press_bypass")
        
        # 4. GK DISTRIBUTION BYPASS (if GK has ball)
        if role == "goalkeeper":
            # Quick throw/kick to winger or striker
            for t in state.outfield_us():
                r = inp.roles.get(t.id, ROLE_DEFENDER)
                if r in (ROLE_STRIKER, ROLE_WIDE_LEFT, ROLE_WIDE_RIGHT) and t.x > 30.0:
                    opp_pos = [(o.x, o.y) for o in state.outfield_them()]
                    if pass_lane_clear(p.x, p.y, t.x, t.y, opp_pos, margin=0.12):
                        plan = plan_lead_pass(p.x, p.y, t.x, t.y, t.vx, t.vy, velocity_weight=0.2)
                        if plan.target_x > 0:
                            dist = geom.distance(p.x, p.y, plan.target_x, plan.target_y)
                            power = min(1.0, max(0.0, dist / 33.3))
                            return (plan.target_x, plan.target_y, power, "pass", "high_press_gk_bypass")
        
        return None

    def _wall_pass_choice(self, inp: PolicyInput, p: Player) -> tuple[float, float, float] | None:
        state = inp.state
        world = inp.world
        if not is_near_wall(p.x, p.y, margin=6.0):
            return None
        
        role = inp.roles.get(p.id, ROLE_DEFENDER)
        
        # ENHANCED WALL PASS: More aggressive in attacking third
        # 1. Winger near byline: cross via wall to striker
        if role in (ROLE_WIDE_LEFT, ROLE_WIDE_RIGHT) and p.x > 45.0:
            for t in state.outfield_us():
                if inp.roles.get(t.id) == ROLE_STRIKER and t.x > 38.0:
                    # Wall pass to striker - bounce off side wall into box
                    wall_side = "left" if role == ROLE_WIDE_LEFT else "right"
                    wall_x = 0.0 if wall_side == "left" else PITCH_LENGTH
                    # Target point on wall that angles into striker
                    contact_y = geom.clamp(t.y, 8.0, 32.0)
                    # Verify wall pass lane is clear
                    opp_pos = [(o.x, o.y) for o in state.outfield_them()]
                    if pass_lane_clear(p.x, p.y, wall_x, contact_y, opp_pos, margin=0.12):
                        # Check second leg (wall to striker)
                        if pass_lane_clear(wall_x, contact_y, t.x, t.y, opp_pos, margin=0.12):
                            dist = geom.distance(p.x, p.y, wall_x, contact_y)
                            power = min(1.0, max(0.0, dist / 33.3))
                            return (wall_x, contact_y, power)
        
        # 2. Central player near wall: wall pass to advancing winger
        if state.ball.x > 30.0 and role in (ROLE_DEFENDER, ROLE_STRIKER):
            for t in state.outfield_us():
                r = inp.roles.get(t.id, ROLE_DEFENDER)
                if r in (ROLE_WIDE_LEFT, ROLE_WIDE_RIGHT) and t.x > p.x and t.x > 25.0:
                    cand = world.wall.contact_for(p.x, p.y, t.x, t.y)
                    if cand is not None:
                        opp_pos = [(o.x, o.y) for o in state.outfield_them()]
                        if pass_lane_clear(p.x, p.y, cand.contact_x, cand.contact_y, opp_pos, margin=0.1):
                            if pass_lane_clear(cand.contact_x, cand.contact_y, t.x, t.y, opp_pos, margin=0.1):
                                dist = geom.distance(p.x, p.y, cand.contact_x, cand.contact_y)
                                power = min(1.0, max(0.0, dist / 33.3))
                                return (cand.contact_x, cand.contact_y, power)
        
        # 3. Original wall pass logic (enhanced scoring)
        direct = self._receivers(inp, p, count=2)
        best_direct = direct[0][1] if direct else 0.0
        best_wall = None
        best_wall_score = best_direct
        for t in state.outfield_us():
            if t.id == p.id:
                continue
            cand = world.wall.contact_for(p.x, p.y, t.x, t.y)
            if cand is None:
                continue
            # Wall pass value competes against the direct pass value.
            openness = min((geom.distance(t.x, t.y, o.x, o.y) for o in state.outfield_them()), default=go_20())
            threat = self._cfg(inp, "wall_usage", 0.5) * (cand.progression + openness * 2.5)
            risk = cand.risk * (1.0 - self._cfg(inp, "wall_pass_threshold", 0.5) * 0.2)
            score = threat - risk * 30.0
            # Bonus in attacking third
            if p.x > 30.0:
                score += 100.0
            if score > best_wall_score:
                best_wall_score = score
                dist = geom.distance(p.x, p.y, cand.contact_x, cand.contact_y)
                best_wall = (cand.contact_x, cand.contact_y, _power_for_distance(dist))
        if best_wall is not None and best_wall_score > best_direct + 1.0:
            return best_wall
        return None

    def _receivers(self, inp: PolicyInput, p: Player, count: int = 3) -> list[tuple[str, float, float, float]]:
        """Rank teammates as direct-pass targets: (receiver_id, tx, ty, power)."""
        state = inp.state
        ball = state.ball
        vertical = self._cfg(inp, "verticality", 0.55)
        risk_tol = self._cfg(inp, "passing_risk", 0.4)
        width_w = self._cfg(inp, "width", 0.7)
        results: list[tuple[float, str, float, float, float]] = []
        for t in state.outfield_us():
            if t.id == p.id:
                continue
            progress = t.x - p.x
            travel = geom.distance(p.x, p.y, t.x, t.y)
            nearest_opp = min((geom.distance(t.x, t.y, o.x, o.y) for o in state.outfield_them()), default=go_20())
            lane_risk = self._lane_risk(inp, p, t)
            near_box = p.x > 42.0
            if progress < 0.0:
                # Backwards passes are a last resort; inside the box they are
                # never worth it.
                progress *= 0.05 if near_box else 0.3
            if near_box and t.x < p.x - 4.0:
                continue  # never recycle across our own box for a worse line
            # Forward progression premium: reward passes that break the halfway line.
            # The saturated lane_risk * 60 veto is only justified when the
            # corridor is truly occupied; an open forward lane should win over
            # a safe backward recycle. Use a massive premium to overwhelm the
            # saturated lane_risk veto against parked blocks (reference recovers
            # ~5% of passes, so the turnover risk is negligible).
            forward_premium = 0.0
            if progress > 0.0 and t.x >= 30.0:
                forward_premium = 500.0
            
            # CHIPPED THROUGH BALLS: When defense is compact (lane_risk high) 
            # but we have space behind their line, add a chipped option
            chip_bonus = 0.0
            if progress > 10.0 and lane_risk > 0.6 and t.x > 30.0:
                # Chipped pass over defensive line - bypasses lane_risk
                chip_bonus = 600.0
            
            # WIDE PLAY BONUS: Force passes to wingers on the flanks when
            # in attacking half. Low blocks leave wings open.
            wide_bonus = 0.0
            role = inp.roles.get(t.id, ROLE_DEFENDER)
            if role in (ROLE_WIDE_LEFT, ROLE_WIDE_RIGHT) and ball.x > 30.0:
                # Big bonus for passing to wingers in attacking half
                wide_bonus = 400.0
            score = (
                progress * (0.5 + vertical)
                + nearest_opp * (1.0 - risk_tol) * 1.5
                + width_w * abs(t.y - 20.0) * 0.12
                - lane_risk * 60.0
                - travel * 0.06
                + forward_premium
                + chip_bonus
                + wide_bonus
            )
            # Collection geometry: a pass cannot be touched again until it has
            # slowed to 5 m/s, and it always leaves at 12 m/s or more, so it
            # unavoidably rolls MIN_PASS_TRAVEL (~15.6 m). A pass aimed at a
            # receiver standing closer than that sails over their head and
            # straight into the press.
            #
            # The pass is only offered when the ball can actually be collected by
            # someone.
            #
            # The veto is justified by the geometry, not by a win rate. An
            # earlier comment here cited 3-0-9 and 3-11 goals against 2-0-10 and
            # 2-12 for a priced variant; those were 12-match runs, and the
            # 60-match A/B in docs/AB_TESTING.md shows the engine is not
            # deterministic, so that gap was noise and has been removed as
            # evidence. What does reproduce is the behaviour: on a 42 m
            # centre-back hold the pre-fix policy chose `pass` where this one
            # chooses `none`.
            plan = plan_lead_pass(p.x, p.y, t.x, t.y, t.vx, t.vy, velocity_weight=0.35)
            collect_x, collect_y, power, _ = pass_collection_point(
                p.x, p.y, plan.target_x, plan.target_y, 0.0
            )
            overshoot = geom.distance(collect_x, collect_y, plan.target_x, plan.target_y)
            if overshoot > 2.5:
                # Unreachable landing spot: the ball stops well beyond the
                # receiver, so this is a turnover, not a pass.
                continue
            miss = geom.distance(collect_x, collect_y, t.x, t.y)
            if miss > 7.0:
                continue
            score += 40.0 - miss * 3.0
            results.append((score, t.id, collect_x, collect_y, power))
        results.sort(key=lambda r: r[0], reverse=True)
        return [(r[1], r[2], r[3], r[4]) for r in results[:count]]

    def _lane_risk(self, inp: PolicyInput, p: Player, t: Player) -> float:
        """0..1 interception risk along the passing lane."""
        state = inp.state
        if not state.them:
            return 0.0
        mx = (p.x + t.x) / 2.0
        my = (p.y + t.y) / 2.0
        touch_x = (t.x - p.x)
        touch_y = (t.y - p.y)
        length = math.hypot(touch_x, touch_y)
        if length <= 1e-6:
            return 1.0
        risk = 0.0
        for o in state.outfield_them():
            d = geom.seg_point_distance_sq(o.x, o.y, p.x, p.y, t.x, t.y)
            if d < 2.0 * 2.0:
                risk += (2.0 - math.sqrt(d)) / 2.0
        risk = min(1.0, risk)
        # Long passes through the centre are intrinsically riskier.
        return risk + (0.0 if my < 6.0 or my > 34.0 else 0.05)

    def _open_goal_y(self, state: GameState, bx: float, by: float) -> float:
        """Aim for the goal side with the bigger angle and fewer bodies on it."""
        best_y, best_score = GOAL_CENTER_Y, -1e9
        for cand in (GOAL_LOW_Y + 1.2, GOAL_CENTER_Y, GOAL_HIGH_Y - 1.2):
            lane = [o for o in state.outfield_them() if geom.seg_point_distance_sq(o.x, o.y, bx, by, OPP_GOAL_X, cand) < 9.0]
            score = -len(lane) * 10.0 + abs(cand - GOAL_CENTER_Y) * 0.5
            if score > best_score:
                best_score, best_y = score, cand
        return best_y

    def _dribble_target(self, inp: PolicyInput, p: Player) -> tuple[float, float, float]:
        """Where to carry the ball: a reachable step toward the goal.

        The old version aimed at "8 m behind their deepest defender", i.e. up
        to 25 m away. move.target is an absolute destination, so that just
        parked the carrier in a corner of the pitch while the defence walked
        into it. A carry has to be a short, re-aimable step: advance, keep the
        ball on the far side from the nearest marker, and steer around anyone
        standing in the lane.

        Stepping 9 m straight ahead is only correct against a defence that
        happens to be standing somewhere else. Measured against wall-elite,
        which parks a wall across halfway, the carrier walked into it every
        time: our ball never passed x=33.3, we took no shots and lost 0-5 --
        while the wall's own outfielders never once entered y<10. The lane was
        wide open on the left for the entire match and we never looked for it,
        because steering only considered the single nearest marker.

        So probe. Score a fan of headings by how much forward ground they win
        and how many opponents stand in the way, and take the best. Against an
        open defence this still returns "straight at goal", because that is
        what wins the most forward ground with nobody in the lane.
        """
        state = inp.state
        bx, by = p.x, p.y

        step = 9.0

        # In the final third, line up the open side of the goal.
        if bx > 36.0:
            return (
                geom.clamp(bx + 7.0, 1.5, OPP_GOAL_X - 1.5),
                geom.clamp(self._open_goal_y(state, bx, by), 2.0, PITCH_WIDTH - 2.0),
                1.0,
            )

        best_score = -1e9
        best = (bx + step, by)
        for deg in DRIBBLE_PROBE_ANGLES:
            rad = math.radians(deg)
            dx, dy = math.cos(rad), math.sin(rad)
            px_, py_ = bx + dx * step, by + dy * step
            if not (1.5 <= px_ <= OPP_GOAL_X - 1.5) or not (2.0 <= py_ <= PITCH_WIDTH - 2.0):
                continue
            # Forward ground won, less whatever is standing in the lane.
            blocked = 0.0
            for o in state.outfield_them():
                # Distance from the opponent to the probe segment.
                vx, vy = px_ - bx, py_ - by
                seg = vx * vx + vy * vy
                t = 0.0 if seg <= 1e-9 else max(0.0, min(1.0, ((o.x - bx) * vx + (o.y - by) * vy) / seg))
                cx_, cy_ = bx + vx * t, by + vy * t
                d = math.hypot(o.x - cx_, o.y - cy_)
                if d < DRIBBLE_LANE_RADIUS:
                    blocked += (DRIBBLE_LANE_RADIUS - d) / DRIBBLE_LANE_RADIUS
            score = dx * step - blocked * DRIBBLE_BLOCKED_PENALTY
            if score > best_score:
                best_score, best = score, (px_, py_)

        tx, ty = best

        # Keep the ball on the far side from a marker tight on the carrier, and
        # slide wide if one is standing in the immediate lane.
        opp = state.nearest_opponent(bx, by)
        if opp is not None:
            gap = geom.distance(bx, by, opp.x, opp.y)
            ax, ay = bx - opp.x, by - opp.y
            norm = math.hypot(ax, ay)
            if norm > 1e-6:
                if gap < 7.0:
                    tx += ax / norm * 4.0
                    ty += ay / norm * 4.0
                ahead = (opp.x - bx) * step / max(step, 1.0)
                if 0.0 < ahead < 5.0 and abs(opp.y - by) < 3.0:
                    side = 1.0 if by >= opp.y else -1.0
                    ty += side * 7.0

        # Stay off our own goal and inside the pitch.
        tx = geom.clamp(tx, 1.5, OPP_GOAL_X - 1.5)
        ty = geom.clamp(ty, 2.0, PITCH_WIDTH - 2.0)
        return tx, ty, 1.0

    def _decide_off_ball(self, inp: PolicyInput, p: Player, possessor: Player | None) -> PlayerIntent:
        state = inp.state
        world = inp.world
        ball = state.ball
        role = inp.roles.get(p.id, ROLE_DEFENDER)

        if state.has_control() and possessor is not None:
            return self._off_ball_attack(inp, p, possessor, role)

        if ball.possessing_team == "them":
            return self._off_ball_defend(inp, p, role)

        return self._off_ball_recover(inp, p)

    def _pick_striker_channel(self, inp: PolicyInput, ball: Ball, striker_x: float) -> float:
        """Pick the y-channel for the striker with fewest opponent markers ahead.

        When ball is in attacking half (x > 30), target the gaps between 
        center-back and fullback (channels at y≈8-12 and y≈28-32) where
        low blocks leave space. In our half, use standard channel selection.
        """
        state = inp.state
        if ball.x > 30.0:
            # Target gaps between CB and FB: left gap (8-12) or right gap (28-32)
            bands = [(10.0, 8.0, 12.0), (30.0, 28.0, 32.0)]
        else:
            # Standard three bands in our half
            bands = [(10.0, 8.0, 12.0), (20.0, 13.0, 27.0), (30.0, 28.0, 32.0)]
        
        best_y = 20.0
        best_count = 100
        for center_y, y_min, y_max in bands:
            count = 0
            for o in state.outfield_them():
                if o.x > striker_x and y_min <= o.y <= y_max:
                    count += 1
            if count < best_count:
                best_count = count
                best_y = center_y
        return geom.clamp(best_y, 8.0, 32.0)

    def _off_ball_attack(self, inp: PolicyInput, p: Player, possessor: Player, role: str) -> PlayerIntent:
        state = inp.state
        world = inp.world
        ball = state.ball
        depth = self._cfg(inp, "depth", 0.6)
        width = self._cfg(inp, "width", 0.7)
        support = self._cfg(inp, "support_distance", 7.0)

        base = self._role_base(role)
        
        # ===============================================================
        # STRIKER MOVEMENT - Goal-first patterns
        # ===============================================================
        if role == ROLE_STRIKER:
            their_deepest = min((q.x for q in state.outfield_them()), default=60.0)
            their_highest = max((q.x for q in state.outfield_them()), default=0.0)
            defensive_line = their_highest  # Their last defender line
            
            high_line = defensive_line > 35.0
            low_block = defensive_line < 25.0 and their_deepest > 40.0
            is_pressing_team = inp.opp is not None and inp.opp.press_intensity() > 0.7
            
            if ball.x < 25.0:
                # Ball in our half / midfield: Target man role
                # Stay high to stretch defense, provide outlet for GK/defenders
                tx = geom.clamp(45.0 + depth * 8.0, 40.0, 54.0)
                # If they press high, drop slightly to receive and lay off
                if is_pressing_team and ball.x < 20.0:
                    tx = geom.clamp(tx - 8.0, 35.0, 45.0)
            elif ball.x < 35.0:
                # Ball in midfield/attacking third transition
                if high_line:
                    # Run in behind: diagonal run between CB and FB
                    tx = geom.clamp(defensive_line + 3.0, 42.0, 56.0)
                    # Time the run: stay onside until ball is played
                    if possessor and possessor.can_act:
                        # Hold run until passer is ready - delay slightly
                        pass
                elif low_block:
                    # Drop into pockets between midfield and defense (half-spaces)
                    # Find the gap between their midfield and defensive line
                    tx = geom.clamp(ball.x + 8.0, 30.0, 40.0)
                    ty = self._pick_striker_channel(inp, ball, tx)
                else:
                    tx = geom.clamp(ball.x + 12.0 * depth + 6.0, 35.0, 54.0)
            else:
                # Ball in final third: Make dangerous runs
                if low_block:
                    # Against low block: find pockets, drag defenders out
                    winger_wide = False
                    for t in state.outfield_us():
                        r = inp.roles.get(t.id, ROLE_DEFENDER)
                        if r in (ROLE_WIDE_LEFT, ROLE_WIDE_RIGHT) and t.x > 45.0:
                            winger_wide = True
                            break
                    
                    if winger_wide and possessor and possessor.x > 40.0:
                        # Winger has ball wide: attack near/far post
                        if possessor.y < 20.0:  # Left winger
                            ty = 10.0  # Near post
                        else:
                            ty = 30.0  # Near post (right side)
                        tx = geom.clamp(possessor.x + 2.0, 44.0, 54.0)
                    else:
                        # No width: drop deep to create overload in pocket, then spin
                        tx = geom.clamp(ball.x + 3.0, 38.0, 48.0)
                        # Run channel between CB and FB
                        ty = self._pick_striker_channel(inp, ball, tx)
                elif high_line:
                    # High line: run in behind on through ball trigger
                    if possessor and possessor.x > 30.0 and possessor.can_act:
                        # Passer ready to play through ball
                        tx = geom.clamp(defensive_line + 5.0, 45.0, 56.0)
                    else:
                        # Hold position, threaten the run
                        tx = geom.clamp(defensive_line + 1.0, 40.0, 52.0)
                else:
                    # Standard: push up with play
                    tx = geom.clamp(ball.x + 15.0 * depth + 8.0, 42.0, 54.0)
            
            # Channel selection: run between CB and FB (half-space)
            ty = self._pick_striker_channel(inp, ball, tx)
            
            # Fine-tune: if making run behind, bend run to stay onside
            if high_line and tx > defensive_line:
                # Arc run: start wider, cut inside
                if ball.x < possessor.x if possessor else True:
                    pass  # Hold width until pass is played
        
        # ===============================================================
        # WINGER MOVEMENT - Pin, stretch, attack box
        # ===============================================================
        elif role in (ROLE_WIDE_LEFT, ROLE_WIDE_RIGHT):
            lane = 2.0 if role == ROLE_WIDE_LEFT else 38.0
            is_left = role == ROLE_WIDE_LEFT
            
            # Check game state
            their_deepest = min((q.x for q in state.outfield_them()), default=60.0)
            their_highest = max((q.x for q in state.outfield_them()), default=0.0)
            high_line = their_highest > 35.0
            low_block = their_deepest > 40.0 and their_highest < 25.0
            
            # Detect if we have overload on this side
            our_players_side = sum(1 for t in state.outfield_us() 
                                   if (is_left and t.y < 20.0) or (not is_left and t.y > 20.0))
            their_players_side = sum(1 for q in state.outfield_them() 
                                     if (is_left and q.y < 20.0) or (not is_left and q.y > 20.0))
            overload = our_players_side > their_players_side
            
            if ball.x > 30.0:
                # Attacking third: drive to byline, cut inside, or play early
                if low_block:
                    # Against low block: stay WIDE to pin fullback, stretch defense
                    # Drive to byline for cross
                    tx = geom.clamp(ball.x + 15.0 * width, 45.0, 52.0)
                    ty = geom.clamp(lane, 1.0, 6.0) if is_left else geom.clamp(lane, 34.0, 39.0)
                    
                    # If near byline, decide: cross, cut inside, or play back
                    if tx > 48.0:
                        # Check if striker is in box for cross
                        striker_in_box = any(inp.roles.get(t.id) == ROLE_STRIKER and t.x > 40.0 
                                             for t in state.outfield_us())
                        if not striker_in_box:
                            # No target: cut inside for shot/combination
                            ty = geom.clamp(lane + 10.0, 12.0, 28.0)
                            tx = geom.clamp(tx - 3.0, 45.0, 50.0)
                        else:
                            # Striker in box - stay wide for cross
                            ty = geom.clamp(lane, 1.0, 6.0) if is_left else geom.clamp(lane, 34.0, 39.0)
                elif high_line:
                    # High line: run in behind fullback
                    tx = geom.clamp(their_highest + 5.0, 42.0, 54.0)
                    ty = geom.clamp(lane + 3.0, 6.0, 34.0)  # Slightly inside for through ball
                else:
                    # Standard: drive forward with ball
                    tx = geom.clamp(ball.x + 12.0 * width, 45.0, 52.0)
                    ty = geom.clamp(lane + (ball.y - lane) * 0.1, 2.0, 38.0)
                    
                    if tx > 48.0:
                        ty = geom.clamp(lane, 2.0, 38.0)
            else:
                # Midfield/our half: provide width, support build-up
                tx = geom.clamp(ball.x + 5.0 * width, 15.0, 40.0)
                ty = geom.clamp(lane + (20.0 - lane) * 0.2, 4.0, 36.0)
                
                # If ball on opposite side, tuck in for overload
                if (is_left and ball.y > 25.0) or (not is_left and ball.y < 15.0):
                    ty = geom.clamp(ty + (5.0 if is_left else -5.0), 10.0, 30.0)
                
                # Overload support: if we have overload, push higher
                if overload and ball.x > 20.0:
                    tx = geom.clamp(tx + 5.0, 25.0, 40.0)
        
        # ===============================================================
        # DEFENDER (Rest Defence) - Split, cover, intercept
        # ===============================================================
        else:  # defender = rest defence
            # Standard rest defence position
            tx = geom.clamp(ball.x * 0.4 + 6.0, 8.0, 22.0)
            # Split across the centre line instead of both sitting on it
            ty = 15.0 if p.y <= 20.0 else 25.0
            
            # Adaptive rest defence based on opponent
            if inp.opp is not None:
                archetype = inp.opp.detect_archetype()
                if archetype == "counter":
                    # Deeper against counter teams
                    tx = min(tx, 18.0)
                elif archetype == "high_press":
                    # Higher to support build-up
                    tx = max(tx, 12.0)
                elif archetype == "low_block":
                    # Push up to stretch
                    tx = max(tx, 20.0)

        # Support triangle: when the possessor is tightly pressed, offer a short,
        # open, early exit instead of the role anchor.
        #
        # Measured against Vanguard FC (elite): this used to fire for *every*
        # player within reach of the possessor and aimed them all at
        # `possessor.y ± 8.0`, a 16 m band centred on the ball, with the
        # side chosen by comparing player *ids*. Vanguard presses at 0.95, so
        # `pressure_on_ball` was almost always above the 0.55 threshold, and
        # the whole team spent the match inside that band: outfield shape
        # collapsed to 11 m of width and 59% of player-ticks sat within 12 m of
        # the centre. A team that narrow has no passing lane, so the ball could
        # only be cleared sideways and never progressed.
        #
        # The offer now goes to the single closest teammate only, and it is
        # placed on that player's own side of the ball, so taking it can never
        # narrow the shape. Everyone else keeps their role anchor and the width.
        if (
            world.pressure_on_ball > 0.55
            and possessor is not None
            and p.id != possessor.id
            and geom.distance(p.x, p.y, possessor.x, possessor.y) < support + 2.0
        ):
            my_gap = geom.distance(p.x, p.y, possessor.x, possessor.y)
            nearest_gap = min(
                (
                    geom.distance(q.x, q.y, possessor.x, possessor.y)
                    for q in state.outfield_us()
                    if q.id not in (possessor.id, p.id)
                ),
                default=None,
            )
            if nearest_gap is not None and nearest_gap < my_gap:
                return PlayerIntent(p.id, tx, ty, 0.7, ball.x, ball.y)
            # Stay on our own side and keep a real gap, so the escape pass has
            # room and the shape is not pulled into the ball.
            side = 1.0 if p.y >= possessor.y else -1.0
            off_x = geom.clamp(possessor.x - 2.0, 2.0, OPP_GOAL_X - 2.0)
            off_y = geom.clamp(possessor.y + side * 10.0, 3.0, PITCH_WIDTH - 3.0)
            return PlayerIntent(p.id, off_x, off_y, 0.85, ball.x, ball.y)
        return PlayerIntent(p.id, tx, ty, 0.7, ball.x, ball.y)

    def _off_ball_defend(self, inp: PolicyInput, p: Player, role: str) -> PlayerIntent:
        state = inp.state
        world = inp.world
        ball = state.ball
        plan: PressPlan = inp.press_plan
        r = plan.role_for(p.id)

        # PROACTIVE WALL LANE BLOCKING: When opponent near wall, position to block
        # both the pass to wall AND the rebound
        wall_block = self._block_wall_lanes(inp, p, ball)
        if wall_block is not None:
            return wall_block

        # WALL PASS INTERCEPT: React to wall passes in progress
        if ball.possessing_team == "them" and state.ball.vx != 0:
            wall_intercept = self._intercept_wall_pass(inp, p, ball)
            if wall_intercept is not None:
                return wall_intercept

        # POSSESSION PRESSING: Against possession teams, intercept passing lanes
        # rather than chasing the ball carrier
        possession_intercept = self._intercept_possession_pass(inp, p, ball)
        if possession_intercept is not None:
            return possession_intercept

        # ELITE DEFENDING: Man-to-man assignments with zonal cover
        # Build assignment map: each defender gets a specific attacker to track
        assignments = self._build_defensive_assignments(inp, state)
        my_assignment = assignments.get(p.id)
        
        # Man-mark their most advanced forward on deep transitions: two deepest
        # teammates split the deepest two attackers goal-side.
        danger = _most_advanced(state, "them")
        if danger is not None and danger.x > 28.0:
            deepest = sorted(
                state.outfield_us(),
                key=lambda q: (q.x, q.id),
            )[:2]
            if p in deepest:
                mark_y = danger.y
                if inp.opp is not None:
                    mark_y = geom.clamp(mark_y + inp.opp.prefer_side() * 3.0, 3.0, PITCH_WIDTH - 3.0)
                tx = geom.clamp((danger.x + 5.0) * 0.5, 6.0, ball.x)
                ty = geom.clamp(mark_y, 3.0, PITCH_WIDTH - 3.0)
                return PlayerIntent(p.id, tx, ty, 1.0, ball.x, ball.y, "none")

        if r == "PRESS":
            tx, ty = ball.x, ball.y
            speed = 1.0
            # Close the space, stay goal-side.
            return PlayerIntent(p.id, tx, ty, speed, ball.x, ball.y, "none")
        if r == "PRESS_SUPPORT":
            # Two-point support: between the presser and the ball's near escape.
            tx, ty = self._support_press_point(inp, p)
            return PlayerIntent(p.id, tx, ty, 0.9, ball.x, ball.y)
        if r == "COVER":
            tx, ty = self._cover_point(inp, p)
            return PlayerIntent(p.id, tx, ty, 0.85, ball.x, ball.y)
        
        # ASSIGNED MAN-TO-MAN: If we have an assignment, track them goal-side
        if my_assignment is not None:
            return self._track_assignment(inp, p, my_assignment, ball, state)
        
        # Rest defence: anchored, goal-side, ready for the counter. The anchor
        # shades toward the side they prefer to attack (§38) and deepens
        # against a swarm-pressing side with a high defensive line.
        #
        # The deep line was tried and rejected on evidence. Holding a high line
        # between the ball and our goal (x up to 40) instead of ceding the
        # middle third looked right on paper -- our outfielders had been
        # averaging x=24..31 -- but it measured much worse against Vanguard FC
        # (elite): 12 matches, 1-0-11, 1-13 goals, versus 3-0-9 and 3-11 for
        # the deep line. Their directness 0.75 and tempo 1.0 turn a high line
        # into a ball over the top, and with four outfielders there is nobody
        # behind it. The empty middle third is cheaper than a exposed channel.
        opp = inp.opp
        base_line = self._cfg(inp, "defensive_line", 26.0)
        defensive_line = base_line
        
        # ADAPTIVE REST DEFENSE: Deeper against counter/direct/physical teams
        if opp is not None:
            archetype = opp.detect_archetype()
            if archetype in ("counter", "direct", "physical"):
                # Much deeper against fast transitions
                defensive_line = min(base_line - 4.0, 18.0)
            elif archetype == "wall":
                # Deeper to cover wall rebounds
                defensive_line = min(base_line - 2.0, 20.0)
            elif opp.press_intensity() > 0.7:
                defensive_line -= 2.0
            if opp.defensive_shape_height() > 34.0:
                defensive_line = min(base_line + 3.0, 30.0)
        
        # Always keep at least 2 defenders goal-side of the ball
        # If ball is in attacking third, defenders stay deeper
        if ball.x > 40.0:
            defensive_line = min(defensive_line, 18.0)
        elif ball.x > 30.0:
            defensive_line = min(defensive_line, 22.0)
        
        tx = defensive_line
        anchor_y = 20.0
        if opp is not None and opp.prefer_side() != 0.0:
            anchor_y = geom.clamp(20.0 + opp.prefer_side() * 8.0, 6.0, 34.0)
        elif state.goalkeeper_us() is not None and ball.y < 20.0:
            anchor_y = 24.0 if role == ROLE_WIDE_RIGHT else 16.0
        
        # COUNTER-PRESS: If we just won the ball, press HIGH immediately
        if inp.counter_press_active:
            # Sprint to the ball carrier's position to disrupt their build-up
            tx = geom.clamp(ball.x - 5.0, 15.0, 40.0)
            anchor_y = ball.y
            speed = 1.0
        else:
            # Compact defensive block - keep defensive line deeper against low blocks
            tx = geom.clamp(tx, 6.0, 26.0)
            speed = 0.7
        return PlayerIntent(p.id, tx, anchor_y, speed, ball.x, ball.y)

    def _block_wall_lanes(self, inp: PolicyInput, p: Player, ball: Ball) -> PlayerIntent | None:
        """Proactively block wall pass lanes when opponent is near a wall.
        
        When opponent possessor is near a touchline, position to cut off
        both the pass to the wall and the potential rebound.
        """
        state = inp.state
        if ball.possessing_team != "them":
            return None
        
        their_possessor = state.their_possessor()
        if their_possessor is None:
            return None
        
        from .wall import is_near_wall
        # Check if opponent is near either wall
        near_left = their_possessor.y < 8.0
        near_right = their_possessor.y > 32.0
        
        if not (near_left or near_right):
            return None
        
        # Only defenders/wingers on the same flank should block
        defender_side = "left" if p.y < 20.0 else "right"
        if near_left and defender_side != "left":
            return None
        if near_right and defender_side != "right":
            return None
        
        # Position between ball and wall, and goal-side of their support
        wall_x = 0.0 if near_left else PITCH_LENGTH
        wall_buffer = 3.0  # Distance from wall to intercept
        
        # Block the pass to the wall
        tx = geom.clamp(wall_x + (wall_buffer if near_left else -wall_buffer), 4.0, 56.0)
        # Stay goal-side of their potential receiver
        ty = geom.clamp(their_possessor.y, 5.0, 35.0)
        
        # If they have a support player wide, mark them
        for opp in state.outfield_them():
            if opp.id == their_possessor.id:
                continue
            if near_left and opp.y < 12.0 and opp.x > their_possessor.x:
                ty = geom.clamp(opp.y, 3.0, 15.0)
                break
            if near_right and opp.y > 28.0 and opp.x > their_possessor.x:
                ty = geom.clamp(opp.y, 25.0, 37.0)
                break
        
        return PlayerIntent(p.id, tx, ty, 0.9, ball.x, ball.y, "none")

    def _intercept_possession_pass(self, inp: PolicyInput, p: Player, ball: Ball) -> PlayerIntent | None:
        """Intercept passes from possession teams by reading passing lanes.
        
        Instead of chasing the ball carrier, position to cut off their
        most likely passing options.
        """
        state = inp.state
        if ball.possessing_team != "them":
            return None
        
        # Only activate against possession-style opponents
        if inp.opp is None:
            return None
        archetype = inp.opp.detect_archetype()
        if archetype not in ("possession", "wall", "tika"):
            return None
        
        their_possessor = state.their_possessor()
        if their_possessor is None:
            return None
        
        # Find their most likely passing targets
        their_players = state.outfield_them()
        targets = [opp for opp in their_players if opp.id != their_possessor.id]
        if not targets:
            return None
        
        # Sort by proximity to possessor and forward position
        targets.sort(key=lambda opp: (geom.distance(opp.x, opp.y, their_possessor.x, their_possessor.y), -opp.x))
        
        # For each target, check if we can intercept the lane
        for target in targets[:2]:  # Check top 2 targets
            lane_dist = geom.seg_point_distance_sq(
                p.x, p.y, 
                their_possessor.x, their_possessor.y, 
                target.x, target.y
            )
            # If we're close to the passing lane, position to intercept
            if lane_dist < 4.0:  # Within 2m of lane
                # Position on the lane, goal-side of target
                tx = geom.clamp(target.x - 3.0, 4.0, their_possessor.x)
                ty = geom.clamp(target.y, 3.0, 37.0)
                return PlayerIntent(p.id, tx, ty, 0.85, ball.x, ball.y, "none")
        
        return None

    def _intercept_wall_pass(self, inp: PolicyInput, p: Player, ball: Ball) -> PlayerIntent | None:
        """Position to intercept opponent wall passes using physics prediction.
        
        When opponent is near a wall with the ball, they may play a wall pass.
        We predict the bounce trajectory and position defenders to intercept.
        """
        state = inp.state
        if ball.possessing_team != "them":
            return None
        
        their_possessor = state.their_possessor()
        if their_possessor is None:
            return None
        
        # Check if opponent possessor is near a wall
        from .wall import is_near_wall
        if not is_near_wall(their_possessor.x, their_possessor.y, margin=5.0):
            return None
        
        # Predict wall bounce for both walls
        opp_poss = their_possessor
        their_players = state.outfield_them()
        
        for wall_side in ("left", "right"):
            wall_x = 0.0 if wall_side == "left" else PITCH_LENGTH
            # Check if possessor is near this wall
            dist_to_wall = abs(opp_poss.y - (0.0 if wall_side == "left" else PITCH_WIDTH))
            if dist_to_wall > 8.0:
                continue
            
            # For each potential receiver, predict the wall pass trajectory
            for opp in their_players:
                if opp.id == opp_poss.id:
                    continue
                # Receiver should be on same side and ahead
                if wall_side == "left" and not (opp.y < 22.0 and opp.x > opp_poss.x):
                    continue
                if wall_side == "right" and not (opp.y > 18.0 and opp.x > opp_poss.x):
                    continue
                
                # Predict the wall pass using physics
                # Wall pass: ball goes from possessor -> wall -> receiver
                # We need to find the contact point on the wall
                # Use mirror method: mirror receiver across wall, line from possessor to mirror hits wall at contact point
                if wall_side == "left":
                    mirrored_rx = -opp.x
                else:
                    mirrored_rx = 2 * PITCH_LENGTH - opp.x
                
                dx = mirrored_rx - opp_poss.x
                dy = opp.y - opp_poss.y
                if abs(dx) < 1e-6:
                    continue
                t = (wall_x - opp_poss.x) / dx
                if t <= 0 or t >= 1:
                    continue
                contact_y = opp_poss.y + dy * t
                contact_y = geom.clamp(contact_y, 2.0, PITCH_WIDTH - 2.0)
                
                # Now we have the contact point (wall_x, contact_y)
                # The ball will rebound toward the receiver
                # We should position a defender on the rebound path
                # Intercept point: between wall and receiver
                intercept_x = geom.clamp(wall_x + (4.0 if wall_side == "left" else -4.0), 3.0, 57.0)
                intercept_y = geom.clamp(contact_y + (opp.y - contact_y) * 0.3, 3.0, PITCH_WIDTH - 3.0)
                
                # Only intercept if we're the right defender for this flank
                defender_side = "left" if p.y < 20.0 else "right"
                if defender_side == wall_side:
                    return PlayerIntent(p.id, intercept_x, intercept_y, 1.0, ball.x, ball.y, "none")
        
        return None

    def _support_press_point(self, inp: PolicyInput, p: Player) -> tuple[float, float]:
        state = inp.state
        ball = state.ball
        pressers = inp.press_plan.pressers + inp.press_plan.second_pressers
        if pressers:
            leader = next((q for q in state.outfield_us() if q.id == pressers[0]), None)
            if leader is not None:
                mx = (leader.x + ball.x) / 2.0
                my = (leader.y + ball.y) / 2.0
                return geom.clamp_point(mx, my)
        return (ball.x, ball.y)

    def _cover_point(self, inp: PolicyInput, p: Player) -> tuple[float, float]:
        """Stand goal-side of the ball on the line to our goal, blocking the
        most dangerous pass/shot lane."""
        state = inp.state
        ball = state.ball
        gx, gy = 0.0, GOAL_CENTER_Y
        # Website-ish: interpolate between ball and our goal.
        tx = geom.lerp(ball.x, gx, 0.45)
        ty = geom.lerp(ball.y, gy, 0.45)
        tx = max(4.0, tx)
        cx = geom.clamp(tx, 4.0, ball.x)
        cy = geom.clamp(ty, 3.0, PITCH_WIDTH - 3.0)
        # Slight lane offset by our y-position to split two attackers.
        return (cx, cy)

    def _off_ball_recover(self, inp: PolicyInput, p: Player) -> PlayerIntent:
        """Loose ball: go where it will finally be slow enough to control.

        Sprinting at a kicked ball's current position is pointless: it is
        travelling at 12-26 m/s and only becomes collectable below 5 m/s after
        rolling for about a second and a half. The first runner therefore aims
        at that meeting point, the second joins if it is reachable for them
        too, and everyone else holds the goal-side shape so winning the ball
        does not expose us to a counter.
        """
        state = inp.state
        ball = state.ball
        mx, my, seconds = loose_ball_meeting_point(ball.x, ball.y, ball.vx, ball.vy)
        mx = geom.clamp(mx, 0.5, PITCH_LENGTH - 0.5)
        my = geom.clamp(my, 0.5, PITCH_WIDTH - 0.5)

        by_dist = sorted(
            state.outfield_us(),
            key=lambda q: (geom.distance(q.x, q.y, mx, my), q.id),
        )
        if not by_dist:
            return PlayerIntent(p.id, ball.x, ball.y, 1.0, ball.x, ball.y)

        for rank, q in enumerate(by_dist[:2]):
            if p.id != q.id:
                continue
            reach = MAX_RUN_SPEED * seconds
            dist = geom.distance(q.x, q.y, mx, my)
            if rank == 1 and dist > reach + 2.0:
                # Cannot get there in time: stay goal-side instead of running
                # past the ball and leaving the middle open.
                tx, ty = self._cover_point(inp, p)
                return PlayerIntent(p.id, tx, ty, 0.9, ball.x, ball.y)
            return PlayerIntent(p.id, mx, my, 1.0, ball.x, ball.y)

        tx, ty = self._cover_point(inp, p)
        return PlayerIntent(p.id, tx, ty, 0.85, ball.x, ball.y)

    # ------------------------------------------------------------------ #
    # Press actions (tackle / slap) applied to pressers near the ball.
    # ------------------------------------------------------------------ #
    def _enforce_press_actions(self, inp: PolicyInput, intents: dict[str, PlayerIntent]) -> None:
        state = inp.state
        if state.ball.possessing_team != "them":
            return
        possessor = state.their_possessor()
        if possessor is None or not possessor.can_act:
            return
        if inp.their_possession_protected():
            return
        ball = state.ball
        gk_them = state.goalkeeper_them()
        for pid, intent in intents.items():
            player = next((q for q in state.us if q.id == pid), None)
            if player is None or not player.can_act:
                continue
            if player.role == "goalkeeper":
                continue
            d = geom.distance(player.x, player.y, ball.x, ball.y)
            if d > TACKLE_MAX:
                continue
            # Never try to yank the ball from their goalkeeper.
            if gk_them is not None and possessor.id == gk_them.id:
                continue
            facing_ok = geom.facing_diff(player.facing, ball.x, ball.y, player.x, player.y) <= math.radians(SLAP_FACING_DEG)
            if d <= SLAP_CLEAN and facing_ok and config_uses_slap(inp):
                intent.action_type = "slap"
            elif d < TACKLE_CLOSE:
                intent.action_type = "tackle"
            elif d <= TACKLE_MAX:
                # Slide only when the lane to our goal is already covered.
                if _box_covered(inp):
                    intent.action_type = "tackle"
            if intent.action_type in ("tackle", "slap"):
                intent.action_target = None
                intent.action_power = None

    def _resolve_ball_carrier_conflicts(self, inp: PolicyInput, intents: dict[str, PlayerIntent]) -> None:
        state = inp.state
        if not state.has_control():
            return
        possessor_id = state.ball.possessing_player
        for intent in intents.values():
            if intent.pid == possessor_id:
                continue
            if intent.action_type in ("pass", "shoot", "clear"):
                intent.action_type = "none"
                intent.action_target = None
                intent.action_power = None

    # ------------------------------------------------------------------ #
    # Kickoff / restart
    # ------------------------------------------------------------------ #
    def _kickoff_intents(self, inp: PolicyInput) -> dict[str, PlayerIntent]:
        """f1 is handed the ball on the centre spot, so we start by carrying it.

        Passing here releases the ball at >=12 m/s into a team that is still
        walking out of its own half, which is how a kickoff turns into a 15 m
        loose ball that nobody is able to collect. Drive forward instead and
        push the rest of the shape ahead of the ball so even a turnover stays
        dangerous.
        """
        state = inp.state
        intents: dict[str, PlayerIntent] = {}
        ball = state.ball
        starter = state.our_possessor()
        for p in state.outfield_us():
            role = inp.roles.get(p.id, ROLE_DEFENDER)
            if starter is not None and p.id == starter.id:
                tx, ty, speed = self._dribble_target(inp, p)
                intents[p.id] = PlayerIntent(p.id, tx, ty, speed, tx, ty)
                continue
            if role == ROLE_STRIKER:
                # Stay high: target man and rebound option.
                tx = geom.clamp(ball.x + 16.0, 38.0, 52.0)
                ty = self._pick_striker_channel(inp, ball, 44.0)
            elif role in (ROLE_WIDE_LEFT, ROLE_WIDE_RIGHT):
                lane = 6.0 if role == ROLE_WIDE_LEFT else 34.0
                tx, ty = geom.clamp(ball.x + 11.0, 34.0, 48.0), lane
            else:
                # Rest defence: never both centre-backs walk past halfway.
                tx, ty = geom.clamp(ball.x * 0.5 + 4.0, 6.0, 20.0), GOAL_CENTER_Y
            intents[p.id] = PlayerIntent(p.id, tx, ty, 0.85, ball.x, ball.y)
        gk = state.goalkeeper_us()
        if gk is not None:
            intents[gk.id] = PlayerIntent(gk.id, 3.0, GOAL_CENTER_Y, 0.5, ball.x, ball.y)
        return intents

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _build_defensive_assignments(self, inp: PolicyInput, state: GameState) -> dict[str, Player]:
        """Assign each defender a specific attacker to track (man-to-man with zonal fallback).
        
        Returns a dict mapping defender_id -> assigned attacker Player.
        """
        assignments: dict[str, Player] = {}
        our_defenders = [p for p in state.outfield_us() if inp.roles.get(p.id) == ROLE_DEFENDER]
        their_attackers = [q for q in state.outfield_them() if q.x > 25.0]  # Attackers in our half
        
        if not our_defenders or not their_attackers:
            return assignments
        
        # Sort by x-position: deepest defender gets deepest attacker
        our_defenders_sorted = sorted(our_defenders, key=lambda q: q.x)
        their_attackers_sorted = sorted(their_attackers, key=lambda q: q.x, reverse=True)
        
        # Assign each defender to an attacker (1-to-1 where possible)
        for i, defender in enumerate(our_defenders_sorted):
            if i < len(their_attackers_sorted):
                assignments[defender.id] = their_attackers_sorted[i]
            else:
                # Extra defenders: assign to space / zonal cover
                pass
        
        return assignments

    def _track_assignment(self, inp: PolicyInput, defender: Player, attacker: Player, 
                          ball: Ball, state: GameState) -> PlayerIntent:
        """Track assigned attacker goal-side, intercept through balls, maintain compactness."""
        
        # Goal-side positioning: between attacker and our goal
        # Ideal position: slightly closer to goal than attacker, on the line to goal
        goal_x, goal_y = 0.0, GOAL_CENTER_Y
        
        # Distance to maintain: 3-5m goal-side of attacker
        track_distance = 4.0
        dx = attacker.x - goal_x
        dy = attacker.y - goal_y
        dist_to_goal = math.hypot(dx, dy)
        
        if dist_to_goal > 1e-3:
            # Position goal-side of attacker
            tx = attacker.x - (dx / dist_to_goal) * track_distance
            ty = attacker.y - (dy / dist_to_goal) * track_distance
        else:
            tx, ty = attacker.x - track_distance, attacker.y
        
        # Clamp to reasonable defensive positions
        tx = geom.clamp(tx, 4.0, ball.x if ball.x > 10.0 else 20.0)
        ty = geom.clamp(ty, 3.0, PITCH_WIDTH - 3.0)
        
        # INTERCEPTION LOGIC: If through ball is played toward attacker,
        # step into the passing lane
        if ball.possessing_team == "them" and ball.vx < -1.0:  # Ball moving toward our goal
            # Check if ball trajectory intersects our assignment zone
            ball_to_attacker = geom.distance(ball.x, ball.y, attacker.x, attacker.y)
            if ball_to_attacker < 15.0:  # Ball played toward our man
                # Move to intercept: position on ball-attacker line, goal-side
                if ball.x > attacker.x:
                    # Through ball behind us: drop deeper
                    tx = min(tx, attacker.x - 2.0)
        
        # COMPACTNESS: Don't get pulled too wide - maintain defensive structure
        our_center_y = sum(q.y for q in state.outfield_us()) / max(1, len(state.outfield_us()))
        max_width_from_center = 15.0
        if abs(ty - our_center_y) > max_width_from_center:
            # Drift back toward center to maintain compactness
            if ty > our_center_y:
                ty = our_center_y + max_width_from_center
            else:
                ty = our_center_y - max_width_from_center
        
        speed = 0.85
        # If attacker is making a run, match their speed
        if attacker.vx > 2.0 or attacker.vy > 2.0:
            speed = 1.0
        
        return PlayerIntent(defender.id, tx, ty, speed, ball.x, ball.y)

    def _role_base(self, role: str) -> tuple[float, float]:
        bases = {
            ROLE_DEFENDER: (14.0, 20.0),
            ROLE_WIDE_LEFT: (24.0, 8.0),
            ROLE_WIDE_RIGHT: (24.0, 32.0),
            ROLE_STRIKER: (40.0, 20.0),
        }
        return bases.get(role, (20.0, 20.0))

    def _safe_pass_plan(self, inp: PolicyInput, src: Player, t: Player) -> tuple[float, float, float]:
        """Conservative pass sizing for the goalkeeper."""
        plan = plan_lead_pass(src.x, src.y, t.x, t.y, t.vx, t.vy, velocity_weight=0.25)
        dist = geom.distance(src.x, src.y, t.x, t.y)
        power = _power_for_distance(dist)
        return plan.target_x, plan.target_y, power



def _power_for_distance(dist: float) -> float:
    """Convert distance to kick power (0.0 to 1.0)."""
    # Simplified: power scales with distance, capped at 1.0
    # Rough calibration: 10m = 0.3, 20m = 0.6, 30m = 0.9
    return min(1.0, max(0.0, dist / 33.3))


def config_uses_slap(inp: PolicyInput) -> bool:
    # Slapping is best when we need to strip a carried ball in a dead play.
    return getattr(inp.config, "risk", 0.45) > 0.3 or inp.world.pressure_on_ball > 0.6


def _box_covered(inp: PolicyInput) -> bool:
    # At least one covered player goal-side of the ball before sliding.
    return len(inp.press_plan.cover) >= 1

