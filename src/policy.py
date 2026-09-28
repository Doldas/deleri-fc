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
    BALL_DECAY,
    KICK_MAX_SPEED,
    KICK_MIN_SPEED,
    MAX_RUN_SPEED,
    SLAP_CLEAN,
    SLAP_FACING_DEG,
    TACKLE_CLOSE,
    TACKLE_MAX,
    RuntimeConfig,
    shooting_distance,
)
from .geom import (
    GOAL_CENTER_Y,
    GOAL_HALF,
    GOAL_HIGH_Y,
    GOAL_LOW_Y,
    OPP_GOAL_X,
    PITCH_LENGTH,
    PITCH_WIDTH,
)


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


def _shot_geometry_target(inp: PolicyInput, p: Player) -> tuple[tuple[float, float], float]:
    """The shot we would actually take from here, as (target_xy, power).

    The aim point is the corner that minimises the goalkeeper's lateral reach
    after he has shifted across during the ball's flight, so the returned
    target is already the best available one and every probability computed
    from it is the real shot we are contemplating.
    """
    gk = inp.state.goalkeeper_them()
    gkx = gk.x if gk is not None else -1.0
    gky = gk.y if gk is not None else GOAL_CENTER_Y
    st = pick_shot_target(p.x, p.y, gkx, gky)
    return (st.x, st.y), st.power


def _shot_ev(inp: PolicyInput, p: Player) -> float:
    """Expected value of the best available shot, from the real geometry.

    This is `ExpectedValueCalculator.ev_shot` on the shot we would actually
    take. It is deliberately *not* a nominal-position test: the value depends
    only on where the ball is, where the keeper is, where the defenders are
    and how well the shot is aimed.
    """
    (tx, ty), power = _shot_geometry_target(inp, p)
    return ExpectedValueCalculator(inp).ev_shot(p, (tx, ty), power)


def _shot_p_goal(inp: PolicyInput, p: Player) -> float:
    """P(goal) of the best available shot -- the decomposed, geometric xG."""
    (tx, ty), power = _shot_geometry_target(inp, p)
    return ExpectedValueCalculator(inp).p_shot_goal(p, (tx, ty), power)


def _shot_worth_taking(inp: PolicyInput, p: Player) -> bool:
    """Should this player shoot right now?

    Two bars, both geometric, and neither of them looks at nominal position:

    * Inside the box the test is on *reaching the goal at all*: is the effort
      on frame and not into a body? At that range an attempt is worth taking
      even against a keeper who is set, because it is the closest the ball
      will ever get and holding it there invites a tackle. Note this is
      deliberately not `P(goal)` -- against a set keeper from the middle the
      goal probability is ~1%, and refusing to shoot there is what produced
      the original 35-shots-0-goals season.
    * Outside the box the bar is a full expected value, and it is high enough
      that the shot has to be a real chance. This is the lever that matters
      against a strong keeper: from 18 m up the centre it is a standing catch
      every time, because he has 0.7 s of flight to cover the far post. The
      only way to make it a goal is to drag him wide first and then hit the far
      corner, and `ev_shot` rewards exactly that and nothing else.

    The old version of this function branched on `role`, so a centre-back who
    had carried the ball upfield was judged on his shirt rather than on the
    geometry, and a striker in the same position got a different answer for
    the identical picture. Both are now the same calculation.
    """
    ev = ExpectedValueCalculator(inp)
    (tx, ty), power = _shot_geometry_target(inp, p)
    dist_goal = OPP_GOAL_X - p.x
    if dist_goal <= SHOT_IN_BOX_DIST:
        reaches = (1.0 - ev.p_shot_blocked(p, (tx, ty), power)) * ev.p_shot_on_target(
            p, (tx, ty), power
        )
        return reaches >= EV_SHOT_REACHES_GOAL_BAR
    return ev.ev_shot(p, (tx, ty), power) > EV_SHOT_WORTH



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

# --- Attacking Action Values -------------------------------------------------
# Floor on a natural striker's shot from inside the box, in the same arbitrary
# units as the candidate list below.
#
# It exists because of ordering, not aggression. The recycling candidates are
# worth switch_play 20.0 and safe_pass 10.0, so a floor below 20.0 means the
# striker passes sideways/backwards from inside the area while the opponents'
# keeper walks up to collect. A genuine chance scores base_value 0.8 -> 80.0,
# so anything under that still loses to a real opening.
#
# This floor previously sat at 10.0 and lost to switch_play. That is what made a
# bogus "rebound setup" candidate load-bearing: it was worth 45.0, so it won the
# striker's decision by brute force, and it aimed at the keeper's body, which is
# the one shot a keeper is guaranteed to hold. See
# test_no_shot_is_aimed_at_the_keeper_body.
IN_BOX_SHOT_FLOOR = 24.0

# Value of a carry that wins real forward ground down a clean lane. It sits
# above safe_pass (10.0) and below switch_play (20.0) on purpose: given the
# loose-ball physics above -- a pass leaves the ball at KICK_MIN_SPEED and has
# to decelerate below the control limit before anybody may touch it, while a
# dribble keeps it glued 0.65 m in front -- a clean carry is the better way to
# move the ball up the pitch, but it should not displace the deliberate
# pattern plays (cross 25, pullback 28, wall pass 30, cutback 35, through 40).
CARRY_CLEAN_VALUE = 15.0
# Value of a carry that goes nowhere forward. Still legal and still better than
# turning the ball over, so it stays positive and remains the last resort.
CARRY_HOLD_VALUE = 5.0
# Forward ground that scores full marks, matching the long step _dribble_target
# takes, so a carry that is cut short by the touchline is worth proportionally
# less.
CARRY_FULL_GAIN = 8.0
# Ending a carry inside this range of the opponent goal is worth a bonus: the
# next decision is a shot rather than another pass.
CARRY_SHOOT_RANGE = 20.0
CARRY_SHOOT_BONUS = 4.0

# --- Expected Value Constants ---------------------------------------------
# Units are "goal-equivalents": EV_GOAL_VALUE = 100 means one goal is worth
# 100, and everything else is priced as a fraction of a goal. That forces
# passes, carries and shots onto one honest scale, which is the whole point:
# when a pass was scored at 20-30 and a real shot at -2, no amount of tuning
# the shot model could ever make the team shoot.
EV_GOAL_VALUE = 100.0

# What a shot that does not score is actually worth, per the engine rules
# (docs/00-game-engine-rules.md). These are *not* turnovers:
#
#  * Goal kicks do not exist in this ruleset. A shot that misses hits the
#    goal-line wall and rebounds off it at 75%, so the ball is still loose and
#    still ours to chase. A miss costs tempo, not possession.
#  * A shot needing 0.8-1.65 m of reach is a dive, and the keeper stays
#    grounded holding the ball for 0.9 s. That is a forced distribution
#    window we get for free by shooting.
#  * A shot needing under 0.8 m is a standing catch, so the keeper plays on
#    normally and we have genuinely lost the ball.
EV_SHOT_ON_TARGET_SAVED = -4.0
EV_SHOT_BLOCKED = -2.0
EV_SHOT_WIDE = 1.0

# The follow-up we are entitled to after forcing a dive: the keeper is
# grounded for GK_GROUNDED_SECONDS and cannot distribute, and a miss rebounds
# off the wall. This is the reason shooting from inside the box is worth doing
# even when P(goal) is low, and it is why the shot EV beats a sideways pass
# at close range instead of losing to it.
GK_GROUNDED_SECONDS = 0.9
EV_SHOT_SECOND_BALL = 6.0

# --- Pass / carry outcomes -------------------------------------------------
# Retaining the ball is worth a small amount; what makes a pass attractive is
# where it puts the ball, and how much of a goal that position is worth. A
# pass into the box is close to a chance; a pass sideways is not.
EV_PASS_COMPLETED = 2.0
# Ground won and ground lost are not symmetric: a pass into the final third is
# worth having, but a pass that walks the ball 20 m back out of their box hands
# them the initiative again and costs more than the same metres gained. Before
# this was signed, a retreat out of the box scored the same flat 2.0 as a
# square pass, so the team recycled possession forever and the shot could
# never win the ranking -- which is most of why the season produced 35 shots
# and no goals, or one shot a match, depending on which patch was live.
EV_PASS_PROGRESS_PER_M = 0.35
EV_PASS_REGRESSION_PER_M = 0.60
# Where the ball ends up matters as much as how far it travelled. These are
# the strongest levers the attack has for getting the ball into the box, which
# is the only place a shot is worth taking against a keeper who covers both
# posts from the centre.
EV_PASS_FINAL_THIRD_BONUS = 5.0
EV_PASS_BOX_BONUS = 10.0
# How far from a set goalkeeper a dribble still counts as contested. Inside his
# reach it is a certain giveaway; out here it is an ordinary carry.
CARRY_KEEPER_RADIUS = 5.0
EV_PASS_INTERCEPTED = -15.0
EV_PASS_OUT_OF_PLAY = 0.0
EV_CARRY_PROGRESS = 8.0
EV_CARRY_LOSS = -20.0
EV_CARRY_HOLD = 1.5

# Losing the ball is priced by where it happens: deep in their half the
# transition is dangerous, in our own half it is merely a reset.
EV_GOAL_CONCEDED = -100.0      # Conceding a goal
EV_LOSS_OF_POSSESSION = -20.0   # Losing possession in dangerous area
EV_LOSS_OF_POSSESSION_SAFE = -5.0  # Losing possession in safe area

# Action-specific bonuses (multiply base outcome)
EV_THROUGH_BALL_MULTIPLIER = 1.5  # Through balls create high-quality chances
EV_CUTBACK_MULTIPLIER = 1.8       # Cutbacks create high-quality shots
EV_CROSS_MULTIPLIER = 1.3         # Crosses create aerial chances
EV_CUTBACK_SHOT_BONUS = 1.5       # Cutback shots are higher quality
EV_CROSS_HEADER_BONUS = 1.2       # Headers from crosses
EV_WALL_PASS_MULTIPLIER = 1.2     # Wall passes bypass defenders
EV_SWITCH_MULTIPLIER = 1.1        # Switches open weak side

# Risk factors
EV_COUNTER_ATTACK_RISK = 0.3      # Probability counter-attack leads to goal conceded
EV_TACKLE_SUCCESS_RATE = 0.6      # Base tackle success rate

# Base values for fallback
EV_BASE_ACTION_VALUE = 0.0

# --- Shot-model geometry (all derived from engine constants) ----------------
# A defender standing exactly in the lane is a certain-ish block; he only has
# to cover the offset *beyond* his own body radius, so BODY_RADIUS is the free
# part and REACH is the point where he cannot get there at all.
SHOT_BLOCK_BODY_RADIUS = 0.5
SHOT_BLOCK_REACH = 2.5
# Share of the shot a genuine blocker removes, from arriving with the ball to
# arriving with time to set himself. Multiple blockers compound.
SHOT_BLOCK_SHARE_SETTLED = 0.88
SHOT_BLOCK_SHARE_LATE = 0.35
# Seconds of slack a defender needs before a block counts as "settled".
SHOT_BLOCK_T_SETTLED = 0.20
# Keeper outcomes on a shot that is on frame: standing catch vs clean goal,
# and a keeper dragged outside his own defensive fifth.
SHOT_GOAL_STANDING_CATCH = 0.02
SHOT_GOAL_CLEAN = 0.95
SHOT_GOAL_GK_OUT = 0.97
# Lateral aiming error of a struck ball: a fixed strike/impact floor plus a
# per-metre component, so the same aim point is far harder to hold from 25 m
# than from 8 m. Used as the standard deviation of the landing distribution
# against the 6 m frame (see ExpectedValueCalculator.p_shot_wide).
SHOT_ERROR_BASE = 0.55
SHOT_ERROR_PER_M = 0.10
# Even a hopelessly wide effort occasionally finds the frame.
SHOT_ON_TARGET_FLOOR = 0.03


def _normal_cdf(z: float) -> float:
    """Standard normal CDF (Abramowitz & Stegun 26.2.17, |err| < 7.5e-8)."""
    t = 1.0 / (1.0 + 0.2316419 * abs(z))
    poly = t * (
        0.319381530
        + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429)))
    )
    tail = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    return 1.0 - tail * poly if z > 0 else tail * poly

# --- Pass race model -------------------------------------------------------
# Logistic scale (seconds) of the receiver-vs-opponent arrival margin. A level
# arrival is a coin flip; this is how fast the odds move away from 0.5.
PASS_MARGIN_SCALE = 0.25
# How far the ball may stop past the aim point before it starts costing us,
# and where the pass is treated as a pure turnover risk.
PASS_OVERSHOLE_FREE = 2.0
PASS_OVERSHOLE_FATAL = 12.0

# --- Shot decision bars ----------------------------------------------------
# Inside the box the bar is on *reaching the goal*, not on scoring: the effort
# has to be on frame and not into a body. At 8 m from the centre against a set
# keeper P(goal) is around 1%, and demanding a real scoring chance there is
# what produced the original 35-shots-0-goals season.
SHOT_IN_BOX_DIST = 11.0
EV_SHOT_REACHES_GOAL_BAR = 0.05
# Outside the box the bar is a full expected value, and with EV_GOAL_VALUE = 100
# it demands roughly a 1-in-20 goal. That is what keeps the hopeless 18 m
# efforts out: from there the keeper has 0.7 s of flight to cover the far post,
# so a central shot prices out negative and only a keeper dragged off his line
# makes it a real chance.
EV_SHOT_WORTH = 5.0
# A wall shot has two legs, so nobody -- including the keeper -- can plan where
# it ends up. That unpredictability is worth a premium over a direct shot.
EV_WALL_SHOT_BONUS = 1.25

# --- Expected Value Calculator -----------------------------------------------

class ExpectedValueCalculator:
    """Expected value of every action we can take, from real geometry.

    Nothing here is a hardcoded action value. Each action is decomposed into
    mutually exclusive outcomes whose probabilities are *derived* from the
    engine's own numbers -- keeper lateral reach, the distance a pass rolls
    before it may be touched, run speed, and the angle of the goalmouth -- and
    the outcome probabilities of a single action always sum to 1 by
    construction. A shot is a chain:

        P(blocked)  ->  P(off target)  ->  P(on target)  ->  P(goal | on target)

    and a pass is a race between three arrival times:

        t_ball (when the ball is controllable)
        t_us   (when our receiver gets there)
        t_them (when the nearest opponent gets there)

    EV is then the probability-weighted sum of the outcome values.
    """

    def __init__(self, inp: PolicyInput):
        self.inp = inp
        self.state = inp.state
        self.ball = self.state.ball

    # --- shared geometry helpers -------------------------------------------

    def _kick_speed(self, power: float) -> float:
        """Map a 0..1 power to the ball speed the engine will actually use."""
        return geom.clamp(
            KICK_MIN_SPEED + (KICK_MAX_SPEED - KICK_MIN_SPEED) * power,
            KICK_MIN_SPEED,
            KICK_MAX_SPEED,
        )

    def _time_to_control(self, speed0: float) -> float:
        """Seconds until a ball kicked at `speed0` is slow enough to control.

        The engine decays ball speed by BALL_DECAY every 60 Hz tick and only
        allows a control at or below BALL_CONTROL_SAFE_SPEED, so this is the
        time cost every pass pays before anybody can touch it.
        """
        if speed0 <= BALL_CONTROL_SAFE_SPEED:
            return 0.0
        ticks = math.log(BALL_CONTROL_SAFE_SPEED / speed0) / math.log(BALL_DECAY)
        return max(0.0, ticks) / TICKS_PER_SECOND

    def _run_eta(self, fx: float, fy: float, tx: float, ty: float) -> float:
        """Seconds for a runner to cover (fx,fy) -> (tx,ty) at run speed."""
        return geom.distance(fx, fy, tx, ty) / MAX_RUN_SPEED

    def _player_eta(self, pl: Player, tx: float, ty: float) -> float:
        """A player who cannot act this tick cannot arrive in time at all."""
        if not pl.can_act:
            return float("inf")
        return self._run_eta(pl.x, pl.y, tx, ty)

    def _nearest_opponent_eta(self, tx: float, ty: float, skip_gk: bool = True) -> float:
        best = float("inf")
        for o in self.state.them:
            if skip_gk and o.role == "goalkeeper":
                continue
            best = min(best, self._player_eta(o, tx, ty))
        return best

    # --- SHOT MODEL --------------------------------------------------------
    #
    # A shot is decomposed into the exclusive outcomes the engine can produce,
    # in the order they are decided, so the probabilities sum to 1 exactly:
    #
    #     P(blocked)                              a defender's body is in the lane
    #   + P(off target)  = (1-block) * P(wide)   the aim misses the frame
    #   + P(saved)       = (1-block) * P(on) * (1 - P(goal | on))
    #   + P(goal)        = (1-block) * P(on) * P(goal | on)
    #
    # `block` and `wide` are pure geometry. `goal | on target` is the keeper
    # model: it uses GK_LATERAL_REACH (how far he can actually move), the
    # flight time of the ball (how long he has to move it), and how much of
    # the goalmouth is left over after his shift. That is the number
    # `shot_beats_keeper` turns into a boolean, kept here as a probability.

    def _shot_block_probability(
        self, p: Player, target: tuple[float, float], power: float
    ) -> float:
        """P(the shot is blocked by a defender before it reaches the goal).

        Real geometry on the shooter -> target segment. For every opponent we
        project them onto that segment and take two numbers:

            offset  perpendicular distance from the lane
            along   how far down the lane the ball meets it

        A defender is a *candidate* blocker when he is close enough to the
        lane to get a leg in at all (offset < SHOT_BLOCK_REACH). He then wins
        the time-to-intercept race: he only has to cover the offset *in
        excess* of his own body radius, at run speed, before the ball arrives
        at that point. Two things scale his share of the block:

            centrality  1.0 standing in the lane, -> 0 at the edge of reach
            timing      LATE when he arrives with the ball, SETTLED when he
                        has time to set himself

        Multiple blockers compound as 1 - prod(1 - share_i), which is the
        correct reading of a wall: two half-chances are more likely to stop a
        shot than either alone, but never certain.
        """
        lane_len = geom.distance(p.x, p.y, target[0], target[1])
        if lane_len <= 1e-6:
            return 0.0
        ux = (target[0] - p.x) / lane_len
        uy = (target[1] - p.y) / lane_len
        speed0 = max(self._kick_speed(power), 1e-6)
        span = max(SHOT_BLOCK_REACH - SHOT_BLOCK_BODY_RADIUS, 1e-6)

        survive = 1.0
        for o in self.state.outfield_them():
            rel_x = o.x - p.x
            rel_y = o.y - p.y
            along = rel_x * ux + rel_y * uy
            # Only defenders between the shooter and the goal line matter.
            if along <= 0.0 or along >= lane_len:
                continue
            offset = math.hypot(rel_x - along * ux, rel_y - along * uy)
            if offset >= SHOT_BLOCK_REACH:
                continue
            if not o.can_act:
                continue  # frozen this tick: he cannot get in the way
            # He only has to cover the part of the offset his body does not.
            excess = max(0.0, offset - SHOT_BLOCK_BODY_RADIUS)
            t_def = excess / MAX_RUN_SPEED
            t_ball = along / speed0
            if t_def > t_ball:
                continue  # the ball is past him before he can get there
            slack = t_ball - t_def
            timing = (
                SHOT_BLOCK_SHARE_SETTLED
                if slack >= SHOT_BLOCK_T_SETTLED
                else SHOT_BLOCK_SHARE_LATE
                + (slack / SHOT_BLOCK_T_SETTLED)
                * (SHOT_BLOCK_SHARE_SETTLED - SHOT_BLOCK_SHARE_LATE)
            )
            centrality = geom.clamp(1.0 - excess / span, 0.0, 1.0)
            share = geom.clamp(centrality * timing, 0.0, 1.0)
            if share > 0.0:
                survive *= 1.0 - share
        return geom.clamp(1.0 - survive, 0.0, 1.0)

    def p_shot_wide_given_not_blocked(
        self, p: Player, target: tuple[float, float], power: float
    ) -> float:
        """P(the shot misses the goal frame entirely | not blocked).

        Modelled as lateral aiming error rather than a "is the target nicely
        framed" lookup. A struck ball lands at the aim point plus a lateral
        error whose spread grows with range, so what matters is how much of
        the 6 m frame the error distribution still covers:

            sigma = SHOT_ERROR_BASE + SHOT_ERROR_PER_M * range
            P(on frame) = Phi((GOAL_HIGH_Y - aim) / sigma)
                        - Phi((GOAL_LOW_Y  - aim) / sigma)

        This is what makes a post-inset aim a genuinely riskier choice than a
        slightly-less-far-post one, and it is why the trade-off against the
        keeper model is real: a central aim is easier to keep on frame but
        much easier for him to cover.
        """
        sigma = SHOT_ERROR_BASE + SHOT_ERROR_PER_M * max(OPP_GOAL_X - p.x, 0.0)
        sigma = max(sigma, 1e-3)
        p_on = _normal_cdf((GOAL_HIGH_Y - target[1]) / sigma) - _normal_cdf(
            (GOAL_LOW_Y - target[1]) / sigma
        )
        return 1.0 - geom.clamp(p_on, SHOT_ON_TARGET_FLOOR, 1.0)

    def shot_outcome_distribution(
        self, p: Player, target: tuple[float, float], power: float
    ) -> dict[str, float]:
        """The four mutually exclusive, exhaustive shot outcomes.

        This is the single place the shot distribution is built, and the only
        place the factors are combined. It exists because the four terms used
        to be mixed in scope: `p_shot_goal` was marginal (it already folded in
        P(not blocked) and P(on frame)) while `p_shot_wide` and `p_shot_saved`
        were conditional on getting through. Summing those gave 1.96 on a
        shot through a three-man wall, and `ev_shot` hid the error by dividing
        by the total -- which silently re-weighted the outcome *values* as
        well as the probabilities, so a blocked shot was credited with part of
        a goal chance that could not happen.

        The chain is now:

            blocked                                        = b
            wide  = (1-b) * wide|clear                      miss the frame
            on    = (1-b) * on|clear                        reach the frame
            goal  = on     * P(goal | on)                   beats the keeper
            saved = on     * (1 - P(goal | on))             he gets a hand to it

        so blocked + wide + saved + goal == 1 exactly, by construction, for
        every target and every power.
        """
        b = self._shot_block_probability(p, target, power)
        clear = 1.0 - b
        wide_given_clear = self.p_shot_wide_given_not_blocked(p, target, power)
        on_given_clear = 1.0 - wide_given_clear
        p_goal_given_on = self.p_shot_goal_given_on_target(p, target, power)
        on = clear * on_given_clear
        goal = on * p_goal_given_on
        return {
            "blocked": b,
            "wide": clear * wide_given_clear,
            "saved": on * (1.0 - p_goal_given_on),
            "goal": goal,
        }

    def p_shot_blocked(self, p: Player, target: tuple[float, float], power: float) -> float:
        return self.shot_outcome_distribution(p, target, power)["blocked"]

    def p_shot_wide(self, p: Player, target: tuple[float, float], power: float) -> float:
        """P(miss the frame entirely) -- marginal, so it excludes blocked shots."""
        return self.shot_outcome_distribution(p, target, power)["wide"]

    def p_shot_on_target(self, p: Player, target: tuple[float, float], power: float) -> float:
        """P(reach the frame) -- marginal, so it excludes blocked shots.

        Equals saved + goal: everything that got through and was not stopped in
        the air.
        """
        d = self.shot_outcome_distribution(p, target, power)
        return d["saved"] + d["goal"]

    def p_shot_saved(self, p: Player, target: tuple[float, float], power: float) -> float:
        """P(the keeper gets a hand to it) -- marginal."""
        return self.shot_outcome_distribution(p, target, power)["saved"]

    def p_shot_goal(self, p: Player, target: tuple[float, float], power: float) -> float:
        """P(goal) for a shot at `target` with `power`.

        The *marginal* goal probability, obtained by composing the exclusive
        chain block -> on-frame -> beats-keeper. It is deliberately not the
        old binary: a shot that "beats the keeper" from 20 m through a wall of
        defenders still has a small goal probability, and a point-blank one
        with the keeper out of position is close to certain.
        """
        return self.shot_outcome_distribution(p, target, power)["goal"]

    def p_shot_goal_given_on_target(
        self, p: Player, target: tuple[float, float], power: float
    ) -> float:
        """P(goal | not blocked, on frame) -- the keeper model.

        The engine gives a keeper at most GK_LATERAL_REACH of lateral reach,
        and he spends it *while the ball is in flight*. So the decisive number
        is the goalmouth left over after his shift:

            flight = distance / kick speed
            shift  = MAX_RUN_SPEED * flight
            gap    = |gky - target_y| - shift

        gap <= GK_DIVE_LATERAL_MIN -> standing catch, the shot is wasted.
        gap >= GK_LATERAL_REACH    -> out of dive range, it is a goal.
        In between it is a dive: a real but not certain chance. A keeper who
        is dragged out of his own defensive fifth (gk.x < GK_AREA_START)
        cannot auto-handle at all.
        """
        gk = self.state.goalkeeper_them()
        if gk is None or gk.x < 0.0:
            return 0.0  # unknown keeper: never assume a free goal
        if gk.x < GK_AREA_START:
            return SHOT_GOAL_GK_OUT  # cannot handle from outside his own fifth
        speed0 = self._kick_speed(power)
        dist = max(OPP_GOAL_X - p.x, 0.0)
        flight = dist / max(speed0, 1.0)
        shift = MAX_RUN_SPEED * flight
        gap = abs(gk.y - target[1]) - shift
        if gap <= GK_DIVE_LATERAL_MIN:
            return SHOT_GOAL_STANDING_CATCH
        if gap >= GK_LATERAL_REACH:
            return SHOT_GOAL_CLEAN
        # A dive: chance rises smoothly from "standing catch" to "clean".
        span = GK_LATERAL_REACH - GK_DIVE_LATERAL_MIN
        t = (gap - GK_DIVE_LATERAL_MIN) / span
        return SHOT_GOAL_STANDING_CATCH + t * (SHOT_GOAL_CLEAN - SHOT_GOAL_STANDING_CATCH)

    # --- PASS MODEL --------------------------------------------------------
    #
    # A pass is not "short = safe, long = risky". The engine releases every
    # pass at >= KICK_MIN_SPEED, and the ball must decay to a controllable
    # speed before anyone can touch it, so a pass always rolls a minimum
    # distance first (MIN_PASS_TRAVEL, ~15.6 m). The real question is a
    # three-way race at the *collection point* -- where the pass first stops
    # being too fast to control:
    #
    #     t_ball = when the ball is finally controllable there
    #     t_us   = when our receiver can run there
    #     t_them = when the nearest opponent can run there
    #
    # We keep the ball if our receiver arrives first, with the margin between
    # the two ETAs deciding how safe the pass is. This is the same loose-ball
    # lifecycle the receiver-movement code was built for.

    def pass_plan(
        self, p: Player, target: tuple[float, float], power: float | None = None
    ) -> tuple[float, float, float, float, float]:
        """Plan a pass: (power, collect_x, collect_y, t_ball, overshoot).

        A pass is only as good as the spot the ball actually *stops* at, and in
        this engine that is set almost entirely by the power: every pass leaves
        at >= KICK_MIN_SPEED and must decay to a controllable speed, so the
        ball rolls 15.6 m at minimum and ~35 m at power 0.7. When no power is
        supplied we solve for the one that lands on the intended spot with
        `speed_for_travel`; when the spot is closer than the minimum travel no
        such power exists, so we use the slowest kick and report how far the
        ball overshoots past the intended receiver.
        """
        dist = geom.distance(p.x, p.y, target[0], target[1])
        if power is None:
            speed0 = speed_for_travel(dist)
            power = geom.clamp(
                (speed0 - KICK_MIN_SPEED) / (KICK_MAX_SPEED - KICK_MIN_SPEED),
                0.0,
                1.0,
            )
        cx, cy, _, _ = pass_collection_point(p.x, p.y, target[0], target[1], power)
        t_ball = self._time_to_control(self._kick_speed(power))
        overshoot = geom.distance(cx, cy, target[0], target[1])
        return power, cx, cy, t_ball, overshoot

    def pass_collection_time(
        self, p: Player, target: tuple[float, float], power: float | None = None
    ) -> tuple[float, float, float]:
        """(collection_x, collection_y, t_ball) for a pass aimed at `target`."""
        _, cx, cy, t_ball, _ = self.pass_plan(p, target, power)
        return cx, cy, t_ball

    def p_pass_complete(
        self,
        p: Player,
        target: tuple[float, float],
        power: float | None = None,
        receiver_id: str | None = None,
    ) -> float:
        """P(the pass is completed by our receiver) from the loose-ball race.

        The receiver is whoever we intended (or the nearest outfielder). Both
        our receiver and the nearest opponent are raced to the *collection
        point* -- where the ball first becomes slow enough to control -- and
        not to the aim point, because in this engine those are different
        places whenever the pass is short.

        The race is on raw arrival times, which is the physically meaningful
        comparison: whoever reaches the ball first is standing there when it
        slows, and an opponent who gets there early simply waits for it. The
        margin between the two ETAs is what decides it, through a logistic so
        a dead heat is a coin flip and a clear margin is a completed pass.

        A pass that overshoots its intended receiver is additionally
        penalised: the ball is stopping somewhere nobody planned for, which is
        exactly the loose-ball turnover the receiver-movement code exists to
        avoid.
        """
        _, cx, cy, _t_ball, overshoot = self.pass_plan(p, target, power)

        receiver = None
        if receiver_id:
            receiver = next(
                (pl for pl in self.state.outfield_us() if pl.id == receiver_id), None
            )
        if receiver is None:
            cands = [pl for pl in self.state.outfield_us() if pl.can_act]
            if not cands:
                return 0.0
            receiver = min(cands, key=lambda pl: geom.distance(pl.x, pl.y, cx, cy))
        if not receiver.can_act:
            return 0.0

        t_us = self._run_eta(receiver.x, receiver.y, cx, cy)
        t_them = self._nearest_opponent_eta(cx, cy)
        if t_them == float("inf"):
            win = 1.0
        else:
            margin = t_them - t_us  # positive: we arrive first
            win = 1.0 / (1.0 + math.exp(-margin / PASS_MARGIN_SCALE))

        # The ball stopping somewhere other than where we aimed is a sign the
        # pass was misjudged; the further it is from the plan, the worse.
        if overshoot > PASS_OVERSHOLE_FREE:
            penalty = geom.clamp(
                1.0
                - (overshoot - PASS_OVERSHOLE_FREE)
                / (PASS_OVERSHOLE_FATAL - PASS_OVERSHOLE_FREE),
                0.0,
                1.0,
            )
            win *= penalty

        return geom.clamp(win, 0.0, 1.0)

    def p_pass_intercepted(
        self, p: Player, target: tuple[float, float], power: float | None = None
    ) -> float:
        """P(an opponent collects the pass) = 1 - P(our receiver does)."""
        return 1.0 - self.p_pass_complete(p, target, power)

    def p_carry_success(self, p: Player, tx: float, ty: float) -> float:
        """P(the carry keeps the ball) -- contested ground lowers it.

        The keeper counts as an obstacle, and a stricter one than the outfield
        radius. He is not in `outfield_them()`, so he was previously invisible
        here: with the carrier at (52, 20) and the keeper set at (58, 20), the
        dribble target (58.5, 18.2) sat 1.8 m from a stationary goalkeeper and
        still scored 0.85 "uncontested". The carrier was then told to run
        straight into him -- carry EV 6.2 against a shot EV of -1.0 -- and that
        is why the striker stands 8 m from an open goal doing nothing.

        A carry that ends inside GK_LATERAL_REACH is not a contest, it is a
        giveaway: the engine collects any loose ball whose swept path passes
        within that distance.

        Beyond his reach the danger decays smoothly out to
        CARRY_KEEPER_RADIUS rather than at a flat 4 m, because the goalkeeper
        is the single most dangerous obstacle on the pitch for a dribbler and a
        step's worth of daylight is the difference between a cutback and a
        gift. A binary 4 m test put 1.8 m and 3.9 m on the same footing, which
        still left the dribble into a set keeper scoring 0.65.
        """
        if self.ball.possessing_team != "us" or self.ball.possessing_player != p.id:
            return 0.95
        keeper = self.state.goalkeeper_them()
        if keeper is not None:
            keeper_gap = geom.distance(keeper.x, keeper.y, tx, ty)
            if keeper_gap <= GK_LATERAL_REACH:
                return 0.05
            if keeper_gap < CARRY_KEEPER_RADIUS:
                decay = (keeper_gap - GK_LATERAL_REACH) / (
                    CARRY_KEEPER_RADIUS - GK_LATERAL_REACH
                )
                base = 0.05 + (0.85 - 0.05) * geom.clamp(decay, 0.0, 1.0)
            else:
                base = 0.85
        else:
            base = 0.85
        contested = any(
            geom.distance(o.x, o.y, tx, ty) < 4.0 for o in self.state.outfield_them()
        )
        return base * (0.65 if contested else 1.0)

    def p_carry_tackled(self, p: Player, tx: float, ty: float) -> float:
        return 1.0 - self.p_carry_success(p, tx, ty)

    def p_tackle_success(self, tackler: Player, ball_holder: Player) -> float:
        dist = geom.distance(tackler.x, tackler.y, ball_holder.x, ball_holder.y)
        if dist <= TACKLE_CLOSE:
            return 0.85
        if dist <= TACKLE_MAX:
            return 0.60
        return 0.0

    # --- VALUE MODELS ------------------------------------------------------

    def value_goal(self) -> float:
        return EV_GOAL_VALUE

    def value_shot_saved(self) -> float:
        return EV_SHOT_ON_TARGET_SAVED

    def value_shot_blocked(self) -> float:
        return EV_SHOT_BLOCKED

    def value_shot_wide(self) -> float:
        return EV_SHOT_WIDE

    def value_pass_completed(self, p: Player, target: tuple[float, float]) -> float:
        """What a completed pass is worth: the ground it wins, plus a small
        amount for having kept the ball at all.

        The progress term is signed on purpose. With `max(0, progress)` a pass
        from the box back to the halfway line scored exactly the same as a
        square pass, so the cheapest-looking option in the whole decision was
        always to roll it sideways or backwards, and the shot -- however good --
        never got chosen. Losing ground is priced more heavily than winning it.
        """
        gain = target[0] - self.ball.x
        if gain >= 0.0:
            base = EV_PASS_COMPLETED + gain * EV_PASS_PROGRESS_PER_M
        else:
            base = EV_PASS_COMPLETED + gain * EV_PASS_REGRESSION_PER_M
        # Landing the ball in a part of the pitch that threatens the goal is
        # worth more than the metres alone: from the final third we can shoot
        # this pass sequence out, from our own half we cannot.
        #
        # These two are the only levers that actually move the ball into the
        # box, and a traced engine match showed why they need to be strong: the
        # carrier was inside the 11 m box on 1 tick out of 276 possession ticks
        # (0.4%) and the team took no shots at all. Every shot this team does
        # take converts, so the whole game is won or lost on how often the ball
        # gets here, not on how good the shot is once it is taken.
        if target[0] > 40.0:
            base += EV_PASS_FINAL_THIRD_BONUS
        if target[0] > 48.0:
            base += EV_PASS_BOX_BONUS
        return base

    def value_pass_intercepted(self, target: tuple[float, float]) -> float:
        if target[0] > 40.0:
            return EV_LOSS_OF_POSSESSION
        if target[0] > 20.0:
            return EV_LOSS_OF_POSSESSION / 2
        return EV_LOSS_OF_POSSESSION_SAFE

    def value_carry_progress(self, p: Player, tx: float) -> float:
        gain = tx - p.x
        if gain <= 0:
            return EV_CARRY_HOLD
        progress = min(gain / CARRY_FULL_GAIN, 1.0)
        base = EV_CARRY_HOLD + (EV_CARRY_PROGRESS - EV_CARRY_HOLD) * progress
        if OPP_GOAL_X - tx <= CARRY_SHOOT_RANGE:
            base += EV_CARRY_PROGRESS * 0.5
        return base

    def value_carry_loss(self, p: Player) -> float:
        if p.x > 40.0:
            return EV_LOSS_OF_POSSESSION
        if p.x > 20.0:
            return EV_LOSS_OF_POSSESSION / 2
        return EV_LOSS_OF_POSSESSION_SAFE

    # --- ACTION EV ---------------------------------------------------------

    def p_shot_forces_dive(self, p: Player, target: tuple[float, float], power: float) -> float:
        """P(the keeper has to dive), given the shot is not blocked.

        A dive (0.8 m to 1.65 m of lateral reach) is the outcome that hands us
        something back: the keeper stays grounded holding the ball for
        GK_GROUNDED_SECONDS and cannot distribute for the whole of it. It is
        also the reason a shot from 8 m against a set keeper is still worth
        taking, even though its P(goal) is about 1%. `ev_shot` prices the dive
        explicitly instead of burying it in a made-up "shoot on sight" bonus.
        """
        keeper = self.state.goalkeeper_them()
        if keeper is None or keeper.x < GK_AREA_START:
            return 0.0
        p_given_on = self.p_shot_goal_given_on_target(p, target, power)
        # STANDING_CATCH -> CLEAN is the whole dive band; CLEAN means out of
        # reach, i.e. a goal, so the dive band is the part just above the
        # standing catch.
        clean = SHOT_GOAL_CLEAN
        floor = SHOT_GOAL_STANDING_CATCH
        if clean <= floor:
            return 0.0
        frac = geom.clamp((p_given_on - floor) / (clean - floor), 0.0, 1.0)
        return frac

    def ev_shot(self, p: Player, target: tuple[float, float], power: float) -> float:
        """EV of a shot: the exclusive chain weighted by outcome values.

        Because block/wide/saved/goal are composed from one distribution that
        sums to 1, this needs no ad-hoc renormalisation -- a blocked shot is
        worth exactly EV_SHOT_BLOCKED times P(block), and so on.

        Two engine facts make a failed shot less bad than a turnover, and both
        are credited here rather than assumed:

        * goal kicks do not exist, so a miss rebounds off the goal-line wall
          and the ball is still loose for us to collect (see EV_SHOT_WIDE);
        * a dive leaves the keeper grounded for GK_GROUNDED_SECONDS, which is a
          free possession we could not otherwise force, so P(dive) is credited
          with EV_SHOT_SECOND_BALL.

        There is no division by the total here. The distribution is exclusive
        and sums to 1 on its own, so renormalising would be a no-op at best; it
        was previously masking a real error, because the four terms were a mix
        of marginal and conditional and summed to nearly 2 through a wall.
        """
        d = self.shot_outcome_distribution(p, target, power)
        # A dive needs the ball to actually reach the frame, and the marginal
        # "reaches the frame" probability is saved + goal.
        dive = self.p_shot_forces_dive(p, target, power) * (d["saved"] + d["goal"])
        return (
            d["blocked"] * self.value_shot_blocked()
            + d["wide"] * self.value_shot_wide()
            + d["saved"] * (self.value_shot_saved() + EV_SHOT_SECOND_BALL * dive)
            + d["goal"] * self.value_goal()
        )

    def ev_pass(
        self,
        p: Player,
        target: tuple[float, float],
        power: float,
        receiver_id: str | None = None,
        multiplier: float = 1.0,
    ) -> float:
        p_complete = self.p_pass_complete(p, target, power, receiver_id)
        p_intercepted = 1.0 - p_complete
        value_complete = self.value_pass_completed(p, target)
        value_intercepted = self.value_pass_intercepted(target)
        return multiplier * (
            p_complete * value_complete + p_intercepted * value_intercepted
        )

    def ev_carry(self, p: Player, tx: float, ty: float) -> float:
        p_success = self.p_carry_success(p, tx, ty)
        p_loss = 1.0 - p_success
        return p_success * self.value_carry_progress(p, tx) + p_loss * self.value_carry_loss(p)

    def ev_through_ball(
        self, p: Player, target: tuple[float, float], power: float, receiver_id: str
    ) -> float:
        base = self.ev_pass(p, target, power, receiver_id, EV_THROUGH_BALL_MULTIPLIER)
        if OPP_GOAL_X - target[0] <= 15.0:
            base *= 1.3
        return base

    def ev_cutback(
        self, p: Player, target: tuple[float, float], power: float, receiver_id: str
    ) -> float:
        base = self.ev_pass(p, target, power, receiver_id, EV_CUTBACK_MULTIPLIER)
        if target[0] > 35.0:
            base *= EV_CUTBACK_SHOT_BONUS
        return base

    def ev_cross(
        self, p: Player, target: tuple[float, float], power: float, receiver_id: str
    ) -> float:
        base = self.ev_pass(p, target, power, receiver_id, EV_CROSS_MULTIPLIER)
        if target[0] > 42.0:
            base *= EV_CROSS_HEADER_BONUS
        return base

    def ev_wall_pass(
        self, p: Player, target: tuple[float, float], power: float, receiver_id: str
    ) -> float:
        p_leg1 = self.p_pass_complete(p, target, power, receiver_id)
        p_leg2 = 0.75
        p_complete = p_leg1 * p_leg2
        p_intercepted = 1.0 - p_complete
        value_complete = self.value_pass_completed(p, target) * EV_WALL_PASS_MULTIPLIER
        return p_complete * value_complete + p_intercepted * self.value_pass_intercepted(target)

    def ev_switch_play(
        self, p: Player, target: tuple[float, float], power: float, receiver_id: str
    ) -> float:
        return self.ev_pass(p, target, power, receiver_id, EV_SWITCH_MULTIPLIER)

    def ev_pullback(
        self, p: Player, target: tuple[float, float], power: float, receiver_id: str
    ) -> float:
        return self.ev_pass(p, target, power, receiver_id, 1.2)

    def ev_onetwo(
        self, p: Player, target: tuple[float, float], power: float, receiver_id: str
    ) -> float:
        p_complete = self.p_pass_complete(p, target, power, receiver_id) ** 2
        return p_complete * self.value_pass_completed(p, target) * 1.3

    def ev_third_man(
        self, p: Player, target: tuple[float, float], power: float, receiver_id: str
    ) -> float:
        return self.ev_pass(p, target, power, receiver_id, 1.2)

    def ev_gk_bypass(
        self, p: Player, target: tuple[float, float], power: float, receiver_id: str
    ) -> float:
        return self.ev_pass(p, target, power, receiver_id, 1.2)

# --- End Expected Value Calculator ------------------------------------------
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
    GK_AREA_START,
    GK_DIVE_LATERAL_MIN,
    GK_LATERAL_REACH,
    MIN_PASS_TRAVEL,
    pass_collection_point,
    pass_lane_clear,
    pick_shot_target,
    plan_lead_pass,
    shot_beats_keeper,
    shot_lane_clear,
    shot_open_angle,
    wall_pass_target,
    wall_shot_target,
    speed_for_travel,
    _power_from_speed,
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
        match_duration = getattr(self, 'match_duration', None)
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
# For shooting_threshold: higher value = longer shooting range (more aggressive).
# Trailing late -> more aggressive shooting (higher threshold).
# Leading late -> more conservative (shorter range).
_CONTEXT_MODS: dict[str, dict[str, float]] = {
    "leading_late": {
        "passing_risk": 0.6,
        "verticality": 0.8,
        "shooting_threshold": 0.8,
        "wall_shot_threshold": 0.8,
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
        "shooting_threshold": 1.2,
        "wall_shot_threshold": 1.15,
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
    # Coordinated receiver movement for passes:
    # When a pass is selected, we store the intended receiver and collection point
    # so that _decide_off_ball can move the receiver toward the collection point.
    _pending_pass_receiver: str | None = None
    _pending_pass_collection: tuple[float, float] | None = None

    def __init__(self) -> None:
        self.log = PolicyLog()
        self._pending_pass_receiver = None
        self._pending_pass_collection = None

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
        
        # Process possessor FIRST so that any pass decision sets the pending
        # pass receiver before off-ball players are evaluated.
        if possessor is not None:
            intent = self._decide_possessor(inp, possessor)
            intents[possessor.id] = intent
        
        for p in ours:
            if possessor is not None and p.id == possessor.id:
                continue
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
        # One calculator for every candidate below. Every value on this board is
        # a goal-equivalent expected value computed by this class, so shots,
        # passes, carries and special balls are all on one scale and can be
        # compared directly.
        #
        # This board used to mix real numbers with a set of made-up constants
        # (cross 25, cutback 35, through ball 40, wall pass 30, pullback 28,
        # one-two 22, switch 20, safe pass 10). Every one of them was larger
        # than any honest shot, and a shot from 8 m against a set keeper is
        # worth about -1. So the striker stood 8 m from an open goal and chose
        # a through ball instead. That single mismatch is the direct cause of
        # the 1-shot-per-match season.
        evcalc = ExpectedValueCalculator(inp)
        
# ---- 1. SHOOT ----
        # Compute shot directly using pick_shot_target (bypasses _shot_choice GK HELL blocking)
        gk = state.goalkeeper_them()
        gkx = gk.x if gk else -1.0
        gky = gk.y if gk else 20.0
        dist_goal = OPP_GOAL_X - p.x
        max_dist = shooting_distance(self._cfg(inp, "shooting_threshold", 0.5))

        # ---- 1. SHOOT: value the real shot, geometrically ----
        # The decision is made by the decomposed outcome distribution
        # (blocked / wide / saved / goal) of the best shot available from
        # here, through `ExpectedValueCalculator.ev_shot`. There is no
        # nominal-position branch anywhere in it: a centre-back who has
        # carried the ball to the same spot as a striker gets the same answer
        # for the same picture, and both are told to pass if the shot is a
        # standing catch.
        if dist_goal <= max_dist:
            (tx, ty), power = _shot_geometry_target(inp, p)
            shot_ev = ExpectedValueCalculator(inp).ev_shot(p, (tx, ty), power)
            if _shot_worth_taking(inp, p):
                intent = PlayerIntent(
                    p.id, p.x, p.y, 0.4, OPP_GOAL_X, GOAL_CENTER_Y,
                    "shoot", (tx, ty), power,
                )
                # The value *is* the EV, so shots compete with passes and
                # carries on the same scale instead of a made-up constant.
                candidates.append((shot_ev, intent, "shoot"))
        
        # ---- Shoot-on-sight in box: if we're in the box with any opening, shoot! ----
        # Low blocks leave small windows - don't wait for perfect lane.
        # NOTE: the old "shoot-on-sight in the box" trigger used to live here,
        # worth a hardcoded 45.0 and gated on `open_angle` rather than on the
        # shot's actual value. It duplicated the block above and, because 45.0
        # was a constant, it outbid a genuine open goal from range no matter
        # how hopeless that was. Inside the box is now handled where it
        # belongs: `EV_SHOT_WORTH_IN_BOX` lowers the bar for close range, so a
        # clear chance at 8 m is still taken while a covered one is not.

        # Wall shot: only when NOT in clear 1v1 (ball close to goal) and near wall
        # In 1v1, direct shot is better - wall shot adds unpredictability
        if (is_near_wall(p.x, p.y, margin=6.0) and 8.0 < dist_goal <= max_dist):
            wall_shot = wall_shot_target(p.x, p.y, gkx, gky)
            if wall_shot:
                wx, wy, power = wall_shot
                # Verify wall shot lane is clear
                opp_pos = [(o.x, o.y) for o in state.outfield_them()]
                if pass_lane_clear(p.x, p.y, wx, wy, opp_pos, margin=0.12):
                    # Check second leg (wall to goal) is also clear
                    if pass_lane_clear(wx, wy, OPP_GOAL_X, wy, opp_pos, margin=0.12):
                        # A wall shot is a real shot: value it through the same
                        # distribution rather than a made-up 30.0, but award it
                        # the unpredictability bonus it earns (two legs means a
                        # deflection nobody can plan for, including the keeper).
                        wall_ev = ExpectedValueCalculator(inp).ev_shot(
                            p, (wx, wy), power
                        ) * EV_WALL_SHOT_BONUS
                        # A wall shot is a `shoot`, not its own action: the
                        # engine only accepts none/pass/shoot/clear/tackle/
                        # slap, so emitting "wall_shot" here put an illegal
                        # action on the wire. The tactical difference is the
                        # target -- the deflection spot rather than the goal --
                        # which is what this intent carries.
                        intent = PlayerIntent(p.id, p.x, p.y, 0.4, wx, wy, "shoot", (wx, wy), power)
                        candidates.append((wall_ev, intent, "wall_shot"))

        # NOTE: the "test the keeper from range" branch used to live here. It
        # was gated on `_is_natural_striker`, i.e. on the player's shirt rather
        # than on the picture, and it fired from up to 1.5x the shooting range
        # on a fixed value of 12.0. That is precisely the exception the shot
        # model is meant to remove: from 15-25 m the only shots worth taking
        # are the ones where the keeper has been dragged off his line, and
        # `ev_shot` already scores those far higher than 12.0 while scoring the
        # covered ones below zero. Long-range shooting is now available to
        # whoever has the ball when -- and only when -- the geometry pays.

        
        # NOTE: a "rebound setup" candidate used to live here -- shoot at the
        # keeper's body from up to 20 m, worth 45.0, tying the best genuine shot
        # in this planner. It was removed because it is the one shot a keeper is
        # guaranteed to deal with: aiming at GOAL_CENTER_Y from in front of a
        # keeper sitting on the centre line needs well under GK_DIVE_LATERAL_MIN
        # of lateral reach, which the engine calls a standing catch. It also
        # never consulted shot_beats_keeper, unlike the test_keeper branch
        # above, and its move target was the goal itself. See
        # test_no_shot_is_aimed_at_the_keeper_body.

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
                # Running to a loose ball is a carry, so it is priced as one.
                # It was a flat 35.0, which is above any honest shot and would
                # win this board outright whenever it fired.
                second_ball_value = evcalc.ev_carry(p, mx, my)
                intent = PlayerIntent(p.id, mx, my, 1.0, mx, my, "none")
                candidates.append((second_ball_value, intent, "second_ball"))
        
        # ---- 2. CROSS FROM WING ----
        if role in (ROLE_WIDE_LEFT, ROLE_WIDE_RIGHT) and p.x > 45.0:
            cross = self._cross_choice(inp, p)
            if cross is not None:
                tx, ty, power, receiver_id, collect_x, collect_y = cross
                # Cross value depends on striker position and box occupancy
                cross_value = evcalc.ev_cross(p, (tx, ty), power, receiver_id)
                intent = PlayerIntent(p.id, p.x, p.y, 0.4, tx, ty, "pass", (tx, ty), power)
                candidates.append((cross_value, intent, "cross"))
        
        # ---- 3. CUTBACK FROM BYLINE ----
        cutback = self._cutback_choice(inp, p)
        if cutback is not None:
            # Cutback creates high-quality chance at edge of box
            tx, ty, power, receiver_id, collect_x, collect_y = cutback
            cutback_value = evcalc.ev_cutback(p, (tx, ty), power, receiver_id)
            intent = PlayerIntent(p.id, p.x, p.y, 0.4, tx, ty, "pass", (tx, ty), power)
            candidates.append((cutback_value, intent, "cutback"))

        # ---- 4. THROUGH BALL ----
        through = self._through_ball_choice(inp, p)
        if through is not None:
            tx, ty, power, receiver_id, collect_x, collect_y = through
            through_value = evcalc.ev_through_ball(p, (tx, ty), power, receiver_id)
            intent = PlayerIntent(p.id, p.x, p.y, 0.4, tx, ty, "pass", (tx, ty), power)
            candidates.append((through_value, intent, "through_ball"))
        
        # ---- 5. WALL PASS ----
        wall = self._wall_pass_choice(inp, p)
        if wall is not None:
            tx, ty, power, receiver_id, collect_x, collect_y = wall
            wall_value = evcalc.ev_wall_pass(p, (tx, ty), power, receiver_id)
            intent = PlayerIntent(p.id, p.x, p.y, 0.4, tx, ty, "pass", (tx, ty), power)
            candidates.append((wall_value, intent, "wall_pass"))
        
        # ---- 6. SWITCH PLAY ----
        switch = self._switch_play_choice(inp, p)
        if switch is not None:
            tx, ty, power, receiver_id, collect_x, collect_y = switch
            switch_value = evcalc.ev_switch_play(p, (tx, ty), power, receiver_id)
            intent = PlayerIntent(p.id, p.x, p.y, 0.4, tx, ty, "pass", (tx, ty), power)
            candidates.append((switch_value, intent, "switch_play"))
        
        # ---- 7. PULL BACK ----
        pullback = self._pullback_choice(inp, p)
        if pullback is not None:
            tx, ty, power, receiver_id, collect_x, collect_y = pullback
            pullback_value = evcalc.ev_pullback(p, (tx, ty), power, receiver_id)
            intent = PlayerIntent(p.id, p.x, p.y, 0.4, tx, ty, "pass", (tx, ty), power)
            candidates.append((pullback_value, intent, "pullback"))
        
        # ---- 8. ONE-TWO (High Press Escape) ----
        onetwo = self._onetwo_choice(inp, p)
        if onetwo is not None:
            tx, ty, power, receiver_id, collect_x, collect_y = onetwo
            onetwo_value = evcalc.ev_onetwo(p, (tx, ty), power, receiver_id)
            intent = PlayerIntent(p.id, p.x, p.y, 0.4, tx, ty, "pass", (tx, ty), power)
            candidates.append((onetwo_value, intent, "high_press_onetwo"))
        
        # ---- 9. THIRD MAN RUN ----
        third = self._third_man_choice(inp, p)
        if third is not None:
            tx, ty, power, receiver_id, collect_x, collect_y = third
            third_value = evcalc.ev_third_man(p, (tx, ty), power, receiver_id)
            intent = PlayerIntent(p.id, p.x, p.y, 0.4, tx, ty, "pass", (tx, ty), power)
            candidates.append((third_value, intent, "high_press_third_man"))
        
        # ---- 10. GK BYPASS (High Press) ----
        if inp.roles.get(p.id) == "goalkeeper":
            bypass = self._gk_bypass_choice(inp, p)
            if bypass is not None:
                tx, ty, power, receiver_id, collect_x, collect_y = bypass
                bypass_value = evcalc.ev_gk_bypass(p, (tx, ty), power, receiver_id)
                intent = PlayerIntent(p.id, p.x, p.y, 0.4, tx, ty, "pass", (tx, ty), power)
                candidates.append((bypass_value, intent, "high_press_gk_bypass"))
        
        # ---- 13. WALL SHOT ----
        # `is_near_wall` is a pitch-boundary test (x <= m or x >= L - m or
        # y <= m or y >= W - m), so without a range bound this also fires for a
        # defender hugging a touchline in our own defensive third. Traced: 5 shot
        # requests were refused by the engine, 100% of all shots we ever ask for,
        # and 2 of them were launched from x ~ 4-5 m, i.e. 55 m from goal, at
        # power 0.95. A wasted action is a wasted possession-protection window.
        if (is_near_wall(p.x, p.y, margin=6.0)
                and 8.0 < (OPP_GOAL_X - p.x) <= max_dist):
            wall_shot = wall_shot_target(p.x, p.y, gkx if 'gkx' in dir() else -1.0, gky if 'gky' in dir() else 20.0)
            if wall_shot:
                wx, wy, power = wall_shot
                # Verify both legs clear
                opp_pos = [(o.x, o.y) for o in state.outfield_them()]
                if pass_lane_clear(p.x, p.y, wx, wy, opp_pos, margin=0.12):
                    if pass_lane_clear(wx, wy, OPP_GOAL_X, wy, opp_pos, margin=0.12):
                        # Same distribution as a direct shot, plus the
                        # unpredictability a two-leg wall shot earns: nobody,
                        # keeper included, can plan where it ends up.
                        wall_shot_value = ExpectedValueCalculator(inp).ev_shot(
                            p, (wx, wy), power
                        ) * EV_WALL_SHOT_BONUS
                        intent = PlayerIntent(p.id, p.x, p.y, 0.4, wx, wy, "shoot", (wx, wy), power)
                        candidates.append((wall_shot_value, intent, "wall_shot"))
        
        # ---- 14. CARRY (Dribble) ----
        # Priced on the goal-equivalent scale with everything else, so a carry
        # and a shot are directly comparable. `_carry_value` scored carries on
        # its own CARRY_* scale, where a clean dribble into the box was worth
        # CARRY_CLEAN_VALUE + CARRY_SHOOT_BONUS and quietly outbid a correctly
        # priced shot -- so the striker would dribble at a keeper instead of
        # shooting past him.
        tx, ty, speed = self._dribble_target(inp, p)
        carry_value = evcalc.ev_carry(p, tx, ty)
        carry_intent = PlayerIntent(p.id, tx, ty, speed, tx, ty)
        candidates.append((carry_value, carry_intent, "carry_forward"))
        
        # ---- 15. SAFE PASS (last resort) ----
        # Priced like every other candidate, on the same scale. This used to be
        # a flat 10.0, which is worse than useless: it was high enough to beat a
        # correctly-priced shot (8 m out is a standing catch at ~1% xG, so the
        # honest shot EV is around -1) and it was *not* directional, so it won
        # the ranking from a position 8 m from goal just as readily as from our
        # own half. The team therefore rolled the ball sideways out of the box
        # instead of shooting, which is the whole 35-shots-0-goals story.
        safe_pass = self._collectable_pass(inp, p)
        if safe_pass is not None:
            best, tx, ty, power, collect_x, collect_y = safe_pass
            pass_value = ExpectedValueCalculator(inp).ev_pass(
                p, (tx, ty), power, receiver_id=best
            )
            intent = PlayerIntent(p.id, p.x, p.y, 0.4, tx, ty, "pass", (tx, ty), power)
            candidates.append((pass_value, intent, f"safe_pass_to_{best}"))
        
        # ---- Select best candidate ----
        # Veto physically impossible passes.
        #
        # A kick leaves at KICK_MIN_SPEED (12 m/s) no matter how little power is
        # asked for, and nobody may touch the ball again until it has slowed to
        # 5 m/s, so it unavoidably rolls MIN_PASS_TRAVEL (~15.6 m). A pass aimed
        # at a point closer than that cannot be collected where it was aimed; it
        # sails past and becomes a turnover. The lead-pass generator already
        # models this (see `_lead_pass_options`, which rejects an overshoot over
        # 2.5 m), but the ten other pass helpers below aim at a receiver's
        # current position without checking it.
        #
        # Measured over 24 traced matches (186 executed passes):
        #   * 66.7% were aimed inside MIN_PASS_TRAVEL, median aim distance 8.8 m
        #     and p10/p25 of 0.00 m / 0.21 m -- aimed at a point on top of the
        #     passer itself;
        #   * median overshoot past the aim point was 14.1 m;
        #   * power sat at the floor (median 0.006) in 68.8% of them, i.e. the
        #     code asked for the shortest legal kick at a target it could not
        #     reach;
        #   * an opponent was closer to the landing point than any teammate in
        #     58.1% of passes, and 59.1% were intercepted outright.
        #
        # So this is the mechanism behind the loose-ball state: 51% of every
        # loose ball in a match is a pass of ours, 97.1% of our possessions end
        # with our own kick, and 67.8% of recoveries are re-lost within 0.5 s.
        # Winning the ball is not the problem -- we win it with the ball at a
        # median 0.00 m/s -- and then immediately launch it out of reach.
        #
        # Dropping these candidates is the secure-control state: with no
        # physically reachable pass on the board, the carry keeps the ball
        # glued to the carrier instead of handing it to the press.
        if candidates:
            reachable = []
            for value, intent, reason in candidates:
                if intent.action_type == "pass" and intent.action_target is not None:
                    if geom.distance(p.x, p.y, intent.action_target[0],
                                     intent.action_target[1]) < MIN_PASS_TRAVEL:
                        continue
                reachable.append((value, intent, reason))
            candidates = reachable

        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            best_value, best_intent, reason = candidates[0]
            
            # If the chosen action is a pass, store the receiver and collection point
            # so that _decide_off_ball can move the receiver toward the collection point.
            if best_intent.action_type == "pass" and best_intent.action_target is not None:
                target = best_intent.action_target
                # Find which teammate is closest to the target (intended receiver)
                for t in inp.state.outfield_us():
                    if t.id != p.id and t.can_act:
                        dist = geom.distance(t.x, t.y, target[0], target[1])
                        if dist < 10.0:  # Close enough to be the intended receiver
                            self._pending_pass_receiver = t.id
                            self._pending_pass_collection = target
                            break
            
            log_entry = {"state": str(inp.tactical_state), "player": p.id, "action": best_intent.action_type, "reason": reason, "value": best_value}
            self.log.record(log_entry)
            return self._veto_own_goal(inp, p, best_intent)
        
        # Fallback
        tx, ty, speed = self._dribble_target(inp, p)
        log_entry = {"state": str(inp.tactical_state), "player": p.id, "action": "dribble", "reason": "fallback_carry"}
        self.log.record(log_entry)
        return PlayerIntent(p.id, tx, ty, speed, tx, ty)
    
    def _cross_choice(self, inp: PolicyInput, p: Player) -> tuple[float, float, float, str, float, float] | None:
        """Cross from wing to striker in box.
        
        Returns: (tx, ty, power, receiver_id, collect_x, collect_y)
        """
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
                        return (collect_x, collect_y, power, striker.id, collect_x, collect_y)
    
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
    def _cutback_choice(self, inp: PolicyInput, p: Player) -> tuple[float, float, float, str, float, float] | None:
        """Cutback from byline when no clear forward path exists.
        
        Returns: (tx, ty, power, receiver_id, collect_x, collect_y)
        """
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
                            return (collect_x, collect_y, power, t.id, collect_x, collect_y)
        return None

    # -- 4. THROUGH BALL -----------------------------------------------
    def _through_ball_choice(self, inp: PolicyInput, p: Player) -> tuple[float, float, float, str, float, float] | None:
        """Through ball to runner behind high defensive line.
        
        Returns: (tx, ty, power, receiver_id, collect_x, collect_y)
        """
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
                        return (collect_x, collect_y, power, t.id, collect_x, collect_y)
        return None

    # -- 6. SWITCH PLAY ------------------------------------------------
    def _switch_play_choice(self, inp: PolicyInput, p: Player) -> tuple[float, float, float, str, float, float] | None:
        """Quick switch to opposite flank when ball is on one side.
        
        Returns: (tx, ty, power, receiver_id, collect_x, collect_y)
        """
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
                                return (collect_x, collect_y, power, t.id, collect_x, collect_y)
        return None

    # -- 7. PULL BACK --------------------------------------------------
    def _pullback_choice(self, inp: PolicyInput, p: Player) -> tuple[float, float, float, str, float, float] | None:
        """Striker pulls back to edge of box for arriving winger/midfielder.
        
        Returns: (tx, ty, power, receiver_id, collect_x, collect_y)
        """
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
                            return (collect_x, collect_y, power, t.id, collect_x, collect_y)
        return None

    # -- 8. ONE-TWO (High Press Escape) --------------------------------
    def _onetwo_choice(self, inp: PolicyInput, p: Player) -> tuple[float, float, float, str, float, float] | None:
        """Quick one-two with nearby teammate under high press.
        
        Returns: (tx, ty, power, receiver_id, collect_x, collect_y)
        """
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
                    collect_x, collect_y, power, _ = pass_collection_point(
                        p.x, p.y, plan.target_x, plan.target_y, 0.0
                    )
                    return (collect_x, collect_y, power, t.id, collect_x, collect_y)
        return None

    # -- 9. THIRD MAN RUN ----------------------------------------------
    def _third_man_choice(self, inp: PolicyInput, p: Player) -> tuple[float, float, float, str, float, float] | None:
        """Third man run - pass to player with space ahead.
        
        Returns: (tx, ty, power, receiver_id, collect_x, collect_y)
        """
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
                        collect_x, collect_y, power, _ = pass_collection_point(
                            p.x, p.y, plan.target_x, plan.target_y, 0.0
                        )
                        return (collect_x, collect_y, power, t.id, collect_x, collect_y)
        return None

    # -- 10. GK BYPASS (High Press) ------------------------------------
    def _gk_bypass_choice(self, inp: PolicyInput, p: Player) -> tuple[float, float, float, str, float, float] | None:
        """GK distribution bypass under high press.
        
        Returns: (tx, ty, power, receiver_id, collect_x, collect_y)
        """
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
                        collect_x, collect_y, power, _ = pass_collection_point(
                            p.x, p.y, plan.target_x, plan.target_y, 0.0
                        )
                        return (collect_x, collect_y, power, t.id, collect_x, collect_y)
        return None

    def _collectable_pass(
        self, inp: PolicyInput, p: Player
    ) -> tuple[str, float, float, float, float, float] | None:
        """Best pass whose receiver can realistically get to the ball.

        RULES.md: a free ball is collectable only below 5 m/s, and every pass
        action releases it at 12 m/s or more. The ball therefore always rolls
        at least ~15 m before anybody may touch it, which no one can chase
        down from rest. So a pass is only worth taking when the receiver is
        already near the landing spot, unmarked, with a clear lane, and the
        pass actually moves us forward. Otherwise we keep carrying the ball.

        Returns: (receiver_id, tx, ty, power, collect_x, collect_y)
        """
        state = inp.state
        best: tuple[str, float, float, float, float, float] | None = None
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
                collect_x, collect_y, _, _ = pass_collection_point(
                    p.x, p.y, tx, ty, power
                )
                best = (rid, tx, ty, power, collect_x, collect_y)
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
        
                # Wall shot first: it is only offered from near a wall, and it is a
        # real shot, so it is valued through the same distribution.
        if (is_near_wall(p.x, p.y, margin=6.0) and 8.0 < dist_goal <= max_dist):
            wall_shot = wall_shot_target(p.x, p.y, gkx, gky)
            if wall_shot:
                wx, wy, power = wall_shot
                opp_pos = [(o.x, o.y) for o in state.outfield_them()]
                if pass_lane_clear(p.x, p.y, wx, wy, opp_pos, margin=0.12):
                    if pass_lane_clear(wx, wy, OPP_GOAL_X, wy, opp_pos, margin=0.12):
                        return (wx, wy, power)

        # The direct shot, aimed at the corner the keeper cannot cover after
        # his own shift, and judged by `ev_shot`. There is no "GK hell" branch
        # any more and no nominal-position exception: a keeper sitting on the
        # centre line already makes the shot a standing catch, so `_shot_ev`
        # prices it below the bar on its own. Dragging him wide is what raises
        # the value, and the attackers are pulled wide by the shape logic
        # rather than by a special case in the shot code.
        (tx, ty), power = _shot_geometry_target(inp, p)
        if not _shot_worth_taking(inp, p):
            return None

        opponents = [(o.x, o.y) for o in state.outfield_them() if o.x > p.x]
        lane_margin = 0.08 if inside_box else 0.15
        if shot_lane_clear(p.x, p.y, tx, ty, opponents, margin=lane_margin):
            return (tx, ty, power)

        # A blocked lane is already priced into `ev_shot` (P(blocked) reduces
        # P(goal)), so there is no need for a second "is the lane clear?"
        # gate here: the shot is worth taking exactly when the distribution
        # says so, lane included.
        return (tx, ty, power)
        # Wall shot: only when NOT in clear 1v1 (ball close to goal) and near wall
        # In 1v1, direct shot is better - wall shot adds unpredictability
        if (is_near_wall(p.x, p.y, margin=6.0) and 8.0 < dist_goal <= max_dist):
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
        # Same boundary-vs-box gate bug as the copy in
        # _evaluate_attack_actions: is_near_wall(margin=4.0) tests the pitch
        # boundary, not the box, so this fallback was unreachable for a central
        # attacker and only rescued players pressed into the byline.
        open_angle = shot_open_angle(p.x, p.y, opponents)
        if inside_box or is_near_wall(p.x, p.y, margin=4.0):
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

    def _wall_pass_choice(self, inp: PolicyInput, p: Player) -> tuple[float, float, float, str, float, float] | None:
        """Wall pass to bypass opponents.
        
        Returns: (tx, ty, power, receiver_id, collect_x, collect_y)
        """
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
                            return (wall_x, contact_y, power, t.id, wall_x, contact_y)
        
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
                                return (cand.contact_x, cand.contact_y, power, t.id, cand.contact_x, cand.contact_y)
        
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
                best_wall = (cand.contact_x, cand.contact_y, _power_from_speed(speed_for_travel(dist)), t.id, cand.contact_x, cand.contact_y)
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

    def _segment_contest(self, state, ax: float, ay: float, bx: float, by: float) -> float:
        """How much of the run a->b the opponents' outfielders stand in, 0.0..1.0.

        Same geometry _dribble_target uses to choose a heading, so the value we
        put on a carry is measured over the lane we actually picked rather than
        a straight line to goal that may never be attempted.
        """
        vx, vy = bx - ax, by - ay
        seg = vx * vx + vy * vy
        worst = 0.0
        for o in state.outfield_them():
            t = 0.0 if seg <= 1e-9 else max(0.0, min(1.0, ((o.x - ax) * vx + (o.y - ay) * vy) / seg))
            d = math.hypot(o.x - (ax + vx * t), o.y - (ay + vy * t))
            if d < DRIBBLE_LANE_RADIUS:
                worst = max(worst, (DRIBBLE_LANE_RADIUS - d) / DRIBBLE_LANE_RADIUS)
        return worst

    def _carry_value(self, inp: PolicyInput, p: Player, tx: float, ty: float) -> float:
        """What a carry is worth: forward ground won, discounted by the contest.

        This replaced

            carry_value = max(5.0, (OPP_GOAL_X - p.x) * 0.1)   # Progress toward goal

        which was not progress toward goal. (OPP_GOAL_X - p.x) is the distance
        *still to run*, so the value fell as the carrier advanced and the 5.0
        floor then bound for every x >= 10: flat, identical at the halfway line
        and five metres out. The only thing it got right was its comment's
        intent, and the effect was that the lowest-valued candidate in the
        planner was the one the loose-ball physics note above calls the only
        reliable way to keep the ball. A carrier with a clean lane and a
        collectable pass available would hand the ball over anyway, because
        5.0 lost to safe_pass's 10.0 every time.
        """
        gain = tx - p.x
        if gain <= 0.0:
            # Sideways or backwards: still keep the ball, but nothing gained.
            return CARRY_HOLD_VALUE
        progress = geom.clamp(gain / CARRY_FULL_GAIN, 0.0, 1.0)
        safety = 1.0 - self._segment_contest(inp.state, p.x, p.y, tx, ty)
        value = CARRY_HOLD_VALUE + (CARRY_CLEAN_VALUE - CARRY_HOLD_VALUE) * progress * safety
        if OPP_GOAL_X - tx <= CARRY_SHOOT_RANGE:
            value += CARRY_SHOOT_BONUS
        return value

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

        # COORDINATED RECEIVER MOVEMENT:
        # If this player is the intended receiver for a pass, move them toward
        # the predicted collection point so they arrive as the ball becomes controllable.
        if (self._pending_pass_receiver == p.id 
                and self._pending_pass_collection is not None 
                and possessor is not None 
                and possessor.can_act):
            tx, ty = self._pending_pass_collection
            # Clear the pending pass so it doesn't persist beyond this decision cycle
            self._pending_pass_receiver = None
            self._pending_pass_collection = None
            return PlayerIntent(p.id, tx, ty, 1.0, tx, ty, "none")
        
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
        # COORDINATED RECEIVER MOVEMENT:
        # If this player is the intended receiver for a pass that was just played,
        # move them toward the predicted collection point (which matches the
        # loose_ball_meeting_point since the pass target IS the collection point).
        if (self._pending_pass_receiver == p.id 
                and self._pending_pass_collection is not None):
            tx, ty = self._pending_pass_collection
            # Clear the pending pass so it doesn't persist beyond this decision cycle
            self._pending_pass_receiver = None
            self._pending_pass_collection = None
            return PlayerIntent(p.id, tx, ty, 1.0, tx, ty, "none")
        
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
        power = _power_from_speed(speed_for_travel(dist))
        return plan.target_x, plan.target_y, power




def config_uses_slap(inp: PolicyInput) -> bool:
    # Slapping is best when we need to strip a carried ball in a dead play.
    return getattr(inp.config, "risk", 0.45) > 0.3 or inp.world.pressure_on_ball > 0.6


def _box_covered(inp: PolicyInput) -> bool:
    # At least one covered player goal-side of the ball before sliding.
    return len(inp.press_plan.cover) >= 1

