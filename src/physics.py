"""Internal physics models for ball flight, passing and shooting.

These model the authoritative engine rules (RULES.md). If a prediction ever
disagrees with the real engine, the engine wins and this module must be fixed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from . import geom
from .geom import (
    GOAL_CENTER_Y,
    GOAL_HALF,
    GOAL_HIGH_Y,
    GOAL_LOW_Y,
    OPP_GOAL_X,
    PITCH_LENGTH,
    PITCH_WIDTH,
    wall_bounce,
)
from .config import (
    BALL_CONTROL_MAX_SPEED,
    BALL_CONTROL_SAFE_SPEED,
    BALL_DECAY,
    KICK_MAX_SPEED,
    KICK_MIN_SPEED,
    MAX_RUN_SPEED,
    kick_speed,
)

_LOG_DECAY = math.log(BALL_DECAY) * 60.0  # log per second


def speed_after(speed0: float, seconds: float) -> float:
    return speed0 * (BALL_DECAY ** (60.0 * seconds))


def distance_travelled(speed0: float, seconds: float) -> float:
    """Closed-form distance travelled by a decaying ball over `seconds`.

    s(t) = v0 * (a^(60t) - 1) / (60 * ln a).
    """
    if seconds <= 0.0:
        return 0.0
    return speed0 * (math.exp(_LOG_DECAY * seconds) - 1.0) / _LOG_DECAY


def flight_time(distance: float, speed0: float) -> float:
    """Time for a decaying ball starting at `speed0` to cover `distance`.

    Returns inf when the ball would never reach that far (small pitch, so rare;
    a 12 m/s ball travels ~48 m in flight, enough to cross the pitch).
    """
    if distance <= 0.0:
        return 0.0
    if speed0 <= 0.0:
        return math.inf
    ratio = 1.0 + _LOG_DECAY * distance / speed0
    if ratio <= 0.0:
        return math.inf
    return math.log(ratio) / _LOG_DECAY


def arrival_speed(distance: float, speed0: float) -> float:
    t = flight_time(distance, speed0)
    if math.isinf(t):
        return 0.0
    return speed_after(speed0, t)


def walk_trajectory(
    ox: float,
    oy: float,
    vx: float,
    vy: float,
    seconds: float,
    step: float = 0.05,
) -> list[tuple[float, float, float, float]]:
    """Walk a decaying ball trajectory (with wall bounces) inside the pitch.

    Returns a list of (x, y, vx, vy) samples every `step` seconds. Used by the
    simulator, replay and physics validation tests.
    """
    x, y = ox, oy
    sample = (x, y, vx, vy)
    out = [sample]
    tick = 1.0 / 60.0
    total = 0.0
    next_sample = step
    while total < seconds:
        x += vx * tick
        y += vy * tick
        hit: str | None = None
        if x <= 0.0:
            if GOAL_LOW_Y <= y <= GOAL_HIGH_Y:
                x = 0.0
                return out  # goal: stop before crossing the line
            x = 0.0
            hit = "left"
        elif x >= PITCH_LENGTH:
            if GOAL_LOW_Y <= y <= GOAL_HIGH_Y:
                x = PITCH_LENGTH
                return out
            x = PITCH_LENGTH
            hit = "right"
        if y <= 0.0:
            y = 0.0
            hit = "bottom"
        elif y >= PITCH_WIDTH:
            y = PITCH_WIDTH
            hit = "top"
        if hit is not None:
            vx, vy = geom.wall_bounce(vx, vy, hit)
        vx *= BALL_DECAY
        vy *= BALL_DECAY
        total += tick
        if total >= next_sample:
            out.append((x, y, vx, vy))
            next_sample += step
    return out


@dataclass
class KickResult:
    time: float
    arrival_x: float
    arrival_y: float
    arrival_speed: float
    intercepted_wall: str | None = None  # 'left'|'right'|'top'|'bottom'


def kick_to(ox: float, oy: float, tx: float, ty: float, power: float) -> KickResult:
    """Straight kick from (ox,oy) to target (tx,ty) with engine decay and
    optional wall bounce. Returns the time to reach the *unbounced* target."""
    speed0 = kick_speed(power)
    dist = geom.distance(ox, oy, tx, ty)
    if dist <= 1e-6:
        return KickResult(0.0, tx, ty, speed0)
    reach_time = flight_time(dist, speed0)
    if not math.isfinite(reach_time):
        # Cannot physically arrive: report a never-arriving flight so callers
        # drop the candidate instead of chasing an inf/NaN position.
        return KickResult(math.inf, tx, ty, 0.0)
    # First wall crossing determines any bounce:
    vx = (tx - ox) / dist * speed0
    vy = (ty - oy) / dist * speed0
    hit_wall, hit_time = _first_wall_time(ox, oy, vx, vy)
    if hit_wall is None or hit_time >= reach_time:
        v = arrival_speed(dist, speed0)
        return KickResult(reach_time, tx, ty, v)
    # Ball bounces before reaching the target: model direct portion, then a
    # reflected straight flight with the same decay continuing.
    remainder = reach_time - hit_time
    if remainder <= 0.0:
        return KickResult(reach_time, tx, ty, 0.0)
    _, rxv, ryv = _reflect_at(wall=hit_wall, vx=vx, vy=vy)
    nx, ny = ox + vx * hit_time, oy + vy * hit_time
    return KickResult(reach_time, nx + rxv * remainder, ny + ryv * remainder, arrival_speed(dist, speed0))


def _first_wall_time(ox: float, oy: float, vx: float, vy: float) -> tuple[str | None, float]:
    times: list[tuple[float, str]] = []
    if vx < 0:
        times.append((-ox / vx, "left"))
    elif vx > 0:
        times.append(((PITCH_LENGTH - ox) / vx, "right"))
    if vy < 0:
        times.append((-oy / vy, "bottom"))
    elif vy > 0:
        times.append(((PITCH_WIDTH - oy) / vy, "top"))
    if not times:
        return None, math.inf
    t, wall = min(times, key=lambda item: item[0])
    return wall, t


def _reflect_at(wall: str, vx: float, vy: float) -> tuple[str, float, float]:
    rvx, rvy = geom.wall_bounce(vx, vy, wall)
    return wall, rvx, rvy


@dataclass
class PassPlan:
    receiver_id: str
    target_x: float
    target_y: float
    power: float
    time: float
    arrival_speed: float
    risk: float  # 0..1 estimated interception probability


def plan_lead_pass(
    ox: float,
    oy: float,
    rx: float,
    ry: float,
    rvx: float,
    rvy: float,
    velocity_weight: float,
    comfort_speed: float = BALL_CONTROL_MAX_SPEED,
) -> PassPlan:
    """Aim a pass so the ball meets a moving receiver.

    Iterate: estimate arrival time assuming the receiver keeps moving, place
    the target at the receiver's expected position, then choose power so the
    arrival speed is at a controllable level. Returns NaN-free numbers.
    """
    t = 0.0
    speed0 = comfort_speed
    power = _power_from_speed(speed0)
    target_x = rx
    target_y = ry
    for _ in range(3):
        # Keep the lead target finite and bounded. A receiver sprinting toward
        # the passer can drive the lead point away from the origin each
        # iteration and diverge to +/-inf, which poisons the emitted pass
        # target and makes the engine discard the intent. Clamping the lead
        # offset keeps every emitted pass target finite and within the pitch.
        lead_x = max(-6.0, min(6.0, rvx * velocity_weight * t))
        lead_y = max(-6.0, min(6.0, rvy * velocity_weight * t))
        target_x = rx + lead_x
        target_y = ry + lead_y
        dist = geom.distance(ox, oy, target_x, target_y)
        if dist <= 1e-6:
            return PassPlan("", rx, ry, 0.5, 0.0, 0.0, 0.0)
        # Pick an initial speed such that the boosted arrival speed reaches the
        # receiver slightly above control speed so it does not die en route but
        # can be controlled on arrival.
        arrive_want = max(comfort_speed, 2.0)
        speed0 = _initial_speed_for_arrival(dist, arrive_want)
        power = _power_from_speed(speed0)
        t = flight_time(dist, speed0)
        if not math.isfinite(t):
            t = 0.0
            target_x = rx
            target_y = ry
    if not math.isfinite(target_x) or not math.isfinite(target_y):
        target_x, target_y = rx, ry
        power = 0.5
        t = 0.0
    arrival = speed_after(speed0, t) if t < math.inf else 0.0
    return PassPlan("", target_x, target_y, power, t, arrival, 0.0)


def _initial_speed_for_arrival(dist: float, arrive: float) -> float:
    """v0 such that a decaying ball arrives at speed `arrive` over `dist`."""
    # arrival = v0 * (BALL_DECAY^(60 t)) and dist = v0 * (a^(60t)-1)/(60 ln a)
    # => dist = arrival * (a^(60t) - 1)/(60 ln a * a^(60t))
    # Solve numerically with a few fixed-point iterations instead of closed form.
    v0 = arrive
    for _ in range(8):
        t = flight_time(dist, v0)
        arrival = speed_after(v0, t)
        v0 *= arrive / max(1e-6, arrival) if arrival > 0 else 1.0
    return max(v0, arrive)


def _power_from_speed(speed0: float) -> float:
    p = (speed0 - 12.0) / 14.0
    return 0.0 if p < 0.0 else 1.0 if p > 1.0 else p


# --- Pass collection geometry (RULES.md "Loose balls" is authoritative) -----
# A pass is released at KICK_MIN_SPEED (12 m/s) or more and loses 0.8% of its
# speed every 60 Hz tick. It cannot be touched again until it has slowed to
# BALL_CONTROL_MAX_SPEED. There is no way to kick it slower, so *every* pass
# rolls at least this far before anybody can collect it -- and any pass aimed
# at a receiver closer than this sails straight over their head.
def _travel_before_control(speed: float) -> float:
    """Metres a ball kicked at `speed` rolls before it is collectable."""
    if speed <= BALL_CONTROL_SAFE_SPEED:
        return 0.0
    ticks = math.log(BALL_CONTROL_SAFE_SPEED / speed) / math.log(BALL_DECAY)
    return speed * (1.0 - BALL_DECAY ** ticks) / (1.0 - BALL_DECAY) / 60.0


MIN_PASS_TRAVEL = _travel_before_control(KICK_MIN_SPEED)
# Relative slack for comparisons against a derived threshold. A pass aimed at
# exactly MIN_PASS_TRAVEL metres is legal, and the mirrored picture is the same
# real distance computed from `40 - y` coordinates -- which `hypot` can land one
# or two ULP below the original. A bare `<` then vetoed the pass in one state
# and allowed it in its mirror. Anything within this relative band counts as
# "on" the threshold, so decisions follow the geometry instead of the rounding
# of the coordinates they were handed.
THRESHOLD_REL_EPS = 1e-9


def at_or_below(value: float, threshold: float) -> bool:
    """`value <= threshold`, tolerant of floating-point noise at the boundary."""
    return value <= threshold + abs(threshold) * THRESHOLD_REL_EPS


def at_or_above(value: float, threshold: float) -> bool:
    """`value >= threshold`, tolerant of floating-point noise at the boundary."""
    return value >= threshold - abs(threshold) * THRESHOLD_REL_EPS

# The ball leaves the kicker's control CONTROLLED_BALL_AHEAD metres in front of
# them, so the useful band of pass distances is narrow: shorter overshoots the
# receiver entirely, longer is a long ball nobody can collect.
MIN_USEFUL_PASS = MIN_PASS_TRAVEL - 1.5
MAX_USEFUL_PASS = MIN_PASS_TRAVEL + 6.0


def pass_collection_point(
    ox: float, oy: float, aim_x: float, aim_y: float, power: float
) -> tuple[float, float, float, float]:
    """Where a pass aimed from (ox,oy) toward (aim_x,aim_y) first stops.

    Returns (collect_x, collect_y, power, speed0). The ball is kicked at
    `kick_speed(power)` along the aim direction, and `collect_*` is where it
    first drops to a controllable speed. If the aim point is closer than the
    ball's minimum travel the ball overshoots it, and this function reports the
    overshoot point so the caller can either aim there or refuse the pass.
    The result is clamped onto the pitch: a collection point in the stands is
    never a legal pass target.
    """
    dx = aim_x - ox
    dy = aim_y - oy
    dist = math.hypot(dx, dy)
    if dist <= 1e-6:
        return ox, oy, 0.0, KICK_MIN_SPEED
    ux, uy = dx / dist, dy / dist
    speed0 = kick_speed(power)
    travel = _travel_before_control(speed0)
    cx = geom.clamp(ox + ux * travel, 0.0, PITCH_LENGTH)
    cy = geom.clamp(oy + uy * travel, 0.0, PITCH_WIDTH)
    return cx, cy, power, speed0


def speed_for_travel(dist: float) -> float:
    """Lowest kick speed whose ball comes to rest `dist` metres away.

    Returns KICK_MIN_SPEED when `dist` is below the unavoidable minimum travel,
    because no slower kick exists. Callers use the shortfall to decide that a
    short pass is a turnover waiting to happen rather than a pass.
    """
    if dist <= MIN_PASS_TRAVEL:
        return KICK_MIN_SPEED
    lo, hi = KICK_MIN_SPEED, KICK_MAX_SPEED
    for _ in range(20):
        mid = (lo + hi) / 2.0
        if _travel_before_control(mid) < dist:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


@dataclass
class ShotTarget:
    x: float
    y: float
    power: float
    quality: float  # 0..1 expected scoring quality accounting for GK coverage


def pick_shot_target(bx: float, by: float, gkx: float, gky: float, power: float = 0.95) -> ShotTarget:
    """Choose the goal corner that minimizes goalkeeper lateral reach.

    Targets are inset slightly inside the posts: aiming exactly at the
    goal-line posts (GOAL_LOW_Y / GOAL_HIGH_Y) makes half the attempts hit the
    frame and rebound, so targets sit `_POST_INSET` metres inside the goal.
    """
    base = OPP_GOAL_X
    post_inset = 0.6
    low = (base, GOAL_LOW_Y + post_inset)
    high = (base, GOAL_HIGH_Y - post_inset)
    d_low = geom.distance(gkx, gky, low[0], low[1])
    d_high = geom.distance(gkx, gky, high[0], high[1])
    # Far corner relative to the shooter gives the worst GK angle. Which corner
    # counts as "far" has to come from the sign of the shooter's lateral offset:
    # under the lateral mirror a one-sided test like `by >= 20.0` becomes
    # `by <= 20.0`, so a shooter on one flank and its mirror aimed at the *same*
    # post rather than opposite ones.
    #
    # A shooter exactly on the centre line is the one case that needs no rule:
    # y = GOAL_CENTER_Y is its own mirror image, so that state is identical to
    # its own reflection and any deterministic choice is already equivariant.
    # Adding a "far" candidate there would only invent a preference the picture
    # does not support, so the two measured posts are left to compete.
    offset = by - GOAL_CENTER_Y
    if offset > 0.0:
        far = (base, GOAL_CENTER_Y - (GOAL_HALF - post_inset))
    elif offset < 0.0:
        far = (base, GOAL_CENTER_Y + (GOAL_HALF - post_inset))
    else:
        far = None
    if far is None:
        candidates = [("low", low, d_low), ("high", high, d_high)]
    else:
        candidates = [
            ("low", low, d_low),
            ("high", high, d_high),
            ("far", far, geom.distance(gkx, gky, far[0], far[1])),
        ]
    # Largest keeper reach wins, and an exact tie is broken towards the far
    # post, which is the point of aiming away from the keeper. Both keys mirror
    # with the pitch. `max` used to keep the first of three equidistant
    # candidates and the far-post identity came from a one-sided comparison, so
    # a state and its mirror could pick the same post for opposite reasons.
    # Any remaining tie is between two candidates sharing a y, because the keys
    # are functions of the post height and the shooter's own height, so the
    # returned target is still unique.
    _, target, reach = max(
        candidates,
        key=lambda c: (
            c[2],
            1.0 if c[0] == "far" else 0.0,
            -abs(by - c[1][1]),
        ),
    )
    quality = clamp01(1.0 - (reach / 3.0) * 0.8)
    return ShotTarget(target[0], target[1], power, quality)


def clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v


# RULES.md "Goalkeeper handling and diving": a save is a *standing catch* when
# the shot needs under 0.8 m of lateral reach and a *dive* from 0.8 m through
# 1.65 m. Past 1.65 m of lateral reach the keeper cannot get there at all, and
# the ball is a goal. That 1.65 m is the single number that decides whether a
# shot is worth taking, so it lives here with the rest of the shot model.
GK_LATERAL_REACH = 1.65
GK_DIVE_LATERAL_MIN = 0.8
# A goalkeeper auto-handles only inside his own defensive fifth, the last fifth
# of the pitch measured from the goal he defends. In our normalised attacking
# frame that goal is at x = PITCH_LENGTH, so the area starts here.
GK_AREA_START = 0.8 * PITCH_LENGTH


def shot_beats_keeper(
    bx: float,
    by: float,
    gkx: float,
    gky: float,
    target_y: float,
    dist_goal: float,
    power: float = 0.92,
) -> bool:
    """Is there any realistic chance this shot beats the keeper?

    A dive covers at most GK_LATERAL_REACH (1.65 m) of lateral reach, and under
    that a save is an easy standing catch. The keeper is not a statue, though:
    while the ball is in flight he keeps shifting across his goal at run speed.
    So the number that decides a shot is not the raw angle but the angle
    *left over* after he has had time to move:

        flight  = dist_goal / kick speed          (RULES.md: 12..26 m/s)
        shift   = MAX_RUN_SPEED * flight          (how far he can get across)
        left    = |gky - target_y| - shift        (gap he still cannot reach)

    * `left <= GK_DIVE_LATERAL_MIN` -> standing catch, the shot is wasted.
    * `left >= GK_LATERAL_REACH`    -> out of dive range, it is a goal.

    This reproduces the measured behaviour against Vanguard FC (elite), where
    35 shots from a mean 15.1 m produced zero goals: our shooters sat on the
    centre line, so the far corner was only 2.4 m from a keeper who had
    several seconds of flight time to cover it. The lever that actually works
    is *angle*, not range -- a wide shooter can put the target 12 m+ from the
    keeper and score from anywhere.
    """
    if gkx < 0.0:
        return False  # unknown keeper: do not assume a free goal
    # Keeper dragged out of his own defensive fifth: he cannot catch at all.
    if gkx < GK_AREA_START:
        return True
    speed0 = kick_speed(power)
    flight = max(dist_goal, 0.0) / max(speed0, 1.0)
    shift = MAX_RUN_SPEED * flight
    left = abs(gky - target_y) - shift
    if left <= GK_DIVE_LATERAL_MIN:
        return False
    if left >= GK_LATERAL_REACH:
        return True
    # In between: a dive is a stretch. Commit when the flight is short, i.e.
    # the shot was taken from close range and he is already going to ground.
    return dist_goal <= 9.0


def shot_open_angle(bx: float, by: float, opponents: list[tuple[float, float]]) -> float:
    """Total angular size (radians) of the goalmouth not covered by opponents
    between the ball and the goal. Used for shot worthiness."""
    def bearing(p: tuple[float, float]) -> float:
        return math.atan2(p[1] - by, p[0] - bx)

    low = bearing((OPP_GOAL_X, GOAL_LOW_Y))
    high = bearing((OPP_GOAL_X, GOAL_HIGH_Y))
    # Order the goalmouth as a continuous angular interval.
    if low > high:
        low, high = high, low
    blockers = sorted(math.atan2(oy - by, ox - bx) for (ox, oy) in opponents if ox > bx)
    if not blockers:
        return high - low
    free = [segment for segment in _free_segments(low, high, blockers)]
    return max((s[1] - s[0] for s in free), default=0.0)


def _free_segments(a: float, b: float, blockers: list[float], pad: float = 0.12):
    """Expose free angular segments of [a, b] given blocker bearings."""
    current = a
    for blk in blockers:
        if blk <= current:
            continue
        if blk >= b:
            break
        yield (current, blk - pad)
        current = blk + pad
    yield (current, b)


def shot_lane_clear(
    bx: float,
    by: float,
    tx: float,
    ty: float,
    opponents: list[tuple[float, float]],
    margin: float = 0.14,
) -> bool:
    """True when the straight lane to the shot target is free of opponents.

    Unlike `shot_open_angle` (which demands a large fraction of the whole
    goalmouth be open), this checks only the angular corridor toward the
    chosen corner, so a shot fires whenever the aimed lane is clear — the
    trigger that converts boxes worth of dribbling into real shots.
    """
    if not opponents:
        return True
    target_bearing = math.atan2(ty - by, tx - bx)
    for ox, oy in opponents:
        # Only blockers between the ball and the target can block the lane.
        if ox <= bx or ox > tx + 1e-6:
            continue
        bearing = math.atan2(oy - by, ox - bx)
        while bearing < target_bearing - math.pi:
            bearing += 2.0 * math.pi
        while bearing > target_bearing + math.pi:
            bearing -= 2.0 * math.pi
        if abs(bearing - target_bearing) < margin:
            return False
    return True


def pass_lane_clear(
    bx: float,
    by: float,
    tx: float,
    ty: float,
    opponents: list[tuple[float, float]],
    margin: float = 0.12,
) -> bool:
    """True when the passing lane is free. Uses same logic as shot_lane_clear."""
    if not opponents:
        return True
    target_bearing = math.atan2(ty - by, tx - bx)
    for ox, oy in opponents:
        if ox <= bx or ox > tx + 1e-6:
            continue
        bearing = math.atan2(oy - by, ox - bx)
        while bearing < target_bearing - math.pi:
            bearing += 2.0 * math.pi
        while bearing > target_bearing + math.pi:
            bearing -= 2.0 * math.pi
        if abs(bearing - target_bearing) < margin:
            return False
    return True


def wall_shot_target(
    bx: float,
    by: float,
    gkx: float,
    gky: float,
    power: float = 0.95,
) -> tuple[float, float, float]:
    """Pick a wall-bounce shot target (contact point on wall, goal target, power)."""
    from .geom import GOAL_HIGH_Y, GOAL_LOW_Y, PITCH_LENGTH, PITCH_WIDTH, wall_bounce

    # Try both side walls for a shot that bounces into the far corner
    post_inset = 0.6
    base = PITCH_LENGTH
    # Same one-sided-comparison trap as `pick_shot_target`: `by < 20.0` becomes
    # `by > 20.0` under the lateral mirror, so a shooter and its mirror aimed at
    # the same post. Take the far post from the sign of the offset; exactly on
    # the centre line the state is its own mirror, so leave the old convention
    # rather than invent a preference the picture does not support.
    offset = by - GOAL_CENTER_Y
    if offset > 0.0:
        far_y = GOAL_LOW_Y + post_inset
    elif offset < 0.0:
        far_y = GOAL_HIGH_Y - post_inset
    else:
        far_y = GOAL_LOW_Y + post_inset

    best = None
    best_key = None
    tied: list[tuple[float, float]] = []
    for wall_side in ("left", "right"):
        # Mirror the goal target across the wall
        if wall_side == "left":
            contact_x = 0.0
            mirrored_goal_x = -PITCH_LENGTH
        else:
            contact_x = PITCH_LENGTH
            mirrored_goal_x = 2 * PITCH_LENGTH

        # Estimate a contact point that gives good angle
        for contact_y in [6.0, 12.0, 28.0, 34.0]:
            # Check if lane from ball to contact is clear
            opponents_pos = [(gkx, gky)]
            if not pass_lane_clear(bx, by, contact_x, contact_y, opponents_pos, margin=0.15):
                continue
            # Check lane from contact to goal
            if not pass_lane_clear(contact_x, contact_y, base, far_y, opponents_pos, margin=0.15):
                continue

            # Score based on GK coverage of the bounce
            bounce_vx = (base - contact_x) / geom.distance(contact_x, contact_y, base, far_y)
            bounce_vy = (far_y - contact_y) / geom.distance(contact_x, contact_y, base, far_y)
            d = geom.distance(gkx, gky, base, far_y)
            score = d * 0.5
            # The candidate contact heights are mirror-symmetric about the
            # centre line and the keeper's distance to the chosen far post is
            # unchanged by the mirror, so the common case ties outright. A fixed
            # scan order then sent a shooter and its mirror to the same wall
            # height. Rank on the shooter's own lateral position, which mirrors,
            # and average anything still tied -- the mirror of a mean is the mean
            # of the mirrors, so that resolves exactly rather than by convention.
            key = (score, -abs(by - contact_y))
            if best_key is None or key > best_key:
                best_key = key
                tied = [(contact_x, contact_y)]
            elif key == best_key:
                tied.append((contact_x, contact_y))

    if tied:
        best = (
            sum(cx for cx, _ in tied) / len(tied),
            sum(cy for _, cy in tied) / len(tied),
            power,
        )
    
    if best:
        return best
    # Fallback: direct shot
    shot = pick_shot_target(bx, by, gkx, gky, power)
    return (shot.x, shot.y, shot.power)


def wall_pass_target(
    bx: float,
    by: float,
    rx: float,
    ry: float,
) -> tuple[float, float, float] | None:
    """Find a wall-bounce pass contact point to reach receiver.
    
    Returns (contact_x, contact_y, power) or None if no good wall pass."""
    from .geom import PITCH_LENGTH, PITCH_WIDTH, wall_bounce
    
    best = None
    best_score = -1e9
    for wall_side in ("left", "right"):
        if wall_side == "left":
            contact_x = 0.0
        else:
            contact_x = PITCH_LENGTH
        
        # Try multiple contact points on the wall
        for contact_y in [4.0, 8.0, 12.0, 16.0, 24.0, 28.0, 32.0, 36.0]:
            # Wall must be between ball and receiver (or near receiver)
            # Actually we want the bounce to reach receiver
            # Mirror receiver across wall
            if wall_side == "left":
                mirrored_rx = -rx
            else:
                mirrored_rx = 2 * PITCH_LENGTH - rx
            
            # Line from ball to mirrored receiver hits wall at contact point
            # Solve for intersection
            dx = mirrored_rx - bx
            dy = ry - by
            if abs(dx) < 1e-6:
                continue
            t = (contact_x - bx) / dx
            if t <= 0 or t >= 1:
                continue
            calc_y = by + dy * t
            if abs(calc_y - contact_y) > 2.0:
                continue
            
            # Good wall pass candidate
            score = -geom.distance(bx, by, contact_x, contact_y)  # Prefer shorter
            if score > best_score:
                best_score = score
                dist = geom.distance(bx, by, contact_x, contact_y)
                power = min(1.0, max(0.0, dist / 33.3))
                best = (contact_x, contact_y, power)
    
    return best