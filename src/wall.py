"""WallModel — the wall as a tactical resource (AISTRATEGI §3, §11, §17, §18).

Walls retain tangential speed and return 75% of perpendicular speed. This model
predicts contact points, rebounds and, crucially, whether a wall pass/shot can
reach a receiver with a better risk/return trade than a direct kick.

All methods are deterministic. Wall plays are hypotheses; the runtime policy
only uses them when they score higher than direct options.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from . import geom
from .geom import PITCH_LENGTH, PITCH_WIDTH


@dataclass
class WallCandidate:
    contact_x: float
    contact_y: float
    rebound_x: float
    rebound_y: float
    target_flag: str  # 'top'|'bottom'|'left'|'right'
    progression: float  # net +x gained by the intended receiver
    risk: float  # 0..1 interception estimate
    value: float  # combined score for the policy


@dataclass
class WallModel:
    # Restitution constants copied from engine rules.
    tangential: float = 1.0
    perpendicular: float = 0.75

    def contact_for(
        self,
        ox: float,
        oy: float,
        receiver_x: float,
        receiver_y: float,
    ) -> WallCandidate | None:
        """Compute a wall contact point (touchline only) such that a kick from
        (ox, oy) rebounds toward the receiver.

        Uses the mirror trick across the touchline, then accounts for the 75%
        perpendicular restitution by pulling the reflected target closer.
        """
        best: WallCandidate | None = None
        for flag in ("top", "bottom"):
            wall_y = PITCH_WIDTH if flag == "top" else 0.0
            # Mirror receiver across the touchline.
            mrx = receiver_x
            mry = 2.0 * wall_y - receiver_y
            # Intersection of line (o -> mirrored receiver) with the line y = wall_y.
            if abs(mry - oy) < 1e-9:
                continue
            t = (wall_y - oy) / (mry - oy)
            if t <= 0.0 or t >= 1.0:
                continue
            cx = ox + (mrx - ox) * t
            cy = wall_y
            if cx <= 0.5 or cx >= PITCH_LENGTH - 0.5:
                continue
            # With 75% restitution the outgoing perpendicular speed is weaker,
            # so the real landing point is closer to the wall than the mirror
            # suggests. Adjust by treating the wall as a soft reflector.
            # Real rebound lands roughly at receiver_y pulled toward the wall:
            reb_y = wall_y + 0.75 * (receiver_y - wall_y)
            # Incoming leg length to the wall:
            d_in = geom.distance(ox, oy, cx, cy)
            # Outgoing leg (scaled by the speed ratio) to the estimated landing:
            # because outgoing speed is ~0.75 of incoming normal component, the
            # outgoing distance is approximated by a compression factor.
            out_dist = geom.distance(cx, cy, receiver_x, receiver_y) * 0.75
            progression = receiver_x - ox
            risk = self._interception_risk(ox, oy, cx, cy, out_dist, flag)
            value = progression - risk * 60.0
            cand = WallCandidate(cx, cy, receiver_x, reb_y, flag, progression, risk, value)
            if best is None or cand.value > best.value:
                best = cand
        return best

    def _interception_risk(
        self,
        ox: float,
        oy: float,
        cx: float,
        cy: float,
        out_dist: float,
        flag: str,
        lane_width: float = 0.9,
    ) -> float:
        """Crude risk: how close to the touchline the whole play stays. The
        closer to the wall, the fewer opponents can legally be in the lane."""
        wall_y = PITCH_WIDTH if flag == "top" else 0.0
        max_lat = max(abs(oy - wall_y), abs(cy - wall_y))
        # Central lanes are naturally risky; hugging the wall is safer.
        if max_lat <= 0.0:
            return 0.05
        central_factor = (20.0 - abs(cy - 20.0)) / 20.0
        risk = 0.15 + central_factor * 0.5 + out_dist / PITCH_LENGTH * 0.2
        return min(1.0, risk)

    def wall_shots(self, bx: float, by: float) -> list[WallCandidate]:
        """Candidate wall-assisted shots: rebound off a touchline toward the
        goal. The wall can create a wider angle than a direct shot when the
        possessor is crowded centrally.

        We evaluate the ‘give it to the wall and attack the rebound’ pattern by
        producing a target whose rebound crosses the goalmouth at an awkward
        height for the goalkeeper.
        """
        candidates: list[WallCandidate] = []
        for flag in ("top", "bottom"):
            wall_y = PITCH_WIDTH if flag == "top" else 0.0
            for cx in (bx * 0.5 + 15.0, bx * 0.5 + 30.0):
                cy = wall_y
                if 0.5 <= cx <= PITCH_LENGTH - 0.5:
                    ba = math.atan2(cy - by, cx - bx)
                    # Reflected direction toward goal center.
                    if flag == "top":
                        nydir = -0.75
                    else:
                        nydir = 0.75
                    nx = 1.0
                    progression = PITCH_LENGTH - bx
                    risk = 0.35 + (20.0 - abs(cy - 20.0)) / 20.0 * 0.4
                    value = progression * 0.5 - risk * 40.0
                    candidates.append(WallCandidate(cx, cy, 0.0, 0.0, flag, progression, risk, value))
        return candidates

    def predict_rebound(self, ox: float, oy: float, vx: float, vy: float) -> tuple[float, float, float, float, str | None]:
        """First wall contact point + post-bounce velocity (engine-accurate).

        Returns (cx, cy, rvx, rvy, wall) or the far target if no wall is hit.
        """
        # Use ray-wall intersection (touchlines + goal lines).
        hits: list[tuple[float, str]] = []
        if vx < 0:
            hits.append((-ox / vx, "left"))
        elif vx > 0:
            hits.append(((PITCH_LENGTH - ox) / vx, "right"))
        if vy < 0:
            hits.append((-oy / vy, "bottom"))
        elif vy > 0:
            hits.append(((PITCH_WIDTH - oy) / vy, "top"))
        if not hits:
            return ox + vx, oy + vy, vx, vy, None
        t, wall = min(hits, key=lambda h: h[0])
        cx = ox + vx * t
        cy = oy + vy * t
        rvx, rvy = geom.wall_bounce(vx, vy, wall)
        # If it crossed a goal opening, it is a goal, not a rebound.
        if wall in ("left", "right") and 17.0 <= cy <= 23.0:
            return cx, cy, 0.0, 0.0, "goal"
        return cx, cy, rvx, rvy, wall


def is_near_wall(x: float, y: float, margin: float = 5.0) -> bool:
    return x <= margin or x >= PITCH_LENGTH - margin or y <= margin or y >= PITCH_WIDTH - margin


def direction_to_wall(x: float, y: float) -> tuple[float, float]:
    """Unit-normalised suggestion (dx, dy) toward the nearest touchline."""
    to_bottom = y
    to_top = PITCH_WIDTH - y
    to_left = x
    to_right = PITCH_LENGTH - x
    best = min((to_bottom, "bottom"), (to_top, "top"), (to_left, "left"), (to_right, "right"), key=lambda p: p[0])
    if best[1] == "bottom":
        return (0.0, -1.0)
    if best[1] == "top":
        return (0.0, 1.0)
    if best[1] == "left":
        return (-1.0, 0.0)
    return (1.0, 0.0)