"""Geometry primitives for the Babylon pitch.

Everything here is deterministic and pure. Coordinates are metres in the
normalized frame where we always attack toward positive X.
"""

from __future__ import annotations

import math
from typing import Iterator

PITCH_LENGTH = 60.0
PITCH_WIDTH = 40.0
GOAL_WIDTH = 6.0
GOAL_CENTER_Y = 20.0
GOAL_HALF = GOAL_WIDTH / 2.0
GOAL_LOW_Y = GOAL_CENTER_Y - GOAL_HALF
GOAL_HIGH_Y = GOAL_CENTER_Y + GOAL_HALF

# Own goal line / opponent goal line (we attack toward +x).
OWN_GOAL_X = 0.0
OPP_GOAL_X = PITCH_LENGTH

# Defensive fifth: first 20% of the pitch from our own goal line.
DEFENSIVE_FIFTH = PITCH_LENGTH * 0.2


def clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


def clamp_point(x: float, y: float, margin: float = 0.0) -> tuple[float, float]:
    return (
        clamp(x, margin, PITCH_LENGTH - margin),
        clamp(y, margin, PITCH_WIDTH - margin),
    )


def distance(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


def distance_sq(ax: float, ay: float, bx: float, by: float) -> float:
    dx = ax - bx
    dy = ay - by
    return dx * dx + dy * dy


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def lerp_point(
    ax: float, ay: float, bx: float, by: float, t: float
) -> tuple[float, float]:
    return (lerp(ax, bx, t), lerp(ay, by, t))


def angle_to(ax: float, ay: float, bx: float, by: float) -> float:
    return math.atan2(by - ay, bx - ax)


def facing_diff(face: float, tx: float, ty: float, px: float, py: float) -> float:
    """Smallest absolute angular difference between `face` and the bearing
    toward (tx, ty) from (px, py)."""
    target = angle_to(px, py, tx, ty)
    diff = (target - face + math.pi) % (2.0 * math.pi) - math.pi
    return abs(diff)


def rotate(vx: float, vy: float, radians: float) -> tuple[float, float]:
    c = math.cos(radians)
    s = math.sin(radians)
    return vx * c - vy * s, vx * s + vy * c


def dot(ax: float, ay: float, bx: float, by: float) -> float:
    return ax * bx + ay * by


def seg_point_distance_sq(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> float:
    """Squared distance from point p to segment a-b."""
    vx = bx - ax
    vy = by - ay
    wx = px - ax
    wy = py - ay
    if vx == 0.0 and vy == 0.0:
        return distance_sq(px, py, ax, ay)
    t = clamp(dot(wx, wy, vx, vy) / (vx * vx + vy * vy), 0.0, 1.0)
    return distance_sq(px, py, ax + vx * t, ay + vy * t)


def segment_intersection_time(
    ox: float,
    oy: float,
    vx: float,
    vy: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> float | None:
    """Time t>0 when the ray (o, v) crosses segment [a,b]; None if it never
    does. Returns the smaller positive time."""
    rx = bx - ax
    ry = by - ay
    denom = vx * ry - vy * rx
    if denom == 0.0:
        return None
    qx = ax - ox
    qy = ay - oy
    t = (qx * ry - qy * rx) / denom
    s = (qx * vy - qy * vx) / denom
    if t < 0.0 or s < 1e-9 or s > 1.0 - 1e-9:
        return None
    return t


def wall_bounce(vx: float, vy: float, wall: str) -> tuple[float, float]:
    """Reflect a velocity against a pitch boundary.

    Walls retain tangential speed and return 75% of the perpendicular speed
    (RULES.md / AISTRATEGI §11). `wall` is one of 'left','right','top','bottom'
    where left/right are our/opponent goal lines and top/bottom are touchlines.
    """
    if wall in ("left", "right"):
        return (-0.75 * vx, vy)
    if wall in ("top", "bottom"):
        return (vx, -0.75 * vy)
    return (vx, vy)


def iter_points_in(a: float, b: float) -> Iterator[float]:
    for i in range(int(a), int(b) + 1):
        yield float(i)