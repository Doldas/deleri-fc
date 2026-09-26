"""SpaceModel — a deterministic spatial control approximation.

For each active grid point it estimates the relative influence of our players
versus the opponents. Used by movement, passing, pressing, MCTS and evaluation.

The model is purely geometric and deterministic; no ML. (AISTRATEGI §15.)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .geom import PITCH_LENGTH, PITCH_WIDTH, distance, distance_sq

# Influence radius (metres) of a single player on the control field.
INFLUENCE = 6.0
INFLUENCE_SQ = INFLUENCE * INFLUENCE


def influence(x: float, y: float, px: float, py: float) -> float:
    d2 = distance_sq(x, y, px, py)
    if d2 >= INFLUENCE_SQ:
        return 0.0
    r = math.sqrt(d2) / INFLUENCE
    return (1.0 - r) ** 2


def control_at(
    x: float,
    y: float,
    ours: list[tuple[float, float]],
    theirs: list[tuple[float, float]],
) -> float:
    """Net control (-1 theirs … +1 ours) at point (x, y)."""
    our = sum(influence(x, y, px, py) for px, py in ours)
    their = sum(influence(x, y, px, py) for px, py in theirs)
    total = our + their
    if total <= 1e-9:
        return 0.0
    return (our - their) / total


@dataclass
class PointInfluence:
    x: float
    y: float
    control: float
    pressure: float  # opponents within influence, weighted


@dataclass
class SpaceModel:
    ours: list[tuple[float, float]] = field(default_factory=list)
    theirs: list[tuple[float, float]] = field(default_factory=list)
    # Sampled control at a hands-off 3 m grid, kept small for speed.
    _grid: list[PointInfluence] = field(default_factory=list)

    def update(self, ours: list[tuple[float, float]], theirs: list[tuple[float, float]]) -> None:
        self.ours = ours
        self.theirs = theirs
        self._grid = []
        step = 6.0
        y = 2.0
        while y < PITCH_WIDTH:
            x = 2.0
            while x < PITCH_LENGTH:
                ctrl = control_at(x, y, ours, theirs)
                pressure = sum(influence(x, y, px, py) for px, py in theirs)
                self._grid.append(PointInfluence(x, y, ctrl, pressure))
                x += step
            y += step

    def control_at(self, x: float, y: float) -> float:
        return control_at(x, y, self.ours, self.theirs)

    def best_forward_point(self, start_x: float, y: float) -> tuple[float, float, float]:
        """Best (x, y, control) sampled point ahead of start_x."""
        best = None
        best_score = -1e9
        for p in self._grid:
            if p.x < start_x + 3.0:
                continue
            score = p.control - distance(p.x, p.y, start_x, y) / PITCH_WIDTH * 0.3
            if score > best_score:
                best_score = score
                best = (p.x, p.y, p.control)
        return best or (start_x + 3.0, y, 0.0)

    def nearest_our_control(self, x: float, y: float) -> tuple[float, float]:
        """Find the closest point with strong our-control (used to aim off-ball
        movement toward space we actually control)."""
        best = None
        best_d = math.inf
        for p in self._grid:
            if p.control < 0.15:
                continue
            d = distance(x, y, p.x, p.y)
            if d < best_d:
                best_d = d
                best = (p.x, p.y)
        return best or (x, y)

    def opposition_weakness_zone(self) -> tuple[float, float] | None:
        """Zone (center) where our team currently has the strongest net control
        in the attacking half — a good place to send support."""
        best = None
        best_c = -1e9
        for p in self._grid:
            if p.x < PITCH_LENGTH * 0.35:
                continue
            if p.control > best_c:
                best_c = p.control
                best = (p.x, p.y)
        return best