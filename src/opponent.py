"""OpponentModel — lightweight online opponent tracking (AISTRATEGI §38).

Tracks cheap features from the observation stream and never does expensive
learning at runtime. Every update is deterministic and O(players).

Detects Elite opponent archetypes for adaptive counter-tactics.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import geom
from .state import GameState
from .wall import is_near_wall


@dataclass
class OpponentModel:
    # Preferred progression side: accumulate signed y-bias of their ball carries.
    side_bias: float = 0.0
    samples: int = 0
    # Press intensity while we possess: average opponent proximity to our ball.
    press_samples: float = 0.0
    press_accum: float = 0.0
    # Average defensive shape: centroid x of their outfield while we possess.
    shape_x_accum: float = 0.0
    shape_samples: int = 0
    # Wall usage: how often they move toward a touchline while possessing.
    wall_freq_accum: float = 0.0
    wall_samples: int = 0
    # Transition behaviour: mean ball speed right after a possession change.
    transition_accum: float = 0.0
    transition_samples: int = 0
    # Shooting behaviour: how far out they shoot.
    shot_dist: float | None = None
    # Running window raw features used by the runtime policy.
    last_their_possessor_speed: float = 0.0
    
    # --- Elite style detection ---
    # Track patterns to identify opponent archetype
    high_press_ticks: int = 0          # Ticks they press high (>40m)
    low_block_ticks: int = 0           # Ticks they sit deep (<25m defensive line)
    counter_attack_ticks: int = 0      # Ticks they counter fast
    direct_play_ticks: int = 0         # Ticks they play long direct
    possession_ticks: int = 0          # Ticks they keep ball
    total_obs_ticks: int = 0
    last_possession_change: int = 0    # Track possession changes
    prev_possessing_team: str | None = None

    def update(self, state: GameState) -> None:
        ball = state.ball
        self.total_obs_ticks += 1
        
        # Track possession changes for transition analysis
        if self.prev_possessing_team is not None and ball.possessing_team != self.prev_possessing_team:
            self.last_possession_change = self.total_obs_ticks
        self.prev_possessing_team = ball.possessing_team
        
        if ball.possessing_team == "them":
            p = state.their_possessor()
            if p is not None:
                self.side_bias += (1.0 if p.y >= 20.0 else -1.0) * (abs(p.y - 20.0) / 20.0)
                self.samples += 1
                self.last_their_possessor_speed = max(abs(p.vx), abs(p.vy))
                if is_near_wall(p.x, p.y, margin=3.5):
                    self.wall_freq_accum += 1.0
                self.wall_samples += 1
                
                # Detect direct play: long forward passes from deep
                if p.x < 20.0 and ball.vx > 10.0:
                    self.direct_play_ticks += 1
                
                # Detect counter-attack: fast transition after winning ball
                if self.last_possession_change > 0 and self.total_obs_ticks - self.last_possession_change < 10:
                    if p.vx > 4.0:
                        self.counter_attack_ticks += 1
                
        elif ball.possessing_team == "us":
            opp = state.outfield_them()
            if opp:
                d = min(geom.distance(ball.x, ball.y, q.x, q.y) for q in opp)
                self.press_accum += d
                self.press_samples += 1
                cx = sum(q.x for q in state.outfield_them()) / max(1, len(state.outfield_them()))
                self.shape_x_accum += cx
                self.shape_samples += 1
                
                # Detect high press: their players in our half
                high_pressers = sum(1 for q in opp if q.x < 30.0)
                if high_pressers >= 3:
                    self.high_press_ticks += 1
                
                # Detect low block: deep defensive line
                if cx > 40.0:
                    self.low_block_ticks += 1
        
        # Track possession style when they have ball
        if ball.possessing_team == "them":
            self.possession_ticks += 1

    # -- derived queries ------------------------------------------------------
    def prefer_side(self) -> float:
        """Larger and positive → they like attacking on their right (y high)."""
        return self.side_bias / self.samples if self.samples else 0.0

    def press_intensity(self) -> float:
        """Inverted average proximity: high when they swarm us."""
        if not self.press_samples:
            return 0.5
        avg = self.press_accum / self.press_samples
        return 1.0 - min(1.0, max(0.0, (avg - 2.0) / 8.0))

    def defensive_shape_height(self) -> float:
        if not self.shape_samples:
            return 30.0
        return self.shape_x_accum / self.shape_samples

    def wall_usage(self) -> float:
        if not self.wall_samples:
            return 0.0
        return self.wall_freq_accum / self.wall_samples

    def summary(self) -> dict:
        return {
            "side_bias": round(self.prefer_side(), 3),
            "press_intensity": round(self.press_intensity(), 3),
            "shape_height": round(self.defensive_shape_height(), 2),
            "wall_usage": round(self.wall_usage(), 3),
        }

    # --- Elite archetype detection ---
    def detect_archetype(self) -> str:
        """Detect the opponent's tactical archetype.
        
        Returns one of: 'high_press', 'low_block', 'counter', 'direct', 
        'possession', 'wall', 'chaos', 'balanced'
        """
        if self.total_obs_ticks < 20:
            return "unknown"
        
        total = max(1, self.total_obs_ticks)
        high_press_ratio = self.high_press_ticks / total
        low_block_ratio = self.low_block_ticks / total
        counter_ratio = self.counter_attack_ticks / max(1, self.total_obs_ticks)
        direct_ratio = self.direct_play_ticks / max(1, self.samples)
        possession_ratio = self.possession_ticks / max(1, self.samples)
        
        # High press: sustained pressure in our half
        if high_press_ratio > 0.4:
            return "high_press"
        
        # Low block: sitting deep consistently
        if low_block_ratio > 0.5:
            return "low_block"
        
        # Counter-attack: frequent fast breaks
        if counter_ratio > 0.15:
            return "counter"
        
        # Direct play: long balls from deep
        if direct_ratio > 0.25:
            return "direct"
        
        # Possession: keep ball, build slowly
        if possession_ratio > 0.6 and self.press_intensity() < 0.4:
            return "possession"
        
        # Wall play: frequent wall usage
        if self.wall_usage() > 0.3:
            return "wall"
        
        return "balanced"

    def get_counter_tactics(self) -> dict:
        """Return specific counter-tactics for the detected archetype."""
        archetype = self.detect_archetype()
        
        counters = {
            "high_press": {
                "description": "Play through the press, quick one-twos, exploit space behind",
                "press_trigger_threshold": 0.6,   # Don't press high ourselves
                "passing_risk": 0.3,              # Safe, short passes
                "verticality": 0.4,               # Play through, not over
                "width": 1.0,                     # Use full width to stretch
                "support_distance": 5.0,          # Close support for combinations
                "counterpress_intensity": 0.7,    # Counter-press when we win it
                "transition_speed": 0.8,          # Fast transitions
            },
            "low_block": {
                "description": "Patient build-up, width to stretch, crosses, cut-backs",
                "press_trigger_threshold": 0.3,   # Don't press high
                "passing_risk": 0.5,              # Mix of safe and risky
                "verticality": 0.7,               # Push forward
                "width": 1.0,                     # Maximum width
                "support_distance": 8.0,          # Support in pockets
                "counterpress_intensity": 0.3,    # Don't overcommit
                "transition_speed": 0.4,          # Patient
            },
            "counter": {
                "description": "Rest defense, control transitions, don't overcommit",
                "press_trigger_threshold": 0.5,
                "passing_risk": 0.4,
                "verticality": 0.6,
                "width": 0.8,
                "support_distance": 7.0,
                "counterpress_intensity": 0.4,
                "defensive_line": 22.0,           # Deeper rest defense
                "transition_speed": 0.5,
            },
            "direct": {
                "description": "Win first contact, control second balls, press high",
                "press_trigger_threshold": 0.35,
                "passing_risk": 0.3,
                "verticality": 0.5,
                "width": 0.9,
                "support_distance": 6.0,
                "counterpress_intensity": 0.8,
                "defensive_line": 24.0,
            },
            "possession": {
                "description": "Press to disrupt rhythm, force errors, counter",
                "press_trigger_threshold": 0.3,
                "passing_risk": 0.3,
                "verticality": 0.5,
                "width": 0.9,
                "support_distance": 6.0,
                "counterpress_intensity": 0.7,
            },
            "wall": {
                "description": "Block wall lanes, force central, press wide",
                "press_trigger_threshold": 0.4,
                "passing_risk": 0.4,
                "verticality": 0.6,
                "width": 0.8,
                "support_distance": 7.0,
                "counterpress_intensity": 0.5,
            },
            "chaos": {
                "description": "Stay compact, minimize risk, punish mistakes",
                "press_trigger_threshold": 0.5,
                "passing_risk": 0.2,
                "verticality": 0.4,
                "width": 0.7,
                "support_distance": 6.0,
                "counterpress_intensity": 0.3,
            },
            "balanced": {
                "description": "Standard adaptive approach",
                "press_trigger_threshold": 0.4,
                "passing_risk": 0.4,
                "verticality": 0.55,
                "width": 0.7,
                "support_distance": 7.0,
                "counterpress_intensity": 0.5,
            },
            "unknown": {
                "description": "Default balanced approach",
                "press_trigger_threshold": 0.4,
                "passing_risk": 0.4,
                "verticality": 0.55,
                "width": 0.7,
                "support_distance": 7.0,
                "counterpress_intensity": 0.5,
            },
        }
        return counters.get(archetype, counters["balanced"])