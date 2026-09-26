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
    
    # --- Additional tracking for refined archetypes ---
    slap_ticks: int = 0                # Ticks they use slap aggressively
    tackle_aggression_ticks: int = 0   # Ticks they tackle aggressively
    physical_ticks: int = 0            # Ticks they play physically
    compact_width_accum: float = 0.0   # Accumulated y-spread when defending
    compact_samples: int = 0           # Samples for compactness
    deep_players_accum: int = 0        # Accumulated count of deep players

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
                
                # Track shooting distance
                if self.shot_dist is None or ball.x < self.shot_dist:
                    self.shot_dist = ball.x
        
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
                
                # Track compactness and deep players
                ys = [q.y for q in opp]
                if ys:
                    self.compact_width_accum += (max(ys) - min(ys))
                    self.compact_samples += 1
                deep_players = sum(1 for q in opp if q.x > 40.0)
                self.deep_players_accum += deep_players
                
                # Track physical play
                for q in opp:
                    if q.vx < -2.0 and q.x > 30.0:  # Aggressive forward movement
                        self.physical_ticks += 1
        
        # Track possession style when they have ball
        if ball.possessing_team == "them":
            self.possession_ticks += 1
            # Track slaps - detect from events would be ideal, approximate via press intensity
            if self.press_intensity() > 0.7:
                self.slap_ticks += 1

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
        'possession', 'wall', 'physical', 'balanced', 'unknown'
        """
        if self.total_obs_ticks < 20:
            return "unknown"
        
        total = max(1, self.total_obs_ticks)
        high_press_ratio = self.high_press_ticks / total
        low_block_ratio = self.low_block_ticks / total
        counter_ratio = self.counter_attack_ticks / max(1, self.total_obs_ticks)
        direct_ratio = self.direct_play_ticks / max(1, self.samples)
        possession_ratio = self.possession_ticks / max(1, self.samples)
        wall_ratio = self.wall_usage()
        slap_ratio = self.slap_ticks / total if total > 0 else 0
        physical_ratio = self.physical_ticks / total if total > 0 else 0
        
        avg_compact_width = self.compact_width_accum / max(1, self.compact_samples)
        avg_deep_players = self.deep_players_accum / max(1, total)
        
        # High press: sustained pressure in our half
        if high_press_ratio > 0.35:
            return "high_press"
        
        # Low block: sitting deep consistently, compact, 3+ deep players
        if low_block_ratio > 0.45 and avg_deep_players >= 2.5 and avg_compact_width < 22.0:
            return "low_block"
        
        # Counter-attack: frequent fast breaks
        if counter_ratio > 0.12:
            return "counter"
        
        # Direct play: long balls from deep
        if direct_ratio > 0.22:
            return "direct"
        
        # Possession: keep ball, build slowly
        if possession_ratio > 0.55 and self.press_intensity() < 0.45:
            return "possession"
        
        # Wall play: frequent wall usage
        if wall_ratio > 0.25:
            return "wall"
        
        # Physical: aggressive slaps and tackles
        if slap_ratio > 0.2 or physical_ratio > 0.15:
            return "physical"
        
        return "balanced"

    def get_counter_tactics(self) -> dict:
        """Return specific counter-tactics for the detected archetype."""
        archetype = self.detect_archetype()
        
        counters = {
            "high_press": {
                "description": "Play through the press, quick one-twos, exploit space behind",
                "press_trigger_threshold": 0.6,   # Don't press high ourselves
                "passing_risk": 0.25,             # Safe, short passes
                "verticality": 0.35,              # Play through, not over
                "width": 1.0,                     # Use full width to stretch
                "support_distance": 5.0,          # Close support for combinations
                "counterpress_intensity": 0.75,   # Counter-press when we win it
                "transition_speed": 0.85,         # Fast transitions
                "defensive_line": 28.0,           # Deeper rest defence
            },
            "low_block": {
                "description": "Patient build-up, width to stretch, crosses, cut-backs, switches",
                "press_trigger_threshold": 0.25,  # Don't press high
                "passing_risk": 0.45,             # Mix of safe and risky
                "verticality": 0.7,               # Push forward
                "width": 1.0,                     # Maximum width
                "support_distance": 8.0,          # Support in pockets
                "counterpress_intensity": 0.25,   # Don't overcommit
                "transition_speed": 0.35,         # Patient
                "defensive_line": 18.0,           # High line to stretch them
            },
            "counter": {
                "description": "Rest defense, control transitions, don't overcommit",
                "press_trigger_threshold": 0.5,
                "passing_risk": 0.35,
                "verticality": 0.55,
                "width": 0.8,
                "support_distance": 7.0,
                "counterpress_intensity": 0.35,
                "defensive_line": 22.0,           # Deeper rest defense
                "transition_speed": 0.45,
            },
            "direct": {
                "description": "Win first contact, control second balls, press high",
                "press_trigger_threshold": 0.3,
                "passing_risk": 0.25,
                "verticality": 0.5,
                "width": 0.9,
                "support_distance": 6.0,
                "counterpress_intensity": 0.85,
                "defensive_line": 24.0,
            },
            "possession": {
                "description": "Press to disrupt rhythm, force errors, counter",
                "press_trigger_threshold": 0.25,
                "passing_risk": 0.25,
                "verticality": 0.5,
                "width": 0.9,
                "support_distance": 6.0,
                "counterpress_intensity": 0.75,
                "defensive_line": 26.0,
            },
            "wall": {
                "description": "Block wall lanes, force central, press wide",
                "press_trigger_threshold": 0.35,
                "passing_risk": 0.35,
                "verticality": 0.55,
                "width": 0.8,
                "support_distance": 7.0,
                "counterpress_intensity": 0.5,
                "defensive_line": 24.0,
            },
            "physical": {
                "description": "Quick release, body positioning, avoid 50-50s",
                "press_trigger_threshold": 0.45,
                "passing_risk": 0.2,
                "verticality": 0.45,
                "width": 0.75,
                "support_distance": 5.5,
                "counterpress_intensity": 0.3,
                "defensive_line": 26.0,
            },
            "balanced": {
                "description": "Standard adaptive approach",
                "press_trigger_threshold": 0.4,
                "passing_risk": 0.35,
                "verticality": 0.55,
                "width": 0.8,
                "support_distance": 7.0,
                "counterpress_intensity": 0.55,
                "defensive_line": 24.0,
            },
            "unknown": {
                "description": "Default balanced approach",
                "press_trigger_threshold": 0.4,
                "passing_risk": 0.35,
                "verticality": 0.55,
                "width": 0.8,
                "support_distance": 7.0,
                "counterpress_intensity": 0.55,
                "defensive_line": 24.0,
            },
        }
        return counters.get(archetype, counters["balanced"])