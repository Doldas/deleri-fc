"""Configuration, tactical genome and reward weights.

The genome holds the evolutionary tactical parameters (AISTRATEGI §34, §63).
All values stay deterministic and keys are stable so evolved individuals and
the runtime policy share the same schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .geom import clamp

# --- Engine constants (authoritative, from RULES.md) -----------------------
MAX_RUN_SPEED = 8.0
MAX_DRIBBLE_SPEED = 6.4
CONTROLLED_BALL_AHEAD = 0.65
BALL_DECAY = 0.992
BALL_CONTROL_RADIUS = 0.9
BALL_CONTROL_MAX_SPEED = 5.0
# The engine allows a touch at up to BALL_CONTROL_MAX_SPEED, but a ball rolling
# at exactly that limit is a coin flip to control. Every "where does this ball
# stop" calculation uses this slightly stricter speed so the policy plans
# against the ball being genuinely under control, not merely touch-legal.
BALL_CONTROL_SAFE_SPEED = BALL_CONTROL_MAX_SPEED - 0.5
TACKLE_MAX = 1.15
TACKLE_CLOSE = 0.8
SLIDE_LOOSE_SPEED = 10.0
SLIDE_GROUNDED = 1.05
ACTION_COOLDOWN = 0.3
POSSESSION_PROTECTION = 0.25
SLAP_MAX = 1.15
SLAP_CLEAN = 0.75
SLAP_KNOCKBACK = 0.25
SLAP_CLEAN_KNOCKBACK = 0.7
SLAP_STAGGER = 0.3
SLAP_KNOCKDOWN = 1.2
SLAP_COOLDOWN = 1.5
SLAP_FACING_DEG = 60.0
KNOCKDOWN_IMMUNITY = 1.25
GK_REACH = 1.65
GK_DIVE_GROUNDED = 0.9
GK_HOLD_AUTO = 1.25
GK_AUTO_SPEED = 20.0
KICK_MIN_SPEED = 12.0
KICK_MAX_SPEED = 26.0

# Master match flow (decision every 100 ms, engine sim 60 Hz).
DECISION_INTERVAL = 0.1
ENGINE_TICKS_PER_SECOND = 60
DECISION_HARD_MS = 100


def kick_speed(power: float) -> float:
    """Power 0..1 maps linearly to initial ball speed 12..26 m/s."""
    return KICK_MIN_SPEED + (KICK_MAX_SPEED - KICK_MIN_SPEED) * power


def power_for(speed: float) -> float:
    return (speed - KICK_MIN_SPEED) / (KICK_MAX_SPEED - KICK_MIN_SPEED)


def shooting_distance(threshold: float) -> float:
    """0..1 shooting aggressiveness maps to 8..26 m shooting range."""
    t = clamp01(threshold)
    return 8.0 + t * 18.0


# --- Genome schema ----------------------------------------------------------

GENOME_KEYS: tuple[str, ...] = (
    "press_intensity",
    "press_trigger_threshold",
    "width",
    "depth",
    "verticality",
    "risk",
    "counterpress_intensity",
    "compactness",
    "passing_risk",
    "shooting_threshold",
    "wall_usage",
    "wall_pass_threshold",
    "wall_shot_threshold",
    "transition_speed",
    "support_distance",
    "defensive_line",
)

# (low, high) per genome key.
GENOME_RANGES: dict[str, tuple[float, float]] = {
    "press_intensity": (0.0, 1.0),
    "press_trigger_threshold": (0.0, 1.0),
    "width": (0.2, 1.0),
    "depth": (0.2, 1.0),
    "verticality": (0.0, 1.0),
    "risk": (0.0, 1.0),
    "counterpress_intensity": (0.0, 1.0),
    "compactness": (0.0, 1.0),
    "passing_risk": (0.0, 1.0),
    "shooting_threshold": (0.3, 1.0),
    "wall_usage": (0.0, 1.0),
    "wall_pass_threshold": (0.0, 1.0),
    "wall_shot_threshold": (0.0, 1.0),
    "transition_speed": (0.0, 1.0),
    "support_distance": (2.0, 12.0),
    "defensive_line": (12.0, 38.0),
}


def default_genome() -> dict[str, float]:
    return {
        # Pressing: High intensity, low threshold = aggressive coordinated press
        "press_intensity": 0.75,
        "press_trigger_threshold": 0.25,  # Lower = press more readily
        # Width: Maximum width to stretch play, especially against low blocks
        "width": 1.0,
        # Depth: Push high up the pitch
        "depth": 0.8,
        # Verticality: High = direct, forward play
        "verticality": 0.7,
        # Risk: Moderate - calculated risks for high reward
        "risk": 0.55,
        # Counter-press: Very aggressive on turnover
        "counterpress_intensity": 0.85,
        # Compactness: Moderate - maintain shape but allow width
        "compactness": 0.55,
        # Passing risk: Lower = safer passes, but we have specific patterns for risk
        "passing_risk": 0.3,
        # Shooting threshold: Lower = shoot more often, especially in box
        "shooting_threshold": 0.4,
        # Wall usage: High = exploit walls aggressively
        "wall_usage": 0.8,
        "wall_pass_threshold": 0.35,  # Lower = more willing to use wall passes
        "wall_shot_threshold": 0.5,   # Willing to take wall shots
        # Transition speed: Fast transitions
        "transition_speed": 0.8,
        # Support distance: Medium - close enough for combinations, far enough for width
        "support_distance": 7.0,
        # Defensive line: Adaptive, but base higher for high press
        "defensive_line": 26.0,
    }


def make_genome(style: dict | None = None) -> dict[str, float]:
    """Build a genome from the builder `tactics.json` style block."""
    genome = default_genome()
    if not style:
        return genome
    press = float(style.get("pressing", 55)) / 100.0
    depth = (100.0 - float(style.get("defensiveDepth", 45))) / 100.0
    direct = float(style.get("directness", 55)) / 100.0
    genome["press_intensity"] = press
    genome["press_trigger_threshold"] = clamp01(0.7 - press * 0.6)
    genome["depth"] = clamp01(depth)
    genome["verticality"] = clamp01(direct)
    genome["defensive_line"] = clamp(30.0 - direct * 14.0, GENOME_RANGES["defensive_line"][0], GENOME_RANGES["defensive_line"][1])
    genome["shooting_threshold"] = clamp01(float(style.get("shootingDistanceMeters", 16)) / 40.0)
    genome["passing_risk"] = clamp01(0.5 - direct * 0.2)
    return genome


def clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v


def genome_hash(g: dict[str, float]) -> str:
    import hashlib
    raw = ",".join(f"{k}:{g[k]:.6f}" for k in GENOME_KEYS)
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


# --- Reward weights (AISTRATEGI §31) ----------------------------------------

REWARD_DEFAULTS: dict[str, float] = {
    "goal": 1.0,
    "goal_conceded": -1.0,
    "territorial_progression": 0.0012,
    "possession_quality": 0.0006,
    "space_created": 0.0004,
    "chance_creation": 0.002,
    "shot_quality": 0.0008,
    "successful_press": 0.004,
    "counterpress_recovery": 0.01,
    "defensive_compactness": 0.0006,
    "passing_quality": 0.0005,
    "wall_progression": 0.001,
    "dangerous_turnover": -0.006,
    "broken_shape": -0.004,
    "failed_press": -0.003,
    "bad_shot": -0.004,
    "unnecessary_risk": -0.002,
}


# --- Runtime config ---------------------------------------------------------

@dataclass
class RuntimeConfig:
    genome: dict[str, float] = field(default_factory=default_genome)
    reward_weights: dict[str, float] = field(
        default_factory=lambda: dict(REWARD_DEFAULTS)
    )
    enable_mcts: bool = False  # runtime-limited MCTS override (off by default)
    mcts_iterations: int = 30
    mcts_horizon: float = 2.0
    log_explanations: bool = False
    force_seed: int | None = None  # set for reproducible tournament decisions
    recycle_count: int = 0  # consecutive backward passes to force forward play

    def style_key(self) -> str:
        return genome_hash(self.genome)