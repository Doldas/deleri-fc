"""Tuning surface for synthetic opponent teams.

An `OpponentSpec` is the complete description of one opponent: which tactical
family it plays like plus every behavioural knob that family exposes. The
`TunableOpponent` in `zoo.py` is a single generic controller driven purely by a
spec, so adding a new opponent is a data change, not a code change (AISTRATEGI
§37 diversity preservation).

`difficulty` is a first-class knob rather than an afterthought: `harden()`
composes a family preset with it so "extremely difficult" opponents stay
internally consistent (fast, sharp, disciplined, well-timed) instead of just
being noisier.

Nothing here is imported by the decision path — this package exists only for
offline evaluation (AGENTS.md §7).
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

from ..geom import PITCH_LENGTH, PITCH_WIDTH

# Tactical families. Every opponent is an instance of exactly one.
FAMILIES: tuple[str, ...] = (
    "possession",   # patient circulation, keep the ball, probe the block
    "high_press",   # coordinated, trigger-based swarm on the ball carrier
    "low_block",    # deep compact zonal block, clear rather than build
    "counter",      # deep, then a vertical sprint the moment they win it
    "direct",       # fast vertical, minimal circulation
    "wall",         # touchline hugging, near-wall passing, cutback shots
    "tika",         # short-pass positional play with constant rotation
    "overload",     # load one channel hard, switch to the empty one
    "physical",     # hunting tackles/slaps, disrupt rhythm
    "gk_hell",      # ultra-aggressive keeper sweeping everything
    "chaos",        # high noise, unstructured, unpredictable
    "lure",         # deception: fake openings, then burst the other way
    "learned",      # online-RL agent, weights learned by self-play
)

# Home-shape anchors in attack-direction fractions: (depth, width) where
# depth 0.0 is our own goal line and 1.0 is the goal we attack, and width 0.5
# is the centre of the pitch. Order matches `BASE_LINEUP` minus the keeper.
ANCHOR_DEPTH: tuple[float, ...] = (0.30, 0.46, 0.46, 0.70)
ANCHOR_WIDTH: tuple[float, ...] = (0.50, 0.20, 0.80, 0.50)


@dataclass(frozen=True)
class OpponentSpec:
    """Every knob an opponent controller can be tuned with.

    All 0..1 knobs are fractions; the two length knobs are metres. Defaults
    describe a competent, fairly direct mid-table side — `harden()` is what
    turns this into a genuinely hard opponent.
    """

    name: str
    family: str
    difficulty: float = 0.5
    blurb: str = ""

    # --- movement ---
    tempo: float = 0.85              # speed multiplier for committed runs
    line_height: float = 0.50        # defensive line as pitch fraction
    compactness: float = 0.60        # how tightly the block narrows to the ball
    width: float = 0.70              # how far the block stretches to the touchlines
    support_depth: float = 0.50      # how far support runners push beyond the ball
    transition_speed: float = 0.70   # burst speed straight after a turnover
    gk_speed: float = 0.60
    gk_aggression: float = 0.30      # how far the keeper leaves its line

    # --- pressing ---
    press_intensity: float = 0.60    # how many outfielders commit to the ball
    press_trigger: float = 0.60      # press quality: press high vs stay compact
    press_delay: float = 0.00        # reaction lag in seconds
    counterpress: float = 0.50       # how hard they swarm after losing it
    tackling: float = 0.50           # tackle attempt rate
    marking: float = 0.60            # how tightly they shadow our receivers
    physical: float = 0.30           # slap/tackle usage against a set shape

    # --- on the ball ---
    directness: float = 0.60         # forward passes vs safe circulation
    risk: float = 0.50               # willingness to pass into traffic
    shoot_range: float = 18.0        # metres from goal where they pull the trigger
    shoot_power: float = 0.85
    pass_power: float = 0.60
    wall_usage: float = 0.00         # probability of pinning play to a wall
    shot_ambition: float = 0.50      # how eagerly they shoot in the box

    # --- meta / variance ---
    tempo_switch: float = 0.30       # plan changes on score/time state
    noise: float = 0.10              # execution error, both sides of the truth
    deception: float = 0.00          # lure behaviour weight
    discipline: float = 0.95         # probability of executing the intended action

    seed_salt: int = 0

    # -- helpers ---------------------------------------------------------------

    def evolve(self, **overrides) -> "OpponentSpec":
        """A copy with some knobs replaced (`name`/`family` included)."""
        return replace(self, **overrides)

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


# Per-family offsets applied on top of the shared defaults. These are *identity*
# traits: a high_press side is a pressing side first and everything else second.
FAMILY_PRESETS: dict[str, dict] = {
    "possession": dict(
        directness=0.30, risk=0.20, compactness=0.70, width=0.80, line_height=0.45,
        press_intensity=0.35, press_trigger=0.40, support_depth=0.55, shoot_range=15.0,
        blurb="patient circulation, keeps the ball away from trouble",
    ),
    "high_press": dict(
        press_intensity=0.95, press_trigger=0.85, counterpress=0.90, tackling=0.85,
        compactness=0.80, line_height=0.72, transition_speed=0.90, tempo=0.95,
        directness=0.65, risk=0.60, blurb="trigger-coordinated swarm, wins it back high",
    ),
    "low_block": dict(
        press_intensity=0.25, press_trigger=0.30, line_height=0.28, compactness=0.95,
        width=0.35, directness=0.25, risk=0.15, shoot_range=13.0, counterpress=0.30,
        transition_speed=0.55, blurb="deep compact zonal block, clears rather than builds",
    ),
    "counter": dict(
        press_intensity=0.35, line_height=0.25, compactness=0.75, transition_speed=0.95,
        directness=0.85, risk=0.65, shoot_range=20.0, shot_ambition=0.75,
        counterpress=0.40, tempo=0.95, blurb="mid block then a vertical breakaway",
    ),
    "direct": dict(
        directness=0.95, risk=0.70, support_depth=0.80, transition_speed=0.90,
        shoot_range=19.0, shot_ambition=0.70, pass_power=0.80, line_height=0.55,
        blurb="fast vertical, minimal circulation",
    ),
    "wall": dict(
        wall_usage=0.85, width=0.95, directness=0.45, shoot_range=17.0, risk=0.30,
        compactness=0.45, blurb="pins play to the touchline, cutback shots",
    ),
    "tika": dict(
        directness=0.25, risk=0.25, compactness=0.90, width=0.65, line_height=0.60,
        support_depth=0.75, shoot_range=14.0, pass_power=0.45, blurb="short-pass rotation under pressure",
    ),
    "overload": dict(
        width=0.90, directness=0.80, risk=0.65, transition_speed=0.85, compactness=0.55,
        shoot_range=19.0, blurb="loads one channel, switches to the empty one",
    ),
    "physical": dict(
        physical=0.95, tackling=0.95, press_intensity=0.70, press_trigger=0.70,
        compactness=0.70, tempo=0.90, blurb="hunts the ball, disrupts rhythm",
    ),
    "gk_hell": dict(
        gk_aggression=0.95, gk_speed=0.95, compactness=0.85, press_intensity=0.75,
        press_trigger=0.75, line_height=0.60, blurb="sweeper-keeper hoovering up everything",
    ),
    "chaos": dict(
        noise=0.55, discipline=0.55, tempo=0.80, compactness=0.30, width=0.85,
        directness=0.55, risk=0.70, blurb="high-variance, barely repeatable",
    ),
    "lure": dict(
        deception=0.90, directness=0.75, risk=0.60, transition_speed=0.90,
        compactness=0.70, shoot_range=19.0, tempo=0.90, press_intensity=0.60,
        blurb="shows a gap, invites the press, bursts the channel you vacated",
    ),
    "learned": dict(
        noise=0.08, tempo=0.90, directness=0.60, compactness=0.70, line_height=0.55,
        blurb="online-RL controller, weights learned by self-play",
    ),
}

# How `difficulty` maps onto the shared knobs. Raising difficulty makes an
# opponent faster, sharper, more disciplined and less error-prone; it never
# just adds randomness, because a merely chaotic opponent is not a hard one.
HARDEN_BIAS: dict[str, float] = {
    "tempo": +0.15,
    "press_intensity": +0.20,
    "press_trigger": +0.25,
    "counterpress": +0.20,
    "tackling": +0.20,
    "marking": +0.20,
    "compactness": +0.15,
    "directness": +0.10,
    "support_depth": +0.10,
    "transition_speed": +0.10,
    "gk_speed": +0.10,
    "gk_aggression": +0.10,
    "shoot_range": +4.0,
    "shoot_power": +0.05,
    "pass_power": +0.05,
    "shot_ambition": +0.15,
    "discipline": +0.05,
    # Errors shrink, but never to zero: a completely noiseless opponent is
    # easier to pattern-match than a very sharp one.
    "noise": -0.08,
    "risk": +0.05,
    "width": +0.10,
}

# The same knob is pushed in the opposite direction at low difficulty, so the
# scale is a genuine two-sided axis rather than a floor.
SOFTEN_BIAS: dict[str, float] = {
    "tempo": -0.20,
    "press_intensity": -0.25,
    "press_trigger": -0.30,
    "counterpress": -0.25,
    "tackling": -0.25,
    "marking": -0.25,
    "compactness": -0.15,
    "directness": -0.15,
    "support_depth": -0.10,
    "transition_speed": -0.15,
    "gk_speed": -0.10,
    "gk_aggression": -0.10,
    "shoot_range": -6.0,
    "shoot_power": -0.10,
    "pass_power": -0.10,
    "shot_ambition": -0.20,
    "discipline": -0.15,
    "noise": +0.14,
    "risk": -0.15,
    "width": -0.15,
}


def _clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v


# Most knobs are 0..1 fractions, but a few are metres. Clamping a length knob
# with `_clamp01` silently turned `shoot_range` into 1.0 m, which disabled
# shooting entirely -- so lengths get their own ranges.
LENGTH_RANGES: dict[str, tuple[float, float]] = {
    "shoot_range": (5.0, 32.0),
}


def _clamp_key(key: str, value: float) -> float:
    lo, hi = LENGTH_RANGES.get(key, (0.0, 1.0))
    return lo if value < lo else hi if value > hi else value


def _apply_bias(spec: OpponentSpec, bias: dict[str, float], scale: float) -> OpponentSpec:
    over: dict = {}
    for key, delta in bias.items():
        over[key] = _clamp_key(key, float(getattr(spec, key)) + delta * scale)
    return replace(spec, **over)


def make_spec(
    name: str,
    family: str,
    difficulty: float = 0.5,
    seed_salt: int = 0,
    **overrides,
) -> OpponentSpec:
    """Build a spec: family preset -> difficulty scaling -> explicit overrides.

    `difficulty` is 0..1 where 0.5 is the neutral preset, ~0.85 is a genuinely
    strong side and >=0.95 is the "extremely difficult" band the arena uses for
    stress testing.
    """
    if family not in FAMILY_PRESETS:
        raise ValueError(f"unknown opponent family: {family!r}")
    d = _clamp01(difficulty)
    spec = OpponentSpec(name=name, family=family, difficulty=d, seed_salt=seed_salt)
    spec = replace(spec, **FAMILY_PRESETS[family])
    # difficulty 0.5 -> no bias; above/below it scales HARDEN/SOFTEN symmetrically.
    if d >= 0.5:
        spec = _apply_bias(spec, HARDEN_BIAS, (d - 0.5) * 2.0)
    else:
        spec = _apply_bias(spec, SOFTEN_BIAS, (0.5 - d) * 2.0)
    spec = replace(spec, difficulty=d)
    if overrides:
        spec = replace(spec, **overrides)
    return spec


# Absolute bounds the generic controller relies on. Kept here so `zoo.py` and
# the registry agree on what "clamped" means.
X_MARGIN = 0.5
Y_MARGIN = 0.5
X_MAX = PITCH_LENGTH - X_MARGIN
Y_MAX = PITCH_WIDTH - Y_MARGIN


def clamp_x(x: float) -> float:
    return X_MARGIN if x < X_MARGIN else X_MAX if x > X_MAX else x


def clamp_y(y: float) -> float:
    return Y_MARGIN if y < Y_MARGIN else Y_MAX if y > Y_MAX else y
