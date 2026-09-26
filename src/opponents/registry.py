"""Registry of every opponent available to the offline simulator.

`src.sim.OPPONENTS` is a name -> factory mapping; the factory takes the match
`rng` and returns a fresh controller. Keeping it a factory (rather than a
singleton) is what makes matches independent and reproducible.

The registry is data, not code: `build_registry` returns factories derived from
the family list in `spec.py`, so adding a family automatically widens the pool.
Names are stable strings because they end up in experiment logs, arena
reports and evolution baselines.
"""

from __future__ import annotations

import random
from collections.abc import Callable

from .learned import make_learned
from .lure import make_lure
from .spec import FAMILIES
from .zoo import TunableOpponent, make_tunable

Controller = Callable[[random.Random], object]

# Families that are strong enough to serve as evolution baselines on their own.
ELITE_FAMILIES = (
    "possession",
    "high_press",
    "counter",
    "direct",
    "wall",
    "tika",
    "overload",
    "physical",
    "gk_hell",
    "low_block",
)

# Difficulty bands, low to high. Evolution samples across all of them.
BANDS = (
    ("rookie", 0.45),
    ("solid", 0.65),
    ("pro", 0.82),
    ("elite", 0.95),
)

# Each lure variant is a different reading of the same feint.
LURE_NAMES = ("serpent", "mimic", "phantom", "trap")

# Learned variants differ in how aggressively they explore and how quickly
# they commit to a habit.
LEARNED_NAMES = (
    "tabula",
    "adaptive",
    "methodic",
    "opportunist",
    "grim",
    "novice",
)

# Names alone are not variety. Each variant overrides real behaviour, otherwise
# the "pool" is one opponent wearing four hats and the arena ranking is noise.
LURE_PROFILES: dict[str, dict] = {
    # Sharp, decisive read of the weak channel; commits hard, few mistakes.
    "serpent": dict(deception=1.0, noise=0.02, tempo=1.0, counterpress=0.9, width=0.7),
    # Same feint, but the channel choice is close to a coin flip.
    "mimic": dict(deception=0.9, noise=0.22, tempo=0.85, counterpress=0.5, width=0.8),
    # Invites the press out far more often, then counters into the vacated half.
    "phantom": dict(deception=1.0, noise=0.08, counterpress=0.95, transition_speed=1.0,
                    line_height=0.25),
    # Rarely feints. Backs off almost entirely and punishes the transition.
    "trap": dict(deception=0.15, noise=0.04, counterpress=1.0, press_intensity=0.35,
                 compactness=0.9, line_height=0.2),
}

LEARNED_PROFILES: dict[str, dict] = {
    # High learning rate and high action noise: explores hard, unstable.
    "tabula": dict(noise=0.20, directness=0.5, risk=0.6),
    # Balanced mid-tempo learner.
    "adaptive": dict(noise=0.08, tempo=0.9, directness=0.6),
    # Low noise, low learning rate: exploits steadily instead of gambling.
    "methodic": dict(noise=0.03, tempo=0.8, compactness=0.7, risk=0.4),
    # High risk and directness: goes forward whenever it can.
    "opportunist": dict(noise=0.10, risk=0.9, directness=0.85, tempo=1.0),
    # Physical: hunts the ball, breaks up circulation.
    "grim": dict(noise=0.05, tackling=1.0, press_intensity=1.0, counterpress=0.9),
    # Deliberately poor: high noise, slow, keeps the ball safe.
    "novice": dict(noise=0.28, tempo=0.6, compactness=0.4, press_intensity=0.4,
                   shoot_range=9.0),
}


def build_registry() -> dict[str, Controller]:
    """Every opponent, keyed by the name used in reports and CLI flags."""
    registry: dict[str, Controller] = {}

    for family in FAMILIES:
        # `chaos`, `lure` and `learned` have dedicated constructors below; a
        # plain tuned preset under those names would be actively misleading.
        if family in ("chaos", "lure", "learned"):
            continue
        for band, difficulty in BANDS:
            name = f"{family}-{band}"

            def factory(rng: random.Random, n=name, f=family, d=difficulty) -> TunableOpponent:
                return make_tunable(rng, n, f, d)

            registry[name] = factory

    for name in LURE_NAMES:
        def lure_factory(rng: random.Random, n=name) -> TunableOpponent:
            return make_lure(rng, n, 0.9, **LURE_PROFILES[n])

        registry[f"lure-{name}"] = lure_factory

    for name in LEARNED_NAMES:
        profile = LEARNED_PROFILES[name]
        # Vary the base difficulty so the learned pool is not one strength.
        difficulty = 0.9 if name in ("methodic", "grim", "opportunist") else 0.6

        def learned_factory(rng: random.Random, n=name, d=difficulty, p=profile):
            return make_learned(rng, n, d, **p)

        registry[f"learned-{name}"] = learned_factory

    # One deliberately chaotic side: a real yardstick for "am I exploiting the
    # opponent's habits?", because it has none.
    registry["chaos-chaos"] = lambda rng: make_tunable(rng, "chaos-chaos", "chaos", 1.0)
    return registry


REGISTRY: dict[str, Controller] = build_registry()


def opponent_names() -> list[str]:
    return sorted(REGISTRY)


__all__ = ["REGISTRY", "build_registry", "opponent_names", "ELITE_FAMILIES", "BANDS"]
