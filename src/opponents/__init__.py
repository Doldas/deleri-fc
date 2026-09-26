"""Offline opponent pool for `src.sim` and the evolution baselines.

Nothing in here is imported at module import time by the team's own policy; the
package is only pulled in when `src.sim` extends `OPPONENTS` or the arena
script asks for a controller. That keeps the built image free of the arena
harness while still making every opponent available in-simulation.
"""

from .registry import REGISTRY, build_registry, opponent_names
from .spec import FAMILIES, OpponentSpec, make_spec
from .zoo import TunableOpponent, make_tunable

__all__ = [
    "REGISTRY",
    "build_registry",
    "opponent_names",
    "FAMILIES",
    "OpponentSpec",
    "make_spec",
    "TunableOpponent",
    "make_tunable",
]
