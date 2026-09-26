"""Offline harness for the real-engine opponent brain.

`src/opponents/engine_brain.py` is written against the engine's observation
dicts, not against `src.light.PlanState`. `src/sim.py` already produces exactly
that shape via `plan_to_obs`, so the brain can be driven by the internal
`LightEngine` with a thin adapter.

This is how the generated `opponents/*/strategy.py` gets tested at all: the same
file that ships inside a bare Alpine container is exercised here, offline, in
under a second per match, before anyone pays for a Docker build.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import random
import sys
import types

ROOT = pathlib.Path(__file__).resolve().parents[2]  # my-team-fc root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.opponents import engine_brain as _engine_brain  # noqa: E402
from src.runtime import RuntimeManager  # noqa: E402
from src.sim import play_match  # noqa: E402


def build_module(params: dict) -> types.ModuleType:
    """A private module instance of the brain with parameters baked in.

    The source is re-executed rather than shallow-copied. Copying `__dict__`
    would look right and silently test nothing: the copied `decide` still closes
    over the *original* module's globals, so every "variant" would quietly run
    with the default parameters.
    """
    import importlib.util

    source = (ROOT / "src" / "opponents" / "engine_brain.py").read_text()
    # A stable digest, not `hash()`: string hashing is salted per process, so
    # module names changed between runs and cross-module brain state could not
    # be reasoned about.
    digest = hashlib.blake2b(
        json.dumps(params, sort_keys=True).encode(), digest_size=4
    ).hexdigest()
    name = f"engine_brain_{digest}"
    module = types.ModuleType(name)
    module.__file__ = str(ROOT / "src" / "opponents" / "engine_brain.py")
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    setattr(module, "TRAINED_PARAMS", dict(params))
    return module


PITCH_LENGTH = 60.0


def mirror(obs: dict) -> dict:
    """Flip the observation into the away team's own frame of reference.

    The brain is written for the real engine, where a team always defends x=0
    and attacks x=60. The internal engine shares one pitch, where "us" attacks
    +x, so the away side has to be mirrored along x and have its sides swapped
    before the brain sees it. Without this the opponent brain is handed the
    home team's viewpoint, controls the home players, and has its kicks
    rejected as illegal.
    """
    obs = dict(obs)
    obs["us"], obs["them"] = obs["them"], obs["us"]

    def flip_player(q: dict) -> dict:
        q = dict(q)
        q["position"] = {"x": PITCH_LENGTH - q["position"]["x"], "y": q["position"]["y"]}
        return q

    obs["us"] = [flip_player(q) for q in obs["us"]]
    obs["them"] = [flip_player(q) for q in obs["them"]]
    ball = dict(obs["ball"])
    ball["position"] = {
        "x": PITCH_LENGTH - ball["position"]["x"],
        "y": ball["position"]["y"],
    }
    ball["velocity"] = {
        "x": -ball.get("velocity", {}).get("x", 0.0),
        "y": ball.get("velocity", {}).get("y", 0.0),
    }
    team = ball.get("possessingTeam")
    if team in ("us", "them"):
        ball["possessingTeam"] = "them" if team == "us" else "us"
    obs["ball"] = ball
    return obs


def make_controller(module: types.ModuleType):
    """A `src.sim` opponent controller backed by the engine brain."""

    class _Controller:
        def __init__(self, rng: random.Random) -> None:
            self._module = module
            self._game_id = f"probe-{rng.randrange(1 << 30)}"
            self._seq = 0

        def decide(self, st, team: str) -> dict:
            from src.sim import plan_to_obs

            obs = mirror(plan_to_obs(st, self._game_id, self._seq))
            # `PlanState` has no tick counter, so drive the engine-shaped
            # `simulationTick` at the real engine's cadence: six 60 Hz ticks per
            # decision. That lets the brain learn the true decision interval
            # here the same way it will in the container.
            obs["simulationTick"] = self._seq * 6
            self._seq += 1
            return self._to_internal(module.decide(obs))

        @staticmethod
        def _to_internal(decision: dict) -> dict:
            """Wire `Decision` -> the internal `{(team, pid): LIntent}` map.

            This is the exact boundary the real engine sits on: everything above
            it is protocol-shaped, everything below is the offline engine's
            convenience type. Keeping the conversion in one place is what makes
            it safe to test the shipped brain offline.
            """
            from src.light import LIntent

            out = {}
            for intent in decision["intents"]:
                move = intent.get("move") or {}
                target = move.get("target") or {}
                action = intent.get("action") or {}
                at = action.get("target")
                out[("them", intent["playerId"])] = LIntent(
                    # Un-mirror x: the brain plans in its own frame (defend 0,
                    # attack 60), the internal engine shares one pitch.
                    tx=PITCH_LENGTH - float(target.get("x", 0.0)),
                    ty=float(target.get("y", 0.0)),
                    speed=float(move.get("speed", 0.5)),
                    act=str(action.get("type", "none")),
                    action_target=(
                        (PITCH_LENGTH - float(at["x"]), float(at["y"])) if at else None
                    ),
                    power=action.get("power"),
                )
            return out

    return _Controller


def play(params: dict, seed: int, decisions: int = 1200) -> dict:
    name = f"engine-{abs(hash(str(params))) % 10**8}"
    import src.sim as sim

    sim.OPPONENTS[name] = make_controller(build_module(params))
    return play_match(
        RuntimeManager(), name, rng=random.Random(seed), decisions=decisions
    ).summary()
