"""Import and exercise every generated opponent's strategy.py.

Run as: python scripts/_import_probe.py <opponents-dir>

Prints a JSON summary and exits non-zero if any team fails to import or
answer a decision. Byte-level drift checking cannot detect a module that is
stably generated but syntactically or semantically broken; this can. It runs
in a subprocess because every team ships a module literally named `strategy`.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

OBSERVATION = {
    "protocolVersion": "1.0", "gameId": "probe", "sequence": 0,
    "simulationTick": 0, "applyAtTick": 0,
    "timeRemainingSeconds": 300.0, "phase": "openPlay",
    "score": {"us": 0, "them": 0},
    "ball": {"id": "b", "position": {"x": 30.0, "y": 20.0},
             "velocity": {"x": 0.0, "y": 0.0},
             "possessingTeam": "us", "possessedBy": "d2"},
    "us": [{"id": i, "role": r, "position": {"x": x, "y": y},
            "velocity": {"x": 0.0, "y": 0.0}, "facingRadians": 0.0,
            "canAct": True}
           for i, r, x, y in (("gk", "goalkeeper", 3.0, 20.0),
                              ("d1", "outfield", 26.0, 20.0),
                              ("d2", "outfield", 22.0, 13.0),
                              ("d3", "outfield", 22.0, 27.0),
                              ("d4", "outfield", 30.0, 20.0))],
    "them": [{"id": i, "role": r, "position": {"x": x, "y": y},
              "velocity": {"x": 0.0, "y": 0.0}, "facingRadians": 0.0,
              "canAct": True}
             for i, r, x, y in (("gk", "goalkeeper", 57.0, 20.0),
                                ("d1", "outfield", 40.0, 18.0),
                                ("d2", "outfield", 42.0, 20.0),
                                ("d3", "outfield", 42.0, 22.0),
                                ("d4", "outfield", 44.0, 20.0))],
}

REQUIRED = ("football-team.json", "Dockerfile", "models.py", "server.py",
            "strategy.py")


def probe_one(directory: pathlib.Path) -> str | None:
    """Return an error string, or None on success."""
    missing = [f for f in REQUIRED if not (directory / f).exists()]
    if missing:
        return f"missing {', '.join(missing)}"
    sys.path.insert(0, str(directory))
    try:
        spec = importlib.util.spec_from_file_location(
            f"probe_{directory.name.replace('-', '_')}", directory / "strategy.py")
        if spec is None or spec.loader is None:
            return "no import spec"
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for sequence in range(3):
            observation = dict(OBSERVATION, sequence=sequence)
            decision = module.decide(observation)
            if decision["gameId"] != "probe" or decision["sequence"] != sequence:
                return f"bad echo at sequence {sequence}"
            intents = decision["intents"]
            if len(intents) != 5:
                return f"{len(intents)} intents, want 5"
            for intent in intents:
                if not 0.0 <= intent["move"]["speed"] <= 1.0:
                    return f"speed {intent['move']['speed']} out of range"
        return None
    except Exception as exc:  # noqa: BLE001 - report anything at all
        return f"{type(exc).__name__}: {exc}"
    finally:
        sys.path.pop(0)
        for name in [m for m in sys.modules if m.startswith("probe_")]:
            del sys.modules[name]


def main() -> int:
    if len(sys.argv) > 1:
        root = pathlib.Path(sys.argv[1])
    else:
        # Default to the repo's opponents/ regardless of the current directory,
        # so the probe behaves the same from the repo root or from the team dir.
        root = pathlib.Path(__file__).resolve().parents[3] / "opponents"
    if not root.is_dir():
        print(json.dumps({"error": f"no opponents directory at {root}"}, indent=2))
        return 2
    failures = []
    ok = 0
    for directory in sorted(p for p in root.iterdir() if p.is_dir()):
        error = probe_one(directory)
        if error:
            failures.append(f"{directory.name}: {error}")
        else:
            ok += 1
    print(json.dumps({"ok": ok, "failures": failures}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
