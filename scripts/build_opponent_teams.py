#!/usr/bin/env python3
"""Generate the real-engine opponent teams under `opponents/`.

Each generated directory is a complete, self-contained team: its Dockerfile only
copies `models.py strategy.py server.py`, so the shared brain from
`src/opponents/engine_brain.py` is inlined into `strategy.py` along with that
team's `PARAMS` block. One implementation, N teams, no hand-edited copies.

Regenerating is always safe: the output is deterministic, and `--check` verifies
the files on disk still match what the generator would produce, so a stale
generated file is a test failure rather than a mystery.

Usage::

    python scripts/build_opponent_teams.py             # write opponents/
    python scripts/build_opponent_teams.py --check     # verify, write nothing
    python scripts/build_opponent_teams.py --only lure-serpent
"""

from __future__ import annotations

import argparse
import json
import pathlib
import pprint
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent  # My Teams/deleri-fc
REPO = ROOT.parent.parent  # repository root, which holds `opponents/`
sys.path.insert(0, str(ROOT))

from src.opponents.engine_brain import _Brain  # noqa: E402  (params schema)
from src.opponents.registry import (  # noqa: E402
    BANDS,
    LEARNED_PROFILES,
    LURE_PROFILES,
)

OPPONENTS_DIR = REPO / "opponents"
BRAIN_SOURCE = ROOT / "src" / "opponents" / "engine_brain.py"
MODELS_SOURCE = ROOT / "training" / "opponents" / "null-pointers-fc" / "models.py"
SERVER_SOURCE = ROOT / "training" / "opponents" / "null-pointers-fc" / "server.py"

# Kit colours per family, with the four bands of a family as shades of one hue so
# they read as related. This exists because `server.py` was originally copied
# verbatim from the starter, so all fifty teams announced themselves as
# "Null Pointers FC" in the same blue and white and were indistinguishable in
# any UI -- including a scoreline read, where two different opponents were
# impossible to tell apart.
FAMILY_COLORS: dict[str, str] = {
    "possession": "#1D4ED8",   # blue
    "high_press": "#DC2626",   # red
    "low_block": "#1E3A8A",    # deep navy
    "counter": "#EA580C",      # orange
    "direct": "#CA8A04",       # amber
    "wall": "#7C3AED",         # violet
    "tika": "#0891B2",         # cyan
    "overload": "#16A34A",     # green
    "physical": "#B91C1C",     # dark red
    "gk_hell": "#4F46E5",      # indigo
    "chaos": "#A21CAF",        # fuchsia
    "lure": "#BE185D",          # pink
    "learned": "#0F766E",       # teal
}
# Multiplier on lightness per band, so `rookie` through `elite` are visibly
# different sides rather than four identical kits.
BAND_SHADE: dict[str, float] = {"rookie": 1.45, "solid": 1.15, "pro": 0.95,
                                "elite": 0.72}
# Six distinct shade factors for the band-less families, so `lure-*` and
# `learned-*` are each their own visual side instead of four or six identical
# kits. The range matches the tuned ladder so the two groups read alike.
NO_BAND_SHADES: tuple[float, ...] = (1.45, 1.15, 0.95, 0.72, 1.30, 0.84)


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*(max(0, min(255, round(c))) for c in rgb))


def shade(hex_color: str, factor: float, mix_white: float = 0.0) -> str:
    """Scale a colour toward black (factor < 1) or white (factor > 1).

    Scaling alone runs into clipping on already-bright channels and muddy
    results on saturated ones, so bright shades are mixed toward white instead.
    """
    r, g, b = _hex_to_rgb(hex_color)
    if factor >= 1.0:
        t = min(1.0, (factor - 1.0) / 0.6)
        r, g, b = (c + (255 - c) * t for c in (r, g, b))
    else:
        r, g, b = (c * factor for c in (r, g, b))
    if mix_white:
        r, g, b = (c + (255 - c) * mix_white for c in (r, g, b))
    return _rgb_to_hex((r, g, b))


# Two-letter family code. The raw family initial is not enough: `possession` and
# `physical` both start with P, so their four bands collided on PRO/PSO/PPR/PEL.
FAMILY_CODE: dict[str, str] = {
    "possession": "PS", "high_press": "HP", "low_block": "LB", "counter": "CN",
    "direct": "DR", "wall": "WL", "tika": "TK", "overload": "OL",
    "physical": "PH", "gk_hell": "GK", "chaos": "CH", "lure": "LR",
    "learned": "LN",
}


def short_name(team_id: str) -> str:
    """A three-letter code that is unique across the pool."""
    parts = [p for p in team_id.replace("_", "-").split("-") if p]
    if len(parts) < 2:
        return parts[0][:3].upper() if parts else "FC"
    family = "_".join(parts[:-1]) if len(parts) > 2 else parts[0]
    family = {"gk-hell": "gk_hell", "high-press": "high_press",
              "low-block": "low_block"}.get(family, family)
    code = FAMILY_CODE.get(family)
    if code is None:  # unknown family: fall back to initials, padded
        code = "".join(p[0] for p in parts).upper()
    return (code + parts[-1][0].upper())[:3]


def team_colors(team_id: str, params: dict) -> dict[str, str]:
    family = params.get("family", "possession")
    base = FAMILY_COLORS.get(family, "#2563EB")
    band = team_id.rsplit("-", 1)[-1]
    if family in ("lure", "learned"):
        # These have no band, so a fixed band would give every variant in the
        # family the same kit. Spread them across the shade range instead: still
        # one family hue, but each variant is its own side.
        ladder = ["elite", "pro", "solid", "rookie"]
        index = _salt(team_id) % len(ladder)
        band = ladder[index]
    factor = BAND_SHADE.get(band, 1.0)
    if family in ("lure", "learned"):
        # No band of their own, so take distinct factors by position within the
        # family. Injective by construction: indexed by `kit_index`, not hashed,
        # so two variants can never land on the same rung.
        ladder = NO_BAND_SHADES
        factor = ladder[params.get("kit_index", 0) % len(ladder)]
    primary = shade(base, factor)
    # Secondary stays light so the two kits never merge on screen; mix it
    # slightly toward the primary hue so each family is still recognisable.
    secondary = shade(base, 0.28, mix_white=0.62)
    return {"primary": primary, "secondary": secondary}


# How each hand-tuned family becomes a real-engine team's parameters. The keys
# are the brain's `TRAINED_PARAMS` schema; `family` additionally selects the
# deception and learning machinery.
FAMILY_TUNING: dict[str, dict] = {
    "possession": dict(press_intensity=0.70, compactness=0.60, line_height=0.55,
                       tempo=0.85, directness=0.45, pass_power=0.40, risk=0.45),
    "high_press": dict(press_intensity=0.95, press_trigger=0.55, tackling=0.85,
                       counterpress=0.75, compactness=0.70, tempo=0.95,
                       directness=0.50, press_delay=0.0),
    "low_block": dict(press_intensity=0.25, press_trigger=0.30, compactness=0.95,
                      line_height=0.28, tempo=0.70, directness=0.30, risk=0.25,
                      pass_power=0.50),
    "counter": dict(press_intensity=0.45, line_height=0.25, compactness=0.75,
                    transition_speed=0.95, tempo=0.90, directness=0.75,
                    shoot_range=24.0, risk=0.60, max_pass_travel=42.0),
    "direct": dict(press_intensity=0.60, tempo=1.00, directness=0.95, risk=0.65,
                   width=0.85, shoot_range=22.0, pass_power=0.55,
                   max_pass_travel=45.0),
    "wall": dict(press_intensity=0.55, compactness=0.75, width=0.95, tempo=0.80,
                 directness=0.55, pass_power=0.35, risk=0.40),
    "tika": dict(press_intensity=0.70, compactness=0.80, tempo=0.75, directness=0.35,
                 pass_power=0.25, risk=0.35, support_distance=8.0),
    "overload": dict(press_intensity=0.60, compactness=0.65, tempo=0.85,
                     directness=0.60, width=0.90, risk=0.50),
    "physical": dict(max_pass_travel=36.0, press_intensity=1.00, tackling=1.00, counterpress=0.85,
                     tempo=0.85, compactness=0.70, noise=0.10),
    "gk_hell": dict(press_intensity=0.65, gk_aggression=1.00, gk_speed=1.00,
                    compactness=0.70, tempo=0.90, directness=0.55,
                    shoot_range=21.0),
    "chaos": dict(press_intensity=0.60, tempo=0.95, compactness=0.20, width=0.95,
                  noise=0.45, discipline=0.25, risk=0.85, directness=0.80),
}

# Difficulty bands map onto the same knobs the offline zoo uses. Harder means
# tighter, faster, less error-prone and more willing to shoot.
BAND_TWEAKS: dict[str, dict] = {
    "rookie": dict(noise=0.22, tempo=0.70, compactness=0.45, press_intensity=0.50,
                   shoot_range=12.0, gk_aggression=0.30, gk_speed=0.55),
    "solid": dict(noise=0.12, tempo=0.85, compactness=0.62, press_intensity=0.70,
                  shoot_range=17.0, gk_aggression=0.55, gk_speed=0.75),
    "pro": dict(noise=0.07, tempo=0.95, compactness=0.75, press_intensity=0.85,
                shoot_range=20.0, gk_aggression=0.75, gk_speed=0.90),
    "elite": dict(noise=0.03, tempo=1.00, compactness=0.85, press_intensity=0.95,
                  shoot_range=22.0, gk_aggression=0.90, gk_speed=1.00),
}

# Reaction lag, in seconds, added at the harder bands. A laggy side is a
# genuinely different opponent, not just a stricter one.
BAND_LAG: dict[str, float] = {"rookie": 0.30, "solid": 0.20, "pro": 0.10, "elite": 0.0}

TEAM_SUFFIX = {
    "possession": "Circulators FC",
    "high_press": "Red Shift FC",
    "low_block": "Blue Wall FC",
    "counter": "Vanguard FC",
    "direct": "Long Ball FC",
    "wall": "Touchline FC",
    "tika": "Pinball FC",
    "overload": "Overload Athletic",
    "physical": "Red Hand FC",
    "gk_hell": "Sweeper United",
}


def team_slug(family: str, band: str) -> str:
    return f"{family}-{band}"


def params_for(family: str, band: str) -> dict:
    """Merge family tuning with the difficulty band, band last so it wins."""
    params: dict = dict(FAMILY_TUNING[family])
    params["family"] = family
    params.update(BAND_TWEAKS[band])
    lag = BAND_LAG[band]
    if lag:
        params["press_delay"] = lag
    params["seed_salt"] = _salt(f"{family}-{band}")
    return params


def _salt(text: str) -> int:
    import hashlib

    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=4).digest(), "big")


def render_strategy(params: dict, team_name: str) -> str:
    """Inline the shared brain and append this team's parameters."""
    brain = BRAIN_SOURCE.read_text()
    header = f'"""Generated by scripts/build_opponent_teams.py -- do not edit by hand.\n\nTeam: {team_name}\nEdit src/opponents/engine_brain.py and regenerate instead.\n"""\n\n'
    # Python literals, not JSON: the learned presets carry `learns`/`pressure`
    # booleans, and JSON's `true`/`false` is a NameError on the very first line
    # of the module. The bot then never binds port 8080 and the engine reports
    # "did not become healthy within 15 seconds" for every seed and side.
    block = (
        f"\n\n# --- {team_name}: generated parameters -------------------------------------\n"
        f"PARAMS = {pprint.pformat(params, indent=4, sort_dicts=True)}\n"
        f"TRAINED_PARAMS = PARAMS\n"
    )
    # Drop the module docstring from the brain; the generated header replaces it.
    marker = '"""\n\nfrom __future__ import annotations'
    if marker in brain:
        brain = brain[brain.index("from __future__ import annotations"):]
    return header + brain.rstrip() + block


def render_server(team_id: str, team_name: str, params: dict) -> str:
    """Stamp this team's identity into the shared `server.py`.

    The worker reads the team's name and kit from `/v1/team`, so the copy of
    `server.py` has to be per-team. Everything below the `TEAM` literal is
    untouched, which keeps the HTTP surface identical to the starter.
    """
    source = SERVER_SOURCE.read_text()
    colors = team_colors(team_id, params)
    family = params.get("family", "possession")
    band = team_id.rsplit("-", 1)[-1]
    description = (
        f"Generated opponent: {family} family, {band} band. "
        f"behaviour from src/opponents/engine_brain.py"
    )
    block = (
        "TEAM = {\n"
        '    "protocolVersion": "1.0",\n'
        f"    \"name\": {json.dumps(team_name)},\n"
        f"    \"shortName\": {json.dumps(short_name(team_id))},\n"
        f"    \"colors\": {json.dumps(colors)},\n"
        '    "authors": ["My Team FC opponent generator"],\n'
        f"    \"description\": {json.dumps(description)},\n"
        "}"
    )
    start = source.index("TEAM = {")
    end = source.index("\n}", start) + 2
    header = (
        f"# Generated by scripts/build_opponent_teams.py -- do not edit by hand.\n"
        f"# Team: {team_name} ({team_id})\n"
    )
    return header + source[:start] + block + source[end:]


def render_dockerfile(team_id: str, team_name: str) -> str:
    return f"""FROM python:3.13-alpine
LABEL football-babylon.protocol-version="1.0" football-babylon.team="{team_id}"
WORKDIR /app
COPY models.py strategy.py server.py ./
USER 65532:65532
EXPOSE 8080
ENTRYPOINT ["python", "server.py"]
"""


def render_manifest(team_id: str, team_name: str, params: dict) -> str:
    return json.dumps(
        {"id": team_id, "name": team_name, "language": "python",
         "image": f"football-team-{team_id}:dev"},
        indent=2,
    ) + "\n"


def render_readme(team_id: str, team_name: str, params: dict) -> str:
    # `kit_index` is a generator-internal key for kit colours, not a brain
    # parameter, so it does not belong in the published tuning table.
    internal = {"family", "kit_index"}
    tuning = "\n".join(
        f"| `{k}` | {v} |" for k, v in sorted(params.items()) if k not in internal
    )
    return f"""# {team_name}

Generated opponent for the Football Babylon kit. **Do not edit by hand** -- run
`python scripts/build_opponent_teams.py` from `My Teams/deleri-fc/` after
changing `src/opponents/engine_brain.py` or the family's tuning table.

* team id: `{team_id}`
* family: `{params.get("family", "possession")}`
* behaviour: `src/opponents/engine_brain.py`, inlined into `strategy.py`

## Playing it

```sh
football-team build --team-dir opponents/{team_id}
football-team simulate --team-dir opponents/{team_id} --decisions 1800
```

The bundled `practice` command only offers its own compiled opponents, so use
`simulate --opponent-path` (or the arena) to see this team play.

## Parameters

| key | value |
| --- | --- |
{tuning}
"""


def build_targets() -> list[tuple[str, str, dict]]:
    """(team_id, display name, params) for every team to generate."""
    targets: list[tuple[str, str, dict]] = []
    for family, display in TEAM_SUFFIX.items():
        for band, _ in BANDS:
            team_id = team_slug(family, band)
            targets.append((team_id, f"{display} ({band})", params_for(family, band)))
    for index, (name, profile) in enumerate(LURE_PROFILES.items()):
        params = dict(FAMILY_TUNING["low_block"])
        params.update(BAND_TWEAKS["elite"])
        params.update(profile)
        params["family"] = "lure"
        params["deception"] = profile.get("deception", 0.8)
        params["seed_salt"] = _salt(f"lure-{name}")
        params["kit_index"] = index
        targets.append((f"lure-{name}", f"Serpent Lure FC ({name})", params))
    for index, (name, profile) in enumerate(LEARNED_PROFILES.items()):
        params = dict(FAMILY_TUNING["possession"])
        params.update(BAND_TWEAKS["pro"])
        params.update(profile)
        params["family"] = "learned"
        params["learns"] = True
        params["seed_salt"] = _salt(f"learned-{name}")
        params["kit_index"] = index
        targets.append((f"learned-{name}", f"Tabula RL ({name})", params))
    return targets


def files_for(team_id: str, team_name: str, params: dict) -> dict[str, str]:
    return {
        "football-team.json": render_manifest(team_id, team_name, params),
        "Dockerfile": render_dockerfile(team_id, team_name),
        "models.py": MODELS_SOURCE.read_text(),
        "server.py": render_server(team_id, team_name, params),
        "strategy.py": render_strategy(params, team_name),
        "README.md": render_readme(team_id, team_name, params),
    }


def compile_failures(root: pathlib.Path, targets: list[tuple[str, str, dict]]):
    """Byte-identical files can still be uncompilable Python.

    A JSON boolean inlined into the parameter block is byte-stable, so the
    drift check alone happily blessed six teams that crashed on import. Compile
    every generated module so that failure mode is caught before Docker runs.
    """
    failures = []
    for team_id, _name, _params in targets:
        for filename in ("strategy.py", "server.py", "models.py"):
            path = root / team_id / filename
            if not path.exists():
                continue
            try:
                compile(path.read_text(), str(path), "exec")
            except SyntaxError as exc:
                failures.append((path, exc.msg))
    return failures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="verify only, write nothing")
    ap.add_argument("--only", nargs="*", default=None, help="limit to these team ids")
    ap.add_argument("--root", type=pathlib.Path, default=OPPONENTS_DIR)
    args = ap.parse_args()

    targets = build_targets()
    if args.only:
        wanted = set(args.only)
        targets = [t for t in targets if t[0] in wanted]
    if not targets:
        print("no teams matched --only", file=sys.stderr)
        return 2

    stale: list[str] = []
    written = 0
    for team_id, team_name, params in targets:
        files = files_for(team_id, team_name, params)
        directory = args.root / team_id
        for filename, content in files.items():
            path = directory / filename
            current = path.read_text() if path.exists() else None
            if current == content:
                continue
            if args.check:
                stale.append(str(path))
                continue
            directory.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
            written += 1

    if args.check:
        broken = compile_failures(args.root, targets)
        if stale or broken:
            if stale:
                print(f"STALE: {len(stale)} generated file(s) differ:")
                for path in stale[:20]:
                    print(f"  {path}")
                print("run: python scripts/build_opponent_teams.py")
            for path, error in broken:
                print(f"  DOES NOT COMPILE: {path}: {error}")
            return 1
        print(f"up to date: {len(targets)} teams")
        return 0

    print(f"{len(targets)} opponent teams in {args.root}")
    print(f"  {written} file(s) written")
    for team_id, team_name, _ in targets:
        print(f"  {team_id:24s} {team_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
