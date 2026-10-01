#!/usr/bin/env python3
"""Run the team against the generated real-engine opponents and report.

`scripts/opponent_arena.py` is fast but runs on `LightEngine`, the internal
coarse model. This is the authoritative pass: it builds each generated team and
plays it with the real Babylon worker in Docker, which is the only place the
difficulty numbers mean anything.

Cost note: each distinct opponent is a Docker build plus a container start per
game, so this is minutes rather than seconds. Use `--games 1` for a smoke test
and `--only` to check a subset.

Why 24 games by default: the engine is fully deterministic per seed, and a
60 s match against a solid generated side is always 1-0 or 0-1. With only a
handful of games the score is decided by which seeds happened to produce the
single goal, and whole families come out bit-identical. Two games is a smoke
test, not a measurement.

Usage::

    python scripts/run_authoritative.py --only high_press-elite lure-serpent
    python scripts/run_authoritative.py --games 4 --duration 90 --json out.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]  # deleri-fc root
REPO = ROOT.parent.parent  # repo root
CLI = REPO / "football-team"
OPPONENTS = ROOT / "training" / "opponents"

# `simulate` prints its own noise; these are the lines worth keeping.
RESULT_RE = re.compile(r"Goals: (.+?)\s+Average: (\S+)\s+Difference: ([+-]?\d+)")
SUMMARY_RE = re.compile(r"Win rate: ([\d.]+) %")
POSSESSION_RE = re.compile(r"Possession: ([\d.]+)%")
SHOTS_RE = re.compile(r"Shots: (\d+)")
CLEAN_RE = re.compile(r"Clean sheets: ([\d.]+) %")
MISSED_RE = re.compile(r"Missed decisions: (\d+)")
GAME_RE = re.compile(r"^\[candidate \d+/\d+\] (\d+)-(\d+) (win|loss|draw)", re.M)


def run_one(name: str, games: int, duration: int, sides: str) -> dict:
    cmd = [
        str(CLI), "simulate",
        "--path", str(ROOT),
        "--opponent-path", str(OPPONENTS / name),
        "--games", str(games),
        "--duration", str(duration),
        "--sides", sides,
    ]
    started = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    out = proc.stdout
    record: dict = {"opponent": name, "games": games, "returncode": proc.returncode}
    if proc.returncode != 0:
        record["error"] = (proc.stderr or out).strip()[-800:]
        return record
    m = RESULT_RE.search(out)
    if m:
        gf, ga = m.group(1).split("-")
        record["goals"] = f"{gf}-{ga}"
        record["goal_diff"] = int(m.group(3))
        # Keep both rates separately so a reader cannot confuse them: "scored"
        # is what we put in, "conceded" is what we let in. Win rate saturates
        # against a solid generated side (every 60 s match is 1-0 or 0-1), so
        # the scoreline is decided by which seed produced the single goal.
        avg_us, avg_them = (float(v) for v in m.group(2).split("-"))
        record["scored_per_game"] = round(avg_us, 3)
        record["conceded_per_game"] = round(avg_them, 3)
    # Keep the per-game scorelines. They expose the degenerate 1-0/0-1-only
    # distribution that makes a short aggregate run unrankable.
    scorelines = [f"{a}-{b}" for a, b, _ in GAME_RE.findall(out)]
    if scorelines:
        record["scorelines"] = scorelines
        distinct = {}
        for line in scorelines:
            distinct[line] = distinct.get(line, 0) + 1
        record["scoreline_histogram"] = distinct
    for key, pattern in (
        ("win_rate", SUMMARY_RE), ("possession", POSSESSION_RE),
        ("shots", SHOTS_RE), ("clean_sheets", CLEAN_RE), ("missed", MISSED_RE),
    ):
        found = pattern.search(out)
        if found:
            record[key] = float(found.group(1))
    record["wall_time"] = round(time.time() - started, 1)
    return record


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=24)
    ap.add_argument("--duration", type=int, default=60)
    ap.add_argument("--sides", default="both", choices=["home", "away", "both"])
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=0, help="cap the number of opponents")
    ap.add_argument("--json", type=pathlib.Path, default=None)
    args = ap.parse_args()

    if not CLI.exists():
        print(f"CLI not found at {CLI}", file=sys.stderr)
        return 2
    names = sorted(
        p.name for p in OPPONENTS.iterdir()
        if p.is_dir() and (p / "football-team.json").exists()
    )
    if args.only:
        wanted = set(args.only)
        names = [n for n in names if n in wanted]
    if args.limit:
        names = names[: args.limit]
    if not names:
        print("no opponents found; run scripts/build_opponent_teams.py", file=sys.stderr)
        return 2

    print(f"authoritative sweep: {len(names)} opponents x {args.games} games "
          f"x {args.duration}s\n")
    results = []
    failed = []
    started = time.time()
    for index, name in enumerate(names, 1):
        record = run_one(name, args.games, args.duration, args.sides)
        results.append(record)
        if record.get("returncode") != 0:
            failed.append(name)
            print(f"{index:3d}/{len(names)} {name:26s} FAILED")
            print("      " + record.get("error", "").replace("\n", "\n      ")[:600])
        else:
            print(
                f"{index:3d}/{len(names)} {name:26s} {record.get('goals','?'):>7s}"
                f"  diff {record.get('goal_diff', 0):+3d}"
                f"  win {record.get('win_rate', 0):5.1f}%"
                f"  sc {record.get('scored_per_game', 0):.2f}"
                f"/ca {record.get('conceded_per_game', 0):.2f}"
                f"  poss {record.get('possession', 0):5.1f}%"
                f"  shots {int(record.get('shots', 0)):3d}"
                f"  CS {record.get('clean_sheets', 0):5.1f}%"
                f"  missed {int(record.get('missed', 0))}"
                f"  {record.get('wall_time', 0):5.1f}s"
            )
        sys.stdout.flush()

    print(f"\nelapsed {time.time() - started:.1f}s")
    ok = [r for r in results if r.get("returncode") == 0]
    if ok:
        # Rank by what we concede, not by goal difference. Against a solid
        # generated side every 60 s match is 1-0 or 0-1, so goal difference is
        # decided by which of the deterministic seeds produced the one goal and
        # is identical across whole families. Conceded rate does separate a
        # generated rookie (~0.00) from a solid/pro/elite side (~0.42), but read
        # it carefully: a low rate also means a weak attack, so the bundled
        # `reference` also sits near 0.00 because it rarely tests us. Within
        # solid/pro/elite the ladder is flat and cannot be ranked at all.
        ranked = sorted(ok, key=lambda r: -r.get("conceded_per_game", 0.0))
        print("\nhardest for us (most goals we concede per game):")
        for r in ranked[:10]:
            print(f"  {r['opponent']:26s} {r.get('goals','?'):>7s}"
                  f"  sc/ca {r.get('scored_per_game', 0):.2f}/"
                  f"{r.get('conceded_per_game', 0):.2f}"
                  f"  win {r.get('win_rate', 0):5.1f}%"
                  f"  poss {r.get('possession', 0):5.1f}%")
        print("\nweakest for us:")
        for r in ranked[-10:][::-1]:
            print(f"  {r['opponent']:26s} {r.get('goals','?'):>7s}"
                  f"  sc/ca {r.get('scored_per_game', 0):.2f}/"
                  f"{r.get('conceded_per_game', 0):.2f}"
                  f"  win {r.get('win_rate', 0):5.1f}%"
                  f"  poss {r.get('possession', 0):5.1f}%")

        # Flag saturation so a flat report is never mistaken for a ranking.
        rates = {r.get("conceded_per_game") for r in ok}
        if len(rates) == 1:
            print(
                f"\nWARNING: all {len(ok)} opponents concede an identical "
                f"{rates.pop():.2f}/game. This run cannot rank them; the "
                "opponents are the same strength. Widen --games, or compare "
                "possession/shots, which do differ."
            )
    if failed:
        print(f"\n{len(failed)} opponent(s) failed: {', '.join(failed)}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(
            {"games": args.games, "duration": args.duration, "sides": args.sides,
             "results": results}, indent=2, sort_keys=True))
        print(f"\nwrote {args.json}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
