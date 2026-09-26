#!/usr/bin/env python3
"""Opponent arena: run the team against the whole opponent pool and rank it.

This is the answer to "how hard is this AI, really?". It runs a fixed grid of
seeds x opponents through the internal `LightEngine` and reports, per opponent:
goals, shots, possession, and the aggregate. Two aggregates matter:

* **GF/GA** — can the team score against it and does it concede?
* **exploit** — the worst-case goal difference. An opponent the team beats 8-0
  on one seed is a different problem from one it draws with.

Usage::

    python scripts/opponent_arena.py                       # full pool
    python scripts/opponent_arena.py --tiers elite         # one difficulty
    python scripts/opponent_arena.py --family lure         # one family
    python scripts/opponent_arena.py --seeds 8 --decisions 1800
    python scripts/opponent_arena.py --json out/arena.json

Note on interpreting numbers: `LightEngine` is a coarse model. It is the right
tool for *relative* comparison and for regression gating, and the wrong tool
for absolute difficulty. Every number here is also produced by the real engine
via `scripts/run_authoritative.py`.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import statistics
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]  # my-team-fc root
sys.path.insert(0, str(ROOT))

from src.opponents import REGISTRY  # noqa: E402
from src.runtime import RuntimeManager  # noqa: E402
from src.sim import OPPONENTS, play_match  # noqa: E402

BASE_SEED = 20240917


def install(names: list[str]) -> None:
    """Make the pool reachable through the simulator's own registry."""
    for name in names:
        OPPONENTS[name] = REGISTRY[name]


def play_one(name: str, seed: int, decisions: int) -> dict:
    return play_match(
        RuntimeManager(), name, rng=random.Random(seed), decisions=decisions
    ).summary()


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {}
    goals_for = [r["score_us"] for r in rows]
    goals_against = [r["score_them"] for r in rows]
    diffs = [r["goal_diff"] for r in rows]
    poss = [r["possession%"] for r in rows]
    shots = sum(r["shots"] for r in rows)
    conceded = sum(r["shots_conceded"] for r in rows)
    return {
        "matches": len(rows),
        "gf": sum(goals_for),
        "ga": sum(goals_against),
        "wins": sum(1 for d in diffs if d > 0),
        "draws": sum(1 for d in diffs if d == 0),
        "losses": sum(1 for d in diffs if d < 0),
        "mean_diff": round(statistics.fmean(diffs), 2),
        # The tail matters more than the mean: one 7-0 is a coaching note.
        "worst_diff": min(diffs),
        "best_diff": max(diffs),
        "mean_possession": round(statistics.fmean(poss), 1),
        "shots": shots,
        "shots_conceded": conceded,
        "clean_sheets": sum(1 for g in goals_against if g == 0),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=4, help="matches per opponent")
    ap.add_argument("--decisions", type=int, default=1200, help="ticks per match")
    ap.add_argument("--tiers", nargs="*", default=None, help="e.g. elite pro")
    ap.add_argument("--family", default=None, help="substring filter on the name")
    ap.add_argument("--opponent", nargs="*", default=None, help="exact names")
    ap.add_argument("--learned", action="store_true", help="only ML opponents")
    ap.add_argument("--json", type=pathlib.Path, default=None)
    args = ap.parse_args()

    names = sorted(REGISTRY)
    if args.opponent:
        names = [n for n in names if n in set(args.opponent)]
    if args.family:
        names = [n for n in names if args.family in n]
    if args.tiers:
        names = [n for n in names if any(t in n for t in args.tiers)]
    if args.learned:
        names = [n for n in names if n.startswith("learned-")]
    if not names:
        print("no opponents matched the filter", file=sys.stderr)
        return 2

    install(names)
    print(f"arena: {len(names)} opponents x {args.seeds} seeds "
          f"x {args.decisions} decisions\n")

    started = time.time()
    results: dict[str, dict] = {}
    for index, name in enumerate(names, 1):
        rows = [
            play_one(name, BASE_SEED + 1009 * s, args.decisions)
            for s in range(args.seeds)
        ]
        agg = aggregate(rows)
        results[name] = agg
        scores = " ".join(f"{r['score_us']}-{r['score_them']}" for r in rows)
        print(
            f"{index:3d}/{len(names)} {name:24s} {scores:<24s}"
            f" diff {agg['mean_diff']:+6.2f}  worst {agg['worst_diff']:+3d}"
            f"  poss {agg['mean_possession']:4.1f}%"
            f"  shots {agg['shots']:3d}-{agg['shots_conceded']:<3d}"
            f"  CS {agg['clean_sheets']}/{agg['matches']}"
        )

    print(f"\nelapsed {time.time() - started:.1f}s")
    ranked = sorted(results.items(), key=lambda kv: (kv[1]["mean_diff"], -kv[1]["worst_diff"]))
    print("\nhardest for us (lowest mean goal difference):")
    for name, agg in ranked[:8]:
        print(f"  {name:24s} mean {agg['mean_diff']:+6.2f}  worst {agg['worst_diff']:+3d}"
              f"  GA {agg['ga']}")
    print("\nweakest for us (highest mean goal difference):")
    for name, agg in ranked[-8:][::-1]:
        print(f"  {name:24s} mean {agg['mean_diff']:+6.2f}  worst {agg['worst_diff']:+3d}"
              f"  GF {agg['gf']}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "seeds": args.seeds,
                    "decisions": args.decisions,
                    "base_seed": BASE_SEED,
                    "results": results,
                },
                indent=2,
                sort_keys=True,
            )
        )
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
