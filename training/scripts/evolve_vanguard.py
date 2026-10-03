#!/usr/bin/env python3
"""Evolve tactical genome parameters against the real Vanguard engine team.

Each individual is installed in the runtime policy artifact, built with the
Team Practice Lab's CLI, and evaluated separately at home and away against
Vanguard Pro and Elite. Selection requires wins across all four buckets.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parent.parent
CLI = REPO / "football-team"
OPPONENTS = {
    "counter-pro": ROOT / "training" / "opponents" / "counter-pro",
    "counter-elite": ROOT / "training" / "opponents" / "counter-elite",
}
POLICY = ROOT / "artifacts" / "policies" / "distilled_policy.json"
sys.path.insert(0, str(ROOT))
from src.config import GENOME_KEYS, GENOME_RANGES, default_genome, genome_hash  # noqa: E402
from src.training import EXPERIMENT_LOG, distill, log_experiment  # noqa: E402

SUMMARY_RE = re.compile(
    r"Matches:\s+(\d+)/(\d+) completed.*?W/D/L:\s+(\d+)/(\d+)/(\d+)"
    r".*?Goals:\s+(\d+)-(\d+).*?Clean sheets:\s+([\d.]+)\s*%",
    re.S,
)
MUTABLE = (
    "press_intensity", "press_trigger_threshold", "width", "depth",
    "verticality", "risk", "counterpress_intensity", "compactness",
    "passing_risk", "shooting_threshold", "transition_speed",
    "support_distance", "defensive_line", "wall_usage", "wall_pass_threshold",
)
PROBES = (
    {"defensive_line": 12.0, "depth": 0.30, "verticality": 0.75, "width": 0.65,
     "support_distance": 6.0, "risk": 0.35, "passing_risk": 0.10,
     "shooting_threshold": 0.30, "press_intensity": 0.40,
     "press_trigger_threshold": 0.50, "counterpress_intensity": 0.20,
     "compactness": 0.90, "transition_speed": 0.70, "wall_usage": 0.40},
    {"defensive_line": 14.0, "depth": 0.25, "verticality": 0.90, "width": 0.55,
     "support_distance": 5.0, "risk": 0.55, "passing_risk": 0.05,
     "shooting_threshold": 0.30, "press_intensity": 0.80,
     "press_trigger_threshold": 0.20, "counterpress_intensity": 0.30,
     "compactness": 0.90, "transition_speed": 0.90, "wall_usage": 0.50},
    {"defensive_line": 12.0, "depth": 0.20, "verticality": 1.00, "width": 0.60,
     "support_distance": 5.0, "risk": 0.50, "passing_risk": 0.0,
     "shooting_threshold": 0.30, "press_intensity": 0.20,
     "press_trigger_threshold": 0.70, "counterpress_intensity": 0.15,
     "compactness": 0.95, "transition_speed": 0.80, "wall_usage": 0.40},
    {"defensive_line": 16.0, "depth": 0.45, "verticality": 0.85, "width": 0.70,
     "support_distance": 8.0, "risk": 0.60, "passing_risk": 0.10,
     "shooting_threshold": 0.30, "press_intensity": 0.50,
     "press_trigger_threshold": 0.40, "counterpress_intensity": 0.40,
     "compactness": 0.80, "transition_speed": 0.80, "wall_usage": 0.60},
    {"defensive_line": 14.0, "depth": 1.00, "verticality": 1.00, "width": 0.90,
     "support_distance": 12.0, "risk": 0.80, "passing_risk": 0.90,
     "shooting_threshold": 0.30, "press_intensity": 0.60,
     "press_trigger_threshold": 0.35, "counterpress_intensity": 0.50,
     "compactness": 0.45, "transition_speed": 1.00, "wall_usage": 0.30},
    {"defensive_line": 12.0, "depth": 0.85, "verticality": 0.95, "width": 1.00,
     "support_distance": 10.0, "risk": 0.70, "passing_risk": 0.55,
     "shooting_threshold": 0.30, "press_intensity": 0.45,
     "press_trigger_threshold": 0.55, "counterpress_intensity": 0.35,
     "compactness": 0.85, "transition_speed": 0.95, "wall_usage": 0.55},
    {"defensive_line": 12.0, "depth": 0.95, "verticality": 1.00, "width": 0.30,
     "support_distance": 2.0, "risk": 1.00, "passing_risk": 0.05,
     "shooting_threshold": 0.30, "press_intensity": 0.55,
     "press_trigger_threshold": 0.25, "counterpress_intensity": 0.85,
     "compactness": 0.60, "transition_speed": 1.00, "wall_usage": 0.15},
    {"defensive_line": 16.0, "depth": 0.70, "verticality": 0.75, "width": 0.95,
     "support_distance": 7.0, "risk": 0.85, "passing_risk": 0.75,
     "shooting_threshold": 0.35, "press_intensity": 0.70,
     "press_trigger_threshold": 0.30, "counterpress_intensity": 0.65,
     "compactness": 0.55, "transition_speed": 0.90, "wall_usage": 0.80},
)


def evaluate(
    genome: dict[str, float], seeds: tuple[str, ...], games: int, duration: int,
    opponents: tuple[str, ...], sides: str,
) -> dict:
    distill(genome, POLICY)
    build = subprocess.run([str(CLI), "build", "--path", str(ROOT)], capture_output=True, text=True)
    if build.returncode:
        raise RuntimeError((build.stdout + build.stderr)[-4000:])
    expected_sides = ("home", "away") if sides == "both" else (sides,)
    blank = lambda: {"matches": 0, "wins": 0, "draws": 0, "goalsFor": 0,
                     "goalsAgainst": 0, "cleanSheets": 0, "normalTimeWins": 0,
                     "openPlayGoalsFor": 0, "openPlayGoalsAgainst": 0,
                     "slaps": 0, "knockdowns": 0}
    by_opponent: dict = {
        opponent: {"bySide": {side: blank() for side in expected_sides}}
        for opponent in opponents
    }
    matches = []
    for seed in seeds:
        for opponent in opponents:
            run = subprocess.run(
                [str(CLI), "simulate", "--path", str(ROOT), "--opponent-path", str(OPPONENTS[opponent]),
                 "--duration", str(duration), "--games", str(games), "--seed-prefix", seed,
                 "--sides", sides, "--jobs", "1", "--skip-build", "--format", "json"],
                capture_output=True, text=True,
            )
            if run.returncode:
                raise RuntimeError((run.stdout + run.stderr)[-4000:])
            try:
                payload = json.loads(run.stdout)
            except json.JSONDecodeError:
                payload = {}
            parsed = [
                (int(match["teamScore"]), int(match["opponentScore"]),
                 str(match["outcome"]), str(match.get("side", "unknown")),
                 str(match.get("resultReason", "unknown")), seed,
                 int((match.get("teamStatistics") or {}).get("slaps", 0)),
                 int((match.get("teamStatistics") or {}).get("knockdowns", 0)))
                for match in payload.get("results", [])
                if match.get("type") == "simulation-match"
            ]
            if not parsed:
                summary = SUMMARY_RE.search(run.stdout)
                if not summary:
                    raise RuntimeError(f"No {opponent} results parsed:\n" + run.stdout[-4000:])
                raise RuntimeError("Gauntlet requires per-fixture JSON to score side buckets")
            for gf, ga, outcome, side, reason, _, slaps, knockdowns in parsed:
                bucket = by_opponent[opponent]["bySide"].get(side)
                if bucket is None:
                    continue
                bucket["matches"] += 1
                bucket["wins"] += outcome == "win"
                bucket["draws"] += outcome == "draw"
                bucket["goalsFor"] += gf
                bucket["goalsAgainst"] += ga
                bucket["cleanSheets"] += ga == 0
                bucket["slaps"] += slaps
                bucket["knockdowns"] += knockdowns
                if reason == "normalTime":
                    bucket["normalTimeWins"] += outcome == "win"
                    bucket["openPlayGoalsFor"] += gf
                    bucket["openPlayGoalsAgainst"] += ga
                matches.append((gf, ga, outcome, side, reason, opponent, seed))

    for opponent in opponents:
        side_rows = list(by_opponent[opponent]["bySide"].values())
        by_opponent[opponent].update({
            key: sum(row[key] for row in side_rows)
            for key in ("matches", "wins", "draws", "goalsFor", "goalsAgainst",
                        "cleanSheets", "normalTimeWins", "openPlayGoalsFor", "openPlayGoalsAgainst",
                        "slaps", "knockdowns")
        })
    scored = sum(v["goalsFor"] for v in by_opponent.values())
    conceded = sum(v["goalsAgainst"] for v in by_opponent.values())
    clean_sheets = sum(v["cleanSheets"] for v in by_opponent.values())
    wins = sum(v["wins"] for v in by_opponent.values())
    draws = sum(v["draws"] for v in by_opponent.values())
    normal_time_wins = sum(v["normalTimeWins"] for v in by_opponent.values())
    open_play_goals = sum(v["openPlayGoalsFor"] for v in by_opponent.values())
    open_play_goals_against = sum(v["openPlayGoalsAgainst"] for v in by_opponent.values())
    slaps = sum(v["slaps"] for v in by_opponent.values())
    knockdowns = sum(v["knockdowns"] for v in by_opponent.values())
    match_count = sum(v["matches"] for v in by_opponent.values())
    # Require victories in each opponent × side bucket, so a good away record
    # cannot hide home losses (or an Elite win hide a Pro loss).
    floor_wins = min(
        (v["bySide"].get(side, {}).get("wins", 0)
         for v in by_opponent.values() for side in expected_sides),
        default=0,
    )
    fitness = (
        50000 * floor_wins + 500 * wins + 10000 * normal_time_wins
        + 5000 * open_play_goals + 100 * draws + 25 * clean_sheets
        + 50 * knockdowns + 5 * slaps
        + 10 * scored - 30 * conceded - 100 * open_play_goals_against
    )
    return {
        "genomeHash": genome_hash(genome), "fitness": fitness, "goalsFor": scored,
        "goalsAgainst": conceded, "cleanSheets": clean_sheets, "wins": wins,
        "draws": draws, "normalTimeWins": normal_time_wins,
        "openPlayGoals": open_play_goals,
        "openPlayGoalsAgainst": open_play_goals_against,
        "slaps": slaps,
        "knockdowns": knockdowns, "matches": match_count,
        "scorelines": [f"{opponent} {side} {a}-{b}:{result} ({reason}; {seed})"
                       for a, b, result, side, reason, opponent, seed in matches],
        "winsFloor": floor_wins, "byOpponent": by_opponent, "genome": dict(genome),
    }


def mutate(parent: dict[str, float], rng: random.Random, sigma: float) -> dict[str, float]:
    child = dict(parent)
    for key in MUTABLE:
        if rng.random() < 0.58:
            lo, hi = GENOME_RANGES[key]
            scale = 1.0 if key in ("support_distance", "defensive_line") else hi - lo
            child[key] = max(lo, min(hi, child[key] + rng.gauss(0.0, sigma * scale)))
    return child


def crossover(a: dict[str, float], b: dict[str, float], rng: random.Random) -> dict[str, float]:
    """Uniform crossover for a diverse winner-takes-all tactical gauntlet."""
    return {key: a[key] if rng.random() < 0.5 else b[key] for key in GENOME_KEYS}


def random_genome(rng: random.Random) -> dict[str, float]:
    return {key: rng.uniform(*GENOME_RANGES[key]) for key in GENOME_KEYS}


def recorded_genome(target_hash: str) -> dict[str, float] | None:
    """Find a previously measured genome by hash in the provenance log."""
    if not EXPERIMENT_LOG.exists():
        return None
    for line in reversed(EXPERIMENT_LOG.read_text(encoding="utf-8").splitlines()):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        candidates = [record.get("champion", {})]
        candidates.extend(record.get("evaluations", []))
        for candidate in candidates:
            if candidate.get("genomeHash") == target_hash and isinstance(candidate.get("genome"), dict):
                genome = default_genome()
                genome.update({k: float(v) for k, v in candidate["genome"].items() if k in genome})
                return genome
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=6302026)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--population", type=int, default=7)
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument("--seed-batches", type=int, default=1,
                        help="independent fixture-seed prefixes per genome")
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--seed-prefix", default="vanguard-evolve-2026")
    parser.add_argument("--opponents", choices=("pro", "elite", "both"), default="both")
    parser.add_argument("--sides", choices=("home", "away", "both"), default="both")
    parser.add_argument("--restore-hash", help="restore a measured genome from the experiment log and exit")
    args = parser.parse_args()
    if args.restore_hash:
        genome = recorded_genome(args.restore_hash)
        if genome is None:
            parser.error(f"genome {args.restore_hash} not found in {EXPERIMENT_LOG}")
        distill(genome, POLICY)
        print(f"restored {args.restore_hash} to {POLICY}")
        return 0
    rng = random.Random(args.seed)
    seeds = tuple(f"{args.seed_prefix}-batch-{i:02d}" for i in range(max(1, args.seed_batches)))
    opponents = ({"pro": ("counter-pro",), "elite": ("counter-elite",),
                  "both": ("counter-pro", "counter-elite")})[args.opponents]

    try:
        base_data = json.loads(POLICY.read_text(encoding="utf-8"))
        base = default_genome()
        for key, value in base_data.get("genome", {}).items():
            if key in base:
                base[key] = float(value)
        population = [base]
        # Seed the bracket with prior measured winners from the experiment log,
        # not only mutations of the current champion.
        if EXPERIMENT_LOG.exists():
            for line in reversed(EXPERIMENT_LOG.read_text(encoding="utf-8").splitlines()):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("kind") != "vanguard_real_engine_evolution":
                    continue
                for candidate in [record.get("champion", {}), *record.get("evaluations", [])]:
                    genome = candidate.get("genome")
                    if not isinstance(genome, dict):
                        continue
                    if any(genome_hash(existing) == candidate.get("genomeHash") for existing in population):
                        continue
                    restored = default_genome()
                    restored.update({k: float(v) for k, v in genome.items() if k in restored})
                    population.append(restored)
                    if len(population) >= args.population:
                        break
                if len(population) >= args.population:
                    break
        # Hand-designed tactical archetypes fill the remaining early bracket
        # slots before the mutation/crossover generations begin.
        for probe in PROBES:
            if len(population) >= args.population:
                break
            genome = dict(base)
            genome.update(probe)
            if all(genome_hash(existing) != genome_hash(genome) for existing in population):
                population.append(genome)
        population = population[:args.population]
        # Deterministic broad mutations fill larger requested populations.
        while len(population) < args.population:
            population.append(mutate(base, rng, 0.32))

        all_rows = []
        champion = None
        for generation in range(args.generations):
            rows = []
            for index, genome in enumerate(population):
                # All genomes face exactly the same multi-prefix gauntlet.
                row = evaluate(genome, seeds, args.games, args.duration, opponents, args.sides)
                row.update(generation=generation, individual=index, fixtureSeeds=list(seeds))
                rows.append(row)
                all_rows.append(row)
                print(f"g{generation} p{index}: W{row['wins']} D{row['draws']} "
                      f"{row['goalsFor']}-{row['goalsAgainst']} "
                      f"CS {row['cleanSheets']}/{row['matches']} fit {row['fitness']} "
                      f"floorW {row['winsFloor']} {row['genomeHash']} {row['byOpponent']}", flush=True)
            rows.sort(key=lambda r: (r["fitness"], r["goalsFor"], -r["goalsAgainst"]), reverse=True)
            if champion is None or (rows[0]["fitness"], rows[0]["goalsFor"], -rows[0]["goalsAgainst"]) > (
                champion["fitness"], champion["goalsFor"], -champion["goalsAgainst"]
            ):
                champion = rows[0]
            if generation + 1 < args.generations:
                survivors = [r["genome"] for r in rows[:min(4, len(rows))]]
                population = [dict(g) for g in survivors[:2]]
                while len(population) < args.population:
                    if rng.random() < 0.15:
                        population.append(random_genome(rng))
                        continue
                    a = survivors[rng.randrange(len(survivors))]
                    b = survivors[rng.randrange(len(survivors))]
                    child = crossover(a, b, rng)
                    population.append(mutate(child, rng, 0.22 if generation == 0 else 0.14))

        assert champion is not None
        # Restore the winning genome as the active runtime artifact.
        distill(champion["genome"], POLICY)
        log_experiment(
            "vanguard_real_engine_evolution", seed=args.seed, seedPrefixes=list(seeds),
            generations=args.generations, population=args.population, games=args.games,
            opponents=list(opponents), sides=args.sides,
            duration=args.duration, objective="wins_on_both_sides_against_each_vanguard_tier",
            evaluations=all_rows, champion=champion,
        )
        print("BEST", json.dumps({k: champion[k] for k in
              ("genomeHash", "goalsFor", "goalsAgainst", "cleanSheets", "matches", "scorelines")},
              sort_keys=True))
        print("BEST_GENOME", json.dumps(champion["genome"], sort_keys=True))
        return 0
    except Exception as exc:
        print(f"evolution failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
