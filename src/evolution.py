"""Evolutionary optimisation of the tactical genome (AISTRATEGI §34, §63-65).

Search over the genome schema in `config.GENOME_KEYS` / `GENOME_RANGES` using
tournament selection, elitism, uniform crossover and gaussian mutation. Fitness
comes from simulated matches (`sim.play_match`) against the scripted opponent
archetypes, giving an offline, deterministic proxy for the engine.

Runtime impact: this module is never imported by the decision path.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Sequence

from .config import GENOME_KEYS, GENOME_RANGES, genome_hash, make_genome
from .runtime import RuntimeManager
from .scenarios import ScenarioResult, evaluate_scenarios
from .sim import SimResult, play_match

PoolEntry = str | tuple[str, dict[str, float]]


# Tactical archetypes (§37 diversity preservation). Keys match `make_genome`'s
# `tactics.json` style block so population seeding stays engine-aligned.
STYLE_TEMPLATES: dict[str, dict] = {
    "possession": {"pressing": 35, "defensiveDepth": 40, "directness": 35, "shootingDistanceMeters": 14},
    "pressing": {"pressing": 90, "defensiveDepth": 70, "directness": 60, "shootingDistanceMeters": 17},
    "direct": {"pressing": 45, "defensiveDepth": 55, "directness": 90, "shootingDistanceMeters": 18},
    "defensive": {"pressing": 40, "defensiveDepth": 20, "directness": 35, "shootingDistanceMeters": 12},
    "wall": {"pressing": 50, "defensiveDepth": 45, "directness": 55, "shootingDistanceMeters": 16},
    "balanced": {"pressing": 55, "defensiveDepth": 45, "directness": 55, "shootingDistanceMeters": 16},
}


def arch_genome(style: str) -> dict[str, float]:
    """Genome for a named tactical archetype (AISTRATEGI §37)."""
    g = make_genome(STYLE_TEMPLATES.get(style, {}))
    if style == "wall":
        g["wall_usage"] = 0.85
        g["wall_pass_threshold"] = 0.7
        g["wall_shot_threshold"] = 0.8
    if style == "direct":
        g["transition_speed"] = 0.8
    if style == "possession":
        g["support_distance"] = 9.0
        g["passing_risk"] = 0.25
    if style == "defensive":
        g["defensive_line"] = 18.0
    return g


def seeded_population(
    rng: random.Random,
    size: int,
    styles: list[str] | None = None,
    seed_genome: dict[str, float] | None = None,
) -> list[dict[str, float]]:
    """Initial population that guarantees every requested archetype survives
    and otherwise stays diverse (no immediate convergence, §37)."""
    styles = styles or list(STYLE_TEMPLATES)
    pop: list[dict[str, float]] = []
    n_styles = len(styles)
    for i in range(size):
        if seed_genome is not None and rng.random() < 0.5:
            base = dict(seed_genome)
        else:
            base = arch_genome(styles[i % n_styles])
        if seed_genome is None and i < n_styles:
            pop.append(base)  # keep each pure archetype unchanged
        else:
            pop.append(mutate(base, rng, rate=0.3, sigma=0.06))
    return pop


def random_genome(rng: random.Random) -> dict[str, float]:
    """A genotype with every gene sampled inside its legal range."""
    g = {}
    for key in GENOME_KEYS:
        lo, hi = GENOME_RANGES[key]
        g[key] = rng.uniform(lo, hi)
    return g


def mutate(genome: dict[str, float], rng: random.Random, rate: float = 0.25, sigma: float = 0.08) -> dict[str, float]:
    """Gaussian mutation; each gene flips with probability `rate`."""
    out = {}
    for key in GENOME_KEYS:
        lo, hi = GENOME_RANGES[key]
        value = genome[key]
        if rng.random() < rate:
            value += rng.gauss(0.0, sigma) * (hi - lo)
        out[key] = max(lo, min(hi, value))
    return out


def crossover(a: dict[str, float], b: dict[str, float], rng: random.Random) -> dict[str, float]:
    """Uniform crossover: inherit each gene from a random parent."""
    return {key: (a[key] if rng.random() < 0.5 else b[key]) for key in GENOME_KEYS}


def tournament(population: list[dict[str, float]], fitnesses: list[float], k: int, rng: random.Random) -> dict[str, float]:
    """Return the fittest individual from `k` random draws."""
    best: dict[str, float] | None = None
    best_fit = -1e18
    for _ in range(k):
        idx = rng.randrange(len(population))
        if fitnesses[idx] > best_fit:
            best_fit = fitnesses[idx]
            best = population[idx]
    assert best is not None
    return best


def evaluate_genome(
    genome: dict[str, float],
    rng: random.Random,
    opponent: str = "possession",
    decisions: int = 400,
    weights: dict[str, float] | None = None,
    opponents: Sequence[PoolEntry] | None = None,
) -> tuple[SimResult, float]:
    """Play matches with `genome` controlling our team.

    With a single opponent this is the original one-match evaluation. With
    `opponents` given, the genome is scored across the whole pool so a
    strategy that only beats one archetype cannot rank highly (§35
    robustness). The pool accepts `"champion"`-tuple entries to test against a
    past/current champion genome head-to-head (§36, §33).

    A fresh RuntimeManager per evaluation keeps contexts from sharing state
    across runs (AGENTS.md §5). `rng` must be externally seeded for
    determinism.
    """
    pool = list(opponents) if opponents else [opponent]
    results, f = evaluate_genome_multi(genome, rng, pool, decisions, weights)
    return results[0], f


def evaluate_genome_multi(
    genome: dict[str, float],
    rng: random.Random,
    opponents: Sequence[PoolEntry],
    decisions: int = 400,
    weights: dict[str, float] | None = None,
) -> tuple[list[SimResult], float]:
    """Score a genome over several opponents and return (results, fitness).

    `opponents` is a list of built-in archetype names (str) and/or
    `("champion", genome)` tuples for head-to-head self-play.
    """
    results: list[SimResult] = []
    for entry in opponents:
        if isinstance(entry, tuple) and entry[0] == "champion":
            them = RuntimeManager()
            them.genome_base = dict(entry[1])
            mgr = RuntimeManager()
            mgr.genome_base = dict(genome)
            results.append(play_match(mgr, "champion", rng=rng, decisions=decisions, them_manager=them))
        elif isinstance(entry, str):
            manager = RuntimeManager()
            manager.genome_base = dict(genome)
            results.append(play_match(manager, entry, rng=rng, decisions=decisions, collect_trace=False))
        else:
            raise ValueError(f"unsupported pool entry: {entry!r}")
    return results, fitness_from_results(results, weights)


def fitness_from(result: SimResult, weights: dict[str, float] | None = None) -> float:
    """Scalar fitness = weighted goals, goal difference, possession, shots."""
    w = weights or {}
    return (
        w.get("goal_diff", 10.0) * result.goal_diff
        + w.get("goals", 2.0) * result.goals
        + w.get("possession", 1.0) * result.possession_ours
        + w.get("shots", 0.5) * min(result.shots, 20) / 20.0
    )


def fitness_from_results(results: list[SimResult], weights: dict[str, float] | None = None) -> float:
    """Aggregate fitness over an opponent pool (§35).

    The mean win/loss performance plus an opponent-robustness bonus: a genome
    must not lose badly to any single archetype to rank highly.
    """
    w = weights or {}
    if not results:
        return -1e9
    base = sum(fitness_from(r, w) for r in results) / len(results)
    robustness = min((r.goal_diff for r in results), default=0.0)
    return base + w.get("opponent_robustness", 2.0) * max(0.0, robustness)


fitness_from_result = fitness_from


@dataclass
class PopulationStats:
    generation: int
    best_fitness: float
    mean_fitness: float
    worst_fitness: float


@dataclass
class EvolutionSummary:
    best_genome: dict[str, float]
    best_fitness: float
    best_result: SimResult | ScenarioResult = field(default_factory=SimResult)
    history: list[PopulationStats] = field(default_factory=list)
    hall_of_fame: list[dict[str, float]] = field(default_factory=list)
    runs: int = 0
    hof_passed: bool | None = None
    hof_details: list[dict] = field(default_factory=list)

    @property
    def generations(self) -> list[PopulationStats]:
        """Alias kept for the static test suite / scripting use."""
        return self.history

    def to_dict(self) -> dict:
        return {
            "best_genome": self.best_genome,
            "genome_hash": genome_hash(self.best_genome),
            "best_fitness": round(self.best_fitness, 4),
            "best_result": self.best_result.summary(),
            "generations": [s.generation for s in self.history],
            "best_fitness_per_gen": [round(s.best_fitness, 4) for s in self.history],
            "runs": self.runs,
            "hof_passed": self.hof_passed,
        }


def hof_test(
    champion: dict[str, float],
    hof: list[dict[str, float]],
    rng: random.Random,
    decisions: int = 400,
) -> tuple[bool, list[dict]]:
    """Regression test a candidate against the Hall of Fame (§36).

    The candidate plays the away side too (swap roles on the second pass) so a
    side advantage cannot hide a regression. Returns (promoted, details) where
    promotion requires no losing goal difference against any HoF champion.
    """
    details: list[dict] = []
    for champ in hof:
        champ_manager = RuntimeManager()
        champ_manager.genome_base = dict(champ)
        ours = RuntimeManager()
        ours.genome_base = dict(champion)
        res = play_match(ours, "brain", rng=rng, decisions=decisions, them_manager=champ_manager)
        # Second leg with roles swapped.
        champ_home = RuntimeManager()
        champ_home.genome_base = dict(champ)
        us_away = RuntimeManager()
        us_away.genome_base = dict(champion)
        res2 = play_match(champ_home, "brain", rng=rng, decisions=decisions, them_manager=us_away)
        combined = res.goal_diff - res2.goal_diff
        details.append(
            {
                "champion_hash": genome_hash(champ),
                "home_gd": res.goal_diff,
                "away_gd": -res2.goal_diff,
                "combined_gd": combined,
                "promote": combined >= 0,
            }
        )
    promoted = all(d["promote"] for d in details)
    return promoted, details


def evolve(
    rng: random.Random,
    generations: int = 16,
    population: int = 10,
    elite: int = 2,
    tournament_k: int = 3,
    opponent: str = "possession",
    decisions: int = 400,
    mutation_rate: float = 0.25,
    mutation_sigma: float = 0.08,
    seed_genome: dict[str, float] | None = None,
    progress: Callable[[int, PopulationStats], None] | None = None,
    opponents: Sequence[PoolEntry] | None = None,
    styles: list[str] | None = None,
    hof: list[dict[str, float]] | None = None,
    scenario_lab: list[tuple[str, str]] | None = None,
    scenario_ticks: int | None = None,
) -> EvolutionSummary:
    """Run the evolutionary loop and return the best genome + history.

    `seed_genome` pins the initial population around an artist's genome when
    provided (safe default otherwise: a diverse archetype-seeded population,
    §37). `opponents` selects the evaluation pool (§35 robustness); `styles`
    selects which archetypes seed the population; `hof` runs the final
    champion through the Hall-of-Fame regression gate (§36). `scenario_lab`
    replaces full-match fitness with the deterministic scenario corpus
    (`(scenario_name, opponent)` pairs) so skills are graded directly.
    """
    pool = list(opponents) if opponents else [opponent]
    if styles is None and opponents is None:
        pop = [dict(seed_genome) if seed_genome is not None else random_genome(rng) for _ in range(population)]
        if seed_genome is not None and population > 1:
            pop[1:] = [mutate(seed_genome, rng, rate=0.4, sigma=0.05) for _ in range(population - 1)]
    else:
        pop = seeded_population(rng, population, styles=styles, seed_genome=seed_genome)

    summary = EvolutionSummary(best_genome=dict(seed_genome) if seed_genome else dict(pop[0]), best_fitness=-1e18)
    for gen in range(generations):
        fit: list[float] = []
        results: list[SimResult | ScenarioResult] = []
        for genome in pop:
            if scenario_lab:
                res_list, f = evaluate_scenarios(genome, rng, scenario_lab, ticks=scenario_ticks)
                res = res_list[0]
            else:
                res, f = evaluate_genome(genome, rng, opponent, decisions, opponents=pool)
            fit.append(f)
            results.append(res)
            summary.runs += 1

        order = sorted(range(len(pop)), key=lambda i: fit[i], reverse=True)
        stats = PopulationStats(
            generation=gen,
            best_fitness=fit[order[0]],
            mean_fitness=sum(fit) / len(fit),
            worst_fitness=fit[order[-1]],
        )
        summary.history.append(stats)
        best_here = fit[order[0]]
        if best_here > summary.best_fitness:
            summary.best_fitness = best_here
            summary.best_genome = dict(pop[order[0]])
            summary.best_result = results[order[0]]
        summary.hall_of_fame.append(dict(pop[order[0]]))
        if progress is not None:
            progress(gen, stats)

        if gen == generations - 1:
            break

        # Elitism: keep the top `elite` unchanged.
        nxt = [dict(pop[order[i]]) for i in range(elite)]
        while len(nxt) < population:
            a = tournament(pop, fit, tournament_k, rng)
            b = tournament(pop, fit, tournament_k, rng)
            child = mutate(crossover(a, b, rng), rng, mutation_rate, mutation_sigma)
            nxt.append(child)
        pop = nxt

    if hof:
        summary.hof_passed, summary.hof_details = hof_test(summary.best_genome, hof, rng, decisions)

    return summary


def run_evolution(
    rng: random.Random,
    generations: int = 16,
    population_size: int = 10,
    decisions: int = 400,
    opponent: str = "possession",
    elitism: int = 2,
    **kwargs,
) -> EvolutionSummary:
    """Thin wrapper around `evolve` with the static-suite parameter names."""
    return evolve(
        rng,
        generations=generations,
        population=population_size,
        elite=elitism,
        opponent=opponent,
        decisions=decisions,
        **kwargs,
    )


__all__ = [
    "random_genome",
    "mutate",
    "crossover",
    "tournament",
    "evaluate_genome",
    "evaluate_genome_multi",
    "fitness_from",
    "fitness_from_result",
    "fitness_from_results",
    "arch_genome",
    "seeded_population",
    "hof_test",
    "evolve",
    "run_evolution",
    "EvolutionSummary",
    "PopulationStats",
    "PoolEntry",
]