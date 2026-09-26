import sys
import unittest
from pathlib import Path
import random
import json
import tempfile

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.evolution import (
    random_genome,
    mutate,
    crossover,
    run_evolution,
    fitness_from_result,
    fitness_from_results,
    arch_genome,
    seeded_population,
    evaluate_genome_multi,
    STYLE_TEMPLATES,
)
from src.sim import SimResult
from src.training import distill, baseline_report, campaign, promote_candidate
from src.config import GENOME_KEYS, GENOME_RANGES, default_genome


class EvolutionTests(unittest.TestCase):
    def test_random_genome_within_ranges(self):
        g = random_genome(random.Random(1))
        for key in GENOME_KEYS:
            low, high = GENOME_RANGES[key]
            self.assertGreaterEqual(g[key], low)
            self.assertLessEqual(g[key], high)

    def test_mutate_and_crossover_preserve_keys(self):
        rng = random.Random(2)
        a = {k: 0.3 for k in GENOME_KEYS}
        b = {k: 0.8 for k in GENOME_KEYS}
        m = mutate(a, rng, rate=0.5)
        c = crossover(a, b, rng)
        for key in GENOME_KEYS:
            self.assertIn(key, m)
            self.assertIn(key, c)
            self.assertTrue(c[key] in (0.3, 0.8))

    def test_fitness_monotonic_in_result(self):
        base = SimResult()
        win = SimResult(goals=2, goals_conceded=0, possession_ours=0.9)
        loss = SimResult(goals=0, goals_conceded=2, possession_ours=0.5)
        self.assertGreater(fitness_from_result(win), fitness_from_result(loss))
        self.assertGreater(fitness_from_result(win), fitness_from_result(base))

    def test_run_evolution_bounded(self):
        rng = random.Random(9)
        result = run_evolution(
            rng,
            generations=2,
            population_size=4,
            decisions=10,
            opponent="possession",
            elitism=1,
        )
        self.assertEqual(len(result.generations), 2)
        self.assertEqual(result.runs, 8)
        self.assertEqual(set(result.best_genome.keys()), set(GENOME_KEYS))

    def test_baseline_report_and_distill(self):
        from src.sim import OPPONENTS
        rows = baseline_report(seed=3, decisions=10)
        self.assertEqual(len(rows), len(OPPONENTS))  # one row per opponent
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "distilled.json"
            distill(default_genome(), out_path=out)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertIn("genome", payload)
            self.assertIn("genomeHash", payload)
            self.assertEqual(set(payload["genome"].keys()), set(GENOME_KEYS))

    def test_archetypes_are_distinct_and_ranged(self):
        genomes = {s: arch_genome(s) for s in STYLE_TEMPLATES}
        # Pressing archetype presses harder than possession archetype.
        self.assertGreater(
            genomes["pressing"]["press_intensity"],
            genomes["possession"]["press_intensity"],
        )
        # Direct archetype plays more vertically than possession.
        self.assertGreater(genomes["direct"]["verticality"], genomes["possession"]["verticality"])
        for g in genomes.values():
            for key in GENOME_KEYS:
                lo, hi = GENOME_RANGES[key]
                self.assertGreaterEqual(g[key], lo)
                self.assertLessEqual(g[key], hi)

    def test_seeded_population_covers_all_styles(self):
        pop = seeded_population(random.Random(11), size=6)
        self.assertEqual(len(pop), 6)
        self.assertEqual(set(pop[0].keys()), set(GENOME_KEYS))
        self.assertTrue(all(len(g) == len(GENOME_KEYS) for g in pop))

    def test_multi_opponent_fitness_bonus_for_robustness(self):
        good = [SimResult(goals=1, goals_conceded=0), SimResult(goals=1, goals_conceded=0)]
        fragile = [SimResult(goals=3, goals_conceded=0), SimResult(goals=0, goals_conceded=3)]
        self.assertGreater(fitness_from_results(good), fitness_from_results(fragile))

    def test_hof_test_head_to_head_same_genome_draws(self):
        from src.evolution import hof_test
        champs = [default_genome()]
        promoted, details = hof_test(default_genome(), champs, random.Random(13), decisions=10)
        self.assertEqual(len(details), 1)
        self.assertIn("combined_gd", details[0])

    def test_evaluate_genome_multi_with_champion_selfplay(self):
        champion = default_genome()
        pop = seeded_population(random.Random(4), size=2, seed_genome=champion)
        results, _ = evaluate_genome_multi(
            pop[0],
            random.Random(6),
            ["possession", ("champion", champion)],
            decisions=10,
        )
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r.score_us == r.goals and r.score_them == r.goals_conceded for r in results))

    def test_campaign_writes_candidate_then_promote(self):
        import tempfile
        from src.training import CAMPAIGN_POLICY_JSON
        report = campaign(
            seed=21,
            generations=2,
            population=4,
            elite=1,
            decisions=10,
            opponents=["possession", "defensive", ("champion", default_genome())],
            styles=["possession", "direct"],
        )
        self.assertIsNotNone(report.artifact_path)
        self.assertTrue(CAMPAIGN_POLICY_JSON.exists())
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "promoted.json"
            promote_candidate(CAMPAIGN_POLICY_JSON, out_path=out)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(set(payload["genome"].keys()), set(GENOME_KEYS))


if __name__ == "__main__":
    unittest.main()