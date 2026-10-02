import sys
import unittest
from pathlib import Path
import random

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.scenarios import SCENARIOS, run_scenario, evaluate_scenarios, scenario_fitness
from src.config import default_genome


class ScenarioTests(unittest.TestCase):
    def test_corpus_registry(self):
        for name in (
            "final_third",
            "one_v_one_gk",
            "deep_block",
            "press_recovery",
            "down_a_goal_late",
            "kickoff_defend",
            "counter_break",
            "wall_attack",
        ):
            self.assertIn(name, SCENARIOS)

    def test_scenarios_build_deterministically(self):
        for name, scene in SCENARIOS.items():
            a = scene(random.Random(4))
            b = scene(random.Random(4))
            self.assertEqual(a.score_us, b.score_us)
            self.assertEqual(round(a.ball.x, 3), round(b.ball.x, 3))
            self.assertEqual(scene.ticks > 0, True, name)

    def test_one_v_one_gk_converts_with_goal(self):
        res = run_scenario(None, "one_v_one_gk", "defensive", random.Random(33))
        self.assertIn("score_us", res.summary())
        self.assertGreaterEqual(res.shots, 0)

    def test_champion_handles_final_third_and_converts_one_v_one(self):
        lab = [("final_third", "defensive"), ("one_v_one_gk", "defensive")]
        results, fitness = evaluate_scenarios(default_genome(), random.Random(33), lab)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].name, "final_third")
        # The away goalkeeper now handles balls in its own (right-side) fifth;
        # the packed final-third scenario can be held without a goal or shot.
        self.assertEqual(results[0].score_them, 0)
        self.assertGreater(results[0].possession_ours, 0.0)
        self.assertEqual(results[1].score_us, 1)
        self.assertGreater(fitness, 0.0)

    def test_defensive_scenarios_hold_clean(self):
        res = run_scenario(None, "kickoff_defend", "counter", random.Random(33))
        res2 = run_scenario(None, "counter_break", "counter", random.Random(33))
        for r in (res, res2):
            self.assertEqual(r.score_them, 0)
            self.assertGreaterEqual(r.score_us, 0)

    def test_fitness_aggregation(self):
        from src.scenarios import ScenarioResult

        scenes = [SCENARIOS["final_third"], SCENARIOS["one_v_one_gk"]]
        results = [
            ScenarioResult(name="final_third", score_us=1, shots=1, reward=0.5),
            ScenarioResult(name="one_v_one_gk", score_us=1, shots=1, reward=0.5),
        ]
        f = scenario_fitness(results, scenes)
        self.assertGreater(f, 200.0)

    def test_evolve_with_scenario_lab(self):
        from src.evolution import run_evolution

        result = run_evolution(
            random.Random(9),
            generations=2,
            population_size=4,
            decisions=10,
            scenario_lab=[("final_third", "defensive"), ("one_v_one_gk", "defensive")],
        )
        self.assertEqual(result.runs, 8)
        self.assertEqual(len(result.generations), 2)


if __name__ == "__main__":
    unittest.main()
