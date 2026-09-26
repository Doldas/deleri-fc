import sys
import unittest
from pathlib import Path
import random

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.sim import play_match, plan_to_obs, plan_reward, OPPONENTS, BASE_LINEUP
from src.light import make_state
from src.config import REWARD_DEFAULTS


class SimTests(unittest.TestCase):
    def test_opponents_registry(self):
        for name in ("possession", "press", "direct", "defensive", "random", "counter", "parkbus", "wall"):
            self.assertIn(name, OPPONENTS)

    def test_all_opponents_playable(self):
        for name in OPPONENTS:
            result = play_match(None, name, rng=random.Random(5), decisions=6)
            self.assertEqual(result.goals, result.score_us)
            self.assertEqual(result.goals_conceded, result.score_them)

    def test_plan_to_obs_roundtrip(self):
        state = make_state(list(BASE_LINEUP), list(BASE_LINEUP), ball=(30.0, 20.0), possess=("us", "st"))
        obs = plan_to_obs(state, "sim-test", 5)
        self.assertEqual(obs["gameId"], "sim-test")
        self.assertEqual(obs["sequence"], 5)
        self.assertEqual(len(obs["us"]), len(BASE_LINEUP))
        self.assertEqual(len(obs["them"]), len(BASE_LINEUP))
        self.assertEqual(len(obs["us"]), 5)
        self.assertEqual(obs["ball"]["possessingTeam"], "us")

    def test_plan_reward_number(self):
        state = make_state(list(BASE_LINEUP), list(BASE_LINEUP), ball=(30.0, 20.0), possess=("us", "st"))
        r = plan_reward(state, REWARD_DEFAULTS, ["kick:shoot"])
        self.assertEqual(r, r)  # not NaN
        self.assertIsInstance(r, float)

    def test_play_match_end_to_end(self):
        result = play_match(None, "possession", rng=random.Random(11), decisions=20)
        self.assertEqual(result.goals, result.score_us)
        self.assertEqual(result.goals_conceded, result.score_them)
        s = result.summary()
        self.assertIn("possession%", s)
        self.assertIn("score_us", s)
        self.assertIn("reward", s)
        self.assertGreaterEqual(s["possession%"], 0.0)
        self.assertLessEqual(s["possession%"], 100.0)


if __name__ == "__main__":
    unittest.main()