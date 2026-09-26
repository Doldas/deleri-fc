import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.runtime import RuntimeManager, seed_int


def observation(game_id):
    return {
        "protocolVersion": "1.0",
        "gameId": game_id,
        "sequence": 2,
        "simulationTick": 2,
        "applyAtTick": 2,
        "timeRemainingSeconds": 290,
        "phase": "openPlay",
        "score": {"us": 0, "them": 0},
        "ball": {
            "position": {"x": 40, "y": 20},
            "velocity": {"x": 0, "y": 0},
            "possessingTeam": "us",
            "possessedBy": "am",
        },
        "us": [
            {"id": "gk", "role": "goalkeeper", "position": {"x": 5, "y": 20}, "velocity": {"x": 0, "y": 0}, "facingRadians": 0, "canAct": True},
            {"id": "cd", "role": "outfield", "position": {"x": 18, "y": 20}, "velocity": {"x": 0, "y": 0}, "facingRadians": 0, "canAct": True},
            {"id": "am", "role": "outfield", "position": {"x": 38, "y": 20}, "velocity": {"x": 0, "y": 0}, "facingRadians": 0, "canAct": True},
            {"id": "w", "role": "outfield", "position": {"x": 34, "y": 8}, "velocity": {"x": 0, "y": 0}, "facingRadians": 0, "canAct": True},
            {"id": "st", "role": "outfield", "position": {"x": 47, "y": 24}, "velocity": {"x": 0, "y": 0}, "facingRadians": 0, "canAct": True},
        ],
        "them": [
            {"id": "tgk", "role": "goalkeeper", "position": {"x": 57, "y": 20}, "velocity": {"x": 0, "y": 0}, "facingRadians": 0, "canAct": True},
            {"id": "cd", "role": "outfield", "position": {"x": 47, "y": 20}, "velocity": {"x": 0, "y": 0}, "facingRadians": 0, "canAct": True},
            {"id": "am", "role": "outfield", "position": {"x": 43, "y": 16}, "velocity": {"x": 0, "y": 0}, "facingRadians": 0, "canAct": True},
            {"id": "st", "role": "outfield", "position": {"x": 41, "y": 26}, "velocity": {"x": 0, "y": 0}, "facingRadians": 0, "canAct": True},
        ],
    }


class RuntimeTests(unittest.TestCase):
    def test_seed_int_stable(self):
        self.assertEqual(seed_int("abc", "g1"), seed_int("abc", "g1"))
        self.assertNotEqual(seed_int("abc", "g1"), seed_int("abc", "g2"))
        self.assertNotEqual(seed_int("abc", "g1"), seed_int("xyz", "g1"))

    def test_decide_is_deterministic_across_matches(self):
        m1 = RuntimeManager()
        m2 = RuntimeManager()
        m1.start_match({"gameId": "gmA", "randomSeed": "S1", "seriesId": "X"})
        m2.start_match({"gameId": "gmA", "randomSeed": "S1", "seriesId": "X"})
        d1 = m1.decide(observation("gmA"))
        d2 = m2.decide(observation("gmA"))
        self.assertEqual(d1, d2)

    def test_end_match_discards_context(self):
        m = RuntimeManager()
        m.start_match({"gameId": "gmZ", "randomSeed": "S2", "seriesId": "X"})
        m.decide(observation("gmZ"))
        summary = m.end_match({"gameId": "gmZ"})
        self.assertIsNotNone(summary)
        self.assertEqual(summary["game_id"], "gmZ")
        self.assertIsNone(m.end_match({"gameId": "gmZ"}))

    def test_decide_echoes_game_id_and_sequence(self):
        m = RuntimeManager()
        m.decide(observation("gmEcho"))
        d = m.decide(observation("gmEcho"))
        self.assertEqual(d["gameId"], "gmEcho")
        self.assertEqual(d["sequence"], 2)
        for intent in d["intents"]:
            self.assertIn(intent["action"]["type"], ("none", "pass", "shoot", "clear", "tackle", "slap"))
            self.assertGreaterEqual(intent["move"]["speed"], 0.0)
            self.assertLessEqual(intent["move"]["speed"], 1.0)


if __name__ == "__main__":
    unittest.main()