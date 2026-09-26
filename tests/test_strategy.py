import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from strategy import decide


class BasicStrategyTests(unittest.TestCase):
    def test_returns_a_valid_intent_for_every_player(self):
        players = [
            {"id": "gk", "role": "goalkeeper", "position": {"x": 3, "y": 20}, "velocity": {"x": 0, "y": 0}, "facingRadians": 0, "canAct": True},
            {"id": "one", "role": "outfield", "position": {"x": 14, "y": 20}, "velocity": {"x": 0, "y": 0}, "facingRadians": 0, "canAct": True},
            {"id": "two", "role": "outfield", "position": {"x": 28, "y": 8}, "velocity": {"x": 0, "y": 0}, "facingRadians": 0, "canAct": True},
            {"id": "three", "role": "outfield", "position": {"x": 28, "y": 32}, "velocity": {"x": 0, "y": 0}, "facingRadians": 0, "canAct": True},
            {"id": "four", "role": "outfield", "position": {"x": 43, "y": 20}, "velocity": {"x": 0, "y": 0}, "facingRadians": 0, "canAct": True},
        ]
        observation = {
            "protocolVersion": "1.0", "gameId": "builder-test", "sequence": 3,
            "simulationTick": 12, "applyAtTick": 18, "timeRemainingSeconds": 30,
            "phase": "openPlay", "score": {"us": 0, "them": 0},
            "ball": {"id": "ball", "position": {"x": 46, "y": 20}, "velocity": {"x": 0, "y": 0}, "possessedBy": "four", "possessingTeam": "us"},
            "us": players, "them": [],
        }
        decision = decide(observation)
        self.assertEqual("1.0", decision["protocolVersion"])
        self.assertEqual(3, decision["sequence"])
        self.assertEqual(5, len(decision["intents"]))
        self.assertEqual(5, len({intent["playerId"] for intent in decision["intents"]}))
        for intent in decision["intents"]:
            self.assertGreaterEqual(intent["move"]["speed"], 0)
            self.assertLessEqual(intent["move"]["speed"], 1)


if __name__ == "__main__":
    unittest.main()
