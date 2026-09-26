import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from strategy import decide


class StrategyTests(unittest.TestCase):
    def test_returns_one_intent_per_player(self):
        players = [
            {
                "id": "gk",
                "role": "goalkeeper",
                "position": {"x": 3, "y": 20},
                "velocity": {"x": 0, "y": 0},
                "facingRadians": 0,
                "canAct": True,
            },
            {
                "id": "f1",
                "role": "outfield",
                "position": {"x": 15, "y": 10},
                "velocity": {"x": 1, "y": 0},
                "facingRadians": 0,
                "canAct": True,
            },
        ]
        observation = {
            "protocolVersion": "1.0",
            "gameId": "test",
            "sequence": 7,
            "simulationTick": 12,
            "applyAtTick": 18,
            "timeRemainingSeconds": 29.8,
            "phase": "openPlay",
            "score": {"us": 0, "them": 0},
            "ball": {
                "id": "ball",
                "position": {"x": 30, "y": 20},
                "velocity": {"x": 0, "y": 0},
                "possessedBy": "f1",
                "possessingTeam": "us",
            },
            "us": players,
            "them": [],
        }
        decision = decide(observation)
        self.assertEqual(observation["protocolVersion"], decision["protocolVersion"])
        self.assertEqual(7, decision["sequence"])
        self.assertEqual(2, len(decision["intents"]))
        self.assertEqual("shoot", decision["intents"][1]["action"]["type"])


if __name__ == "__main__":
    unittest.main()
