import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.config import REWARD_DEFAULTS
from src.evaluate import reward
from src.state import GameState, WorldModel


def state_with(ball_x, possess):
    obs = {
        "protocolVersion": "1.0",
        "gameId": "e-test",
        "sequence": 1,
        "simulationTick": 1,
        "applyAtTick": 1,
        "timeRemainingSeconds": 300,
        "phase": "openPlay",
        "score": {"us": 0, "them": 0},
        "ball": {
            "position": {"x": ball_x, "y": 20},
            "velocity": {"x": 0, "y": 0},
            "possessingTeam": possess,
            "possessedBy": ("st" if possess == "us" else "st"),
        },
        "us": [
            {"id": "gk", "role": "goalkeeper", "position": {"x": 5, "y": 20}, "velocity": {"x": 0, "y": 0}, "facingRadians": 0, "canAct": True},
            {"id": "cd", "role": "outfield", "position": {"x": 18, "y": 20}, "velocity": {"x": 0, "y": 0}, "facingRadians": 0, "canAct": True},
            {"id": "am", "role": "outfield", "position": {"x": 29, "y": 16}, "velocity": {"x": 0, "y": 0}, "facingRadians": 0, "canAct": True},
            {"id": "w", "role": "outfield", "position": {"x": 27, "y": 6}, "velocity": {"x": 0, "y": 0}, "facingRadians": 0, "canAct": True},
            {"id": "st", "role": "outfield", "position": {"x": 43, "y": 24}, "velocity": {"x": 0, "y": 0}, "facingRadians": 0, "canAct": True},
        ],
        "them": [
            {"id": "tgk", "role": "goalkeeper", "position": {"x": 56, "y": 20}, "velocity": {"x": 0, "y": 0}, "facingRadians": 0, "canAct": True},
            {"id": "cd", "role": "outfield", "position": {"x": 40, "y": 20}, "velocity": {"x": 0, "y": 0}, "facingRadians": 0, "canAct": True},
            {"id": "am", "role": "outfield", "position": {"x": 46, "y": 16}, "velocity": {"x": 0, "y": 0}, "facingRadians": 0, "canAct": True},
        ],
    }
    return obs


class EvaluateTests(unittest.TestCase):
    def test_reward_increases_with_territorial_progress(self):
        near = reward(GameState.from_observation(state_with(48, "us")), WorldModel.build(GameState.from_observation(state_with(48, "us"))), REWARD_DEFAULTS)
        far = reward(GameState.from_observation(state_with(12, "us")), WorldModel.build(GameState.from_observation(state_with(12, "us"))), REWARD_DEFAULTS)
        self.assertGreater(near, far)

    def test_goal_event_adds_weight(self):
        st = GameState.from_observation(state_with(48, "us"))
        wm = WorldModel.build(st)
        base = reward(st, wm, REWARD_DEFAULTS, prev={})
        scored = reward(st, wm, REWARD_DEFAULTS, prev={"goal": True})
        conceded = reward(st, wm, REWARD_DEFAULTS, prev={"goal_conceded": True})
        self.assertEqual(scored - base, REWARD_DEFAULTS["goal"])
        self.assertEqual(conceded - base, REWARD_DEFAULTS["goal_conceded"])
        self.assertLess(REWARD_DEFAULTS["goal_conceded"], 0.0)

    def test_defaults_have_expected_keys(self):
        for key in (
            "goal",
            "goal_conceded",
            "territorial_progression",
            "possession_quality",
            "space_created",
            "chance_creation",
            "shot_quality",
            "successful_press",
            "counterpress_recovery",
            "defensive_compactness",
            "passing_quality",
            "wall_progression",
            "dangerous_turnover",
            "broken_shape",
            "failed_press",
            "bad_shot",
            "unnecessary_risk",
        ):
            self.assertIn(key, REWARD_DEFAULTS)


if __name__ == "__main__":
    unittest.main()