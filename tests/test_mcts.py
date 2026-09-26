import sys
import unittest
from pathlib import Path
import random

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.config import REWARD_DEFAULTS, default_genome, RuntimeConfig
from src.mcts import MCTSPlanner
from src.tactics import TacticalState, assign_roles, PressPlan
from src.policy import PolicyController, PolicyInput
from src.state import GameState, WorldModel

SLOTS = [
    {"id": "defender", "role": "defender", "position": {"x": 14, "y": 20}},
    {"id": "left", "role": "winger", "position": {"x": 28, "y": 8}},
    {"id": "right", "role": "winger", "position": {"x": 28, "y": 32}},
    {"id": "striker", "role": "striker", "position": {"x": 43, "y": 20}},
]


def build_inp():
    observation = {
        "protocolVersion": "1.0",
        "gameId": "m-test",
        "sequence": 1,
        "simulationTick": 1,
        "applyAtTick": 1,
        "timeRemainingSeconds": 300,
        "phase": "openPlay",
        "score": {"us": 0, "them": 0},
        "ball": {
            "position": {"x": 35, "y": 20},
            "velocity": {"x": 0, "y": 0},
            "possessingTeam": "us",
            "possessedBy": "am",
        },
        "us": [
            {"id": "gk", "role": "goalkeeper", "position": {"x": 5, "y": 20}, "velocity": {"x": 0, "y": 0}, "facingRadians": 0, "canAct": True},
            {"id": "cd", "role": "outfield", "position": {"x": 18, "y": 20}, "velocity": {"x": 0, "y": 0}, "facingRadians": 0, "canAct": True},
            {"id": "am", "role": "outfield", "position": {"x": 35, "y": 20}, "velocity": {"x": 0, "y": 0}, "facingRadians": 0, "canAct": True},
            {"id": "w", "role": "outfield", "position": {"x": 32, "y": 8}, "velocity": {"x": 0, "y": 0}, "facingRadians": 0, "canAct": True},
            {"id": "st", "role": "outfield", "position": {"x": 47, "y": 22}, "velocity": {"x": 0, "y": 0}, "facingRadians": 0, "canAct": True},
        ],
        "them": [
            {"id": "tgk", "role": "goalkeeper", "position": {"x": 58, "y": 20}, "velocity": {"x": 0, "y": 0}, "facingRadians": 0, "canAct": True},
            {"id": "cd", "role": "outfield", "position": {"x": 48, "y": 20}, "velocity": {"x": 0, "y": 0}, "facingRadians": 0, "canAct": True},
            {"id": "am", "role": "outfield", "position": {"x": 44, "y": 16}, "velocity": {"x": 0, "y": 0}, "facingRadians": 0, "canAct": True},
            {"id": "st", "role": "outfield", "position": {"x": 42, "y": 14}, "velocity": {"x": 0, "y": 0}, "facingRadians": 0, "canAct": True},
        ],
    }
    state = GameState.from_observation(observation)
    world = WorldModel.build(state)
    roles = assign_roles(state, SLOTS)
    return PolicyInput(
        state=state,
        world=world,
        config=RuntimeConfig(genome=default_genome()),
        tactical_state=TacticalState.PROGRESSION,
        press_plan=PressPlan(),
        roles=roles,
    )


class MCTSTests(unittest.TestCase):
    def test_choose_returns_valid_candidates(self):
        inp = build_inp()
        planner = MCTSPlanner(default_genome(), REWARD_DEFAULTS, random.Random(7), iterations=6, horizon=0.8)
        base = PolicyController().decide(inp)
        best = planner.choose(inp, base)
        if best is not None:
            for pid, it in best.items():
                self.assertEqual(pid, it.pid)
                self.assertGreaterEqual(it.speed, 0.0)
                self.assertLessEqual(it.speed, 1.0)

    def test_choose_not_crashing_on_empty_them(self):
        inp = build_inp()
        inp.state = GameState.from_observation(
            {
                "protocolVersion": "1.0",
                "gameId": "m-test-2",
                "sequence": 1,
                "simulationTick": 1,
                "applyAtTick": 1,
                "timeRemainingSeconds": 300,
                "phase": "openPlay",
                "score": {"us": 0, "them": 0},
                "ball": {"position": {"x": 35, "y": 20}, "velocity": {"x": 0, "y": 0}, "possessingTeam": "us", "possessedBy": "am"},
                "us": [
                    {"id": "gk", "role": "goalkeeper", "position": {"x": 5, "y": 20}, "velocity": {"x": 0, "y": 0}, "facingRadians": 0, "canAct": True},
                    {"id": "cd", "role": "outfield", "position": {"x": 18, "y": 20}, "velocity": {"x": 0, "y": 0}, "facingRadians": 0, "canAct": True},
                    {"id": "am", "role": "outfield", "position": {"x": 35, "y": 20}, "velocity": {"x": 0, "y": 0}, "facingRadians": 0, "canAct": True},
                    {"id": "w", "role": "outfield", "position": {"x": 32, "y": 8}, "velocity": {"x": 0, "y": 0}, "facingRadians": 0, "canAct": True},
                    {"id": "st", "role": "outfield", "position": {"x": 47, "y": 22}, "velocity": {"x": 0, "y": 0}, "facingRadians": 0, "canAct": True},
                ],
                "them": [],
            }
        )
        planner = MCTSPlanner(default_genome(), REWARD_DEFAULTS, random.Random(1), iterations=6, horizon=0.8)
        base = PolicyController().decide(inp)
        result = planner.choose(inp, base)
        self.assertTrue(result is None or isinstance(result, dict))


if __name__ == "__main__":
    unittest.main()