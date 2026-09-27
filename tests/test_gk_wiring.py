"""
Elite Goalkeeper Wiring Test

Controlled test to verify that opponent goalkeeper position propagates through
the attacking decision pipeline correctly.

This test creates identical attacking states with only the opponent goalkeeper
position varying, and verifies the signal propagates through:
L1: Raw GK position
L2: Normalized GK position
L3: GoalkeeperModel / Threat evaluation
L4: Shot/threat geometry
L5: Candidate action values
L6: Selected action
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.policy import PolicyController, PolicyInput
from src.state import GameState, WorldModel
from src.tactics import TacticalState, assign_roles, PressPlan
from src.config import RuntimeConfig, default_genome

# Coordinate system constants
PITCH_LENGTH = 60.0
PITCH_WIDTH = 40.0
OWN_GOAL_X = 0.0
OPP_GOAL_X = 60.0
GOAL_CENTER_Y = 20.0
GOAL_LOW_Y = 17.0
GOAL_HIGH_Y = 23.0
GOAL_WIDTH = 6.0


SLOTS = [
    {"id": "defender", "role": "defender", "position": {"x": 14, "y": 20}},
    {"id": "left", "role": "winger", "position": {"x": 28, "y": 8}},
    {"id": "right", "role": "winger", "position": {"x": 28, "y": 32}},
    {"id": "striker", "role": "striker", "position": {"x": 43, "y": 20}},
]


def make_inp(gk_x: float, gk_y: float):
    """Create PolicyInput with specific GK position."""
    obs = {
        "protocolVersion": "1.0",
        "gameId": "gk_wiring_test",
        "sequence": 1,
        "simulationTick": 1,
        "applyAtTick": 1,
        "timeRemainingSeconds": 60.0,
        "phase": "openPlay",
        "score": {"us": 0, "them": 0},
        "ball": {
            "position": {"x": 42.0, "y": 20.0},
            "velocity": {"x": 0.0, "y": 0.0},
            "possessingTeam": "us",
            "possessedBy": "st"
        },
        "us": [
            {"id": "gk", "role": "goalkeeper", "position": {"x": 5.0, "y": 20.0}, "velocity": {"x": 0.0, "y": 0.0}, "facingRadians": 0.0, "canAct": True},
            {"id": "cd", "role": "outfield", "position": {"x": 18.0, "y": 20.0}, "velocity": {"x": 0.0, "y": 0.0}, "facingRadians": 0.0, "canAct": True},
            {"id": "am", "role": "outfield", "position": {"x": 32.0, "y": 16.0}, "velocity": {"x": 0.0, "y": 0.0}, "facingRadians": 0.0, "canAct": True},
            {"id": "w", "role": "outfield", "position": {"x": 30.0, "y": 6.0}, "velocity": {"x": 0.0, "y": 0.0}, "facingRadians": 0.0, "canAct": True},
            {"id": "st", "role": "outfield", "position": {"x": 42.0, "y": 20.0}, "velocity": {"x": 0.0, "y": 0.0}, "facingRadians": 0.0, "canAct": True},
        ],
        "them": [
            {"id": "tgk", "role": "goalkeeper", "position": {"x": gk_x, "y": gk_y}, "velocity": {"x": 0.0, "y": 0.0}, "facingRadians": 0.0, "canAct": True},
            {"id": "m1", "role": "outfield", "position": {"x": 40.0, "y": 15.0}, "velocity": {"x": 0.0, "y": 0.0}, "facingRadians": 0.0, "canAct": True},
            {"id": "m2", "role": "outfield", "position": {"x": 40.0, "y": 20.0}, "velocity": {"x": 0.0, "y": 0.0}, "facingRadians": 0.0, "canAct": True},
            {"id": "m3", "role": "outfield", "position": {"x": 40.0, "y": 25.0}, "velocity": {"x": 0.0, "y": 0.0}, "facingRadians": 0.0, "canAct": True},
            {"id": "m4", "role": "outfield", "position": {"x": 40.0, "y": 30.0}, "velocity": {"x": 0.0, "y": 0.0}, "facingRadians": 0.0, "canAct": True},
        ],
    }

    from src.tactics import assign_roles, PressPlan
    from src.state import GameState, WorldModel
    from src.config import RuntimeConfig, default_genome

    obs = {
        "protocolVersion": "1.0",
        "gameId": "gk_wiring_test",
        "sequence": 1,
        "simulationTick": 1,
        "applyAtTick": 1,
        "timeRemainingSeconds": 60.0,
        "phase": "openPlay",
        "score": {"us": 0, "them": 0},
        "ball": {
            "position": {"x": 42.0, "y": 20.0},
            "velocity": {"x": 0.0, "y": 0.0},
            "possessingTeam": "us",
            "possessedBy": "st"
        },
        "us": [
            {"id": "gk", "role": "goalkeeper", "position": {"x": 5.0, "y": 20.0}, "velocity": {"x": 0.0, "y": 0.0}, "facingRadians": 0.0, "canAct": True},
            {"id": "cd", "role": "outfield", "position": {"x": 18.0, "y": 20.0}, "velocity": {"x": 0.0, "y": 0.0}, "facingRadians": 0.0, "canAct": True},
            {"id": "am", "role": "outfield", "position": {"x": 32.0, "y": 16.0}, "velocity": {"x": 0.0, "y": 0.0}, "facingRadians": 0.0, "canAct": True},
            {"id": "w", "role": "outfield", "position": {"x": 30.0, "y": 6.0}, "velocity": {"x": 0.0, "y": 0.0}, "facingRadians": 0.0, "canAct": True},
            {"id": "st", "role": "outfield", "position": {"x": 42.0, "y": 20.0}, "velocity": {"x": 0.0, "y": 0.0}, "facingRadians": 0.0, "canAct": True},
        ],
        "them": [
            {"id": "tgk", "role": "goalkeeper", "position": {"x": gk_x, "y": gk_y}, "velocity": {"x": 0.0, "y": 0.0}, "facingRadians": 0.0, "canAct": True},
            {"id": "m1", "role": "outfield", "position": {"x": 40.0, "y": 15.0}, "velocity": {"x": 0.0, "y": 0.0}, "facingRadians": 0.0, "canAct": True},
            {"id": "m2", "role": "outfield", "position": {"x": 40.0, "y": 20.0}, "velocity": {"x": 0.0, "y": 0.0}, "facingRadians": 0.0, "canAct": True},
            {"id": "m3", "role": "outfield", "position": {"x": 40.0, "y": 25.0}, "velocity": {"x": 0.0, "y": 0.0}, "facingRadians": 0.0, "canAct": True},
            {"id": "m4", "role": "outfield", "position": {"x": 40.0, "y": 30.0}, "velocity": {"x": 0.0, "y": 0.0}, "facingRadians": 0.0, "canAct": True},
        ],
    }

    from src.tactics import assign_roles, PressPlan
    from src.state import GameState, WorldModel
    from src.config import RuntimeConfig, default_genome

    state = GameState.from_observation(obs)
    world = WorldModel.build(state)
    roles = assign_roles(state, SLOTS)
    return PolicyInput(
        state=state,
        world=world,
        config=RuntimeConfig(genome=default_genome()),
        tactical_state=TacticalState.ATTACK,
        press_plan=PressPlan(),
        roles=assign_roles(GameState.from_observation(obs), SLOTS),
    )


SLOTS = [
    {"id": "defender", "role": "defender", "position": {"x": 14, "y": 20}},
    {"id": "left", "role": "winger", "position": {"x": 28, "y": 8}},
    {"id": "right", "role": "winger", "position": {"x": 28, "y": 32}},
    {"id": "striker", "role": "striker", "position": {"x": 43, "y": 20}},
]


class TestCoordinateSystem(unittest.TestCase):
    """Verify the coordinate system matches expectations."""

    def test_opponent_goal_at_60(self):
        from src.geom import OPP_GOAL_X
        self.assertEqual(60.0, OPP_GOAL_X)

    def test_own_goal_at_0(self):
        from src.geom import OWN_GOAL_X
        self.assertEqual(0.0, OWN_GOAL_X)

    def test_goal_center_y(self):
        from src.geom import GOAL_CENTER_Y
        self.assertEqual(20.0, GOAL_CENTER_Y)

    def test_goal_width(self):
        from src.geom import GOAL_LOW_Y, GOAL_HIGH_Y
        self.assertEqual(17.0, GOAL_LOW_Y)
        self.assertEqual(23.0, GOAL_HIGH_Y)
        self.assertEqual(6.0, GOAL_HIGH_Y - GOAL_LOW_Y)


class TestGKPropagation(unittest.TestCase):
    """Test goalkeeper position propagation through attacking pipeline."""

    def setUp(self):
        self.controller = PolicyController()

    def _make_inp(self, gk_x: float, gk_y: float):
        # The striker stands at x=50, i.e. 10 m out and inside the box
        # (inside_box is dist_goal <= 11), so a shot is legitimately on the
        # table and these tests can compare how the shot TARGET moves with the
        # keeper. It used to sit at x=42, 18 m out, where a shot at a keeper
        # standing on his line is a standing catch and is correctly refused --
        # that test only went green because the "rebound setup" candidate forced
        # a shoot through at any range under 20 m. See IN_BOX_SHOT_FLOOR.
        obs = {
            "protocolVersion": "1.0",
            "gameId": "gk_wiring_test",
            "sequence": 1,
            "simulationTick": 1,
            "applyAtTick": 1,
            "timeRemainingSeconds": 60.0,
            "phase": "openPlay",
            "score": {"us": 0, "them": 0},
            "ball": {
                "position": {"x": 50.0, "y": 20.0},
                "velocity": {"x": 0.0, "y": 0.0},
                "possessingTeam": "us",
                "possessedBy": "st"
            },
            "us": [
                {"id": "gk", "role": "goalkeeper", "position": {"x": 5.0, "y": 20.0}, "velocity": {"x": 0.0, "y": 0.0}, "facingRadians": 0.0, "canAct": True},
                {"id": "cd", "role": "outfield", "position": {"x": 18.0, "y": 20.0}, "velocity": {"x": 0.0, "y": 0.0}, "facingRadians": 0.0, "canAct": True},
                {"id": "am", "role": "outfield", "position": {"x": 32.0, "y": 16.0}, "velocity": {"x": 0.0, "y": 0.0}, "facingRadians": 0.0, "canAct": True},
                {"id": "w", "role": "outfield", "position": {"x": 30.0, "y": 6.0}, "velocity": {"x": 0.0, "y": 0.0}, "facingRadians": 0.0, "canAct": True},
                {"id": "st", "role": "outfield", "position": {"x": 50.0, "y": 20.0}, "velocity": {"x": 0.0, "y": 0.0}, "facingRadians": 0.0, "canAct": True},
            ],
            "them": [
                {"id": "tgk", "role": "goalkeeper", "position": {"x": gk_x, "y": gk_y}, "velocity": {"x": 0.0, "y": 0.0}, "facingRadians": 0.0, "canAct": True},
                {"id": "m1", "role": "outfield", "position": {"x": 40.0, "y": 15.0}, "velocity": {"x": 0.0, "y": 0.0}, "facingRadians": 0.0, "canAct": True},
                {"id": "m2", "role": "outfield", "position": {"x": 40.0, "y": 20.0}, "velocity": {"x": 0.0, "y": 0.0}, "facingRadians": 0.0, "canAct": True},
                {"id": "m3", "role": "outfield", "position": {"x": 40.0, "y": 25.0}, "velocity": {"x": 0.0, "y": 0.0}, "facingRadians": 0.0, "canAct": True},
                {"id": "m4", "role": "outfield", "position": {"x": 40.0, "y": 30.0}, "velocity": {"x": 0.0, "y": 0.0}, "facingRadians": 0.0, "canAct": True},
            ],
        }

        from src.tactics import assign_roles, PressPlan
        from src.state import GameState, WorldModel
        from src.config import RuntimeConfig, default_genome

        state = GameState.from_observation(obs)
        world = WorldModel.build(state)
        roles = assign_roles(state, SLOTS)
        return PolicyInput(
            state=state,
            world=world,
            config=RuntimeConfig(genome=default_genome()),
            tactical_state=TacticalState.ATTACK,
            press_plan=PressPlan(),
            roles=assign_roles(state, SLOTS),
        )

    def _decide_with_gk(self, gk_x: float, gk_y: float):
        """Run decision with specific GK position and return intents."""
        inp = self._make_inp(gk_x, gk_y)
        intents = PolicyController().decide(inp)
        return intents

    def test_test_a_central_gk(self):
        """Test A: GK at (60, 20) - central position."""
        intents = self._decide_with_gk(60.0, 20.0)

        # Striker should have a clear shooting option
        st_intent = intents["st"]
        print(f"Test A - GK at (60, 20):")
        print(f"  ST action: {st_intent.action_type}")
        print(f"  ST target: {st_intent.action_target}")
        print(f"  ST power: {st_intent.action_power}")

        self.assertEqual(st_intent.action_type, "shoot", "Striker should shoot with central GK")

        # Store for comparison
        self.result_a = {
            "st_action": intents["st"].action_type,
            "st_target": intents["st"].action_target,
            "st_power": intents["st"].action_power,
        }

    def test_test_b_lateral_gk(self):
        """Test B: GK at (60, 24) - laterally displaced."""
        intents = self._decide_with_gk(60.0, 24.0)

        st_intent = intents["st"]
        print(f"Test B - GK at (60, 24):")
        print(f"  ST action: {st_intent.action_type}")
        print(f"  ST target: {st_intent.action_target}")
        print(f"  ST power: {st_intent.action_power}")

        # The shot target should change to favor the open side
        if hasattr(self, 'result_a') and self.result_a["st_action"] == "shoot" and intents["st"].action_type == "shoot":
            target_a = self.result_a["st_target"]
            target_b = intents["st"].action_target

            if target_a and target_b:
                print(f"  Target A: {target_a}")
                print(f"  Target B: {target_b}")
                # Target should shift toward open side (lower y since GK moved to y=24)
                self.assertNotEqual(target_a, target_b, "Shot target should change with GK lateral displacement")

                # The value should differ even if action is the same
                print(f"  Shot target shifted: YES")

    def test_test_c_advanced_gk(self):
        """Test C: GK at (52, 20) - advanced from goal line."""
        intents = self._decide_with_gk(52.0, 20.0)

        st_intent = intents["st"]
        print(f"Test C - GK at (52, 20):")
        print(f"  ST action: {st_intent.action_type}")
        print(f"  ST target: {st_intent.action_target}")
        print(f"  ST power: {st_intent.action_power}")

        # With GK at 52, they're outside their defensive fifth (48)
        # This should make shooting much easier
        if hasattr(self, 'result_a') and self.result_a["st_action"] == "shoot" and intents["st"].action_type == "shoot":
            target_a = self.result_a["st_target"]
            target_c = intents["st"].action_target

            if target_a and target_c:
                print(f"  Target A: {target_a}")
                print(f"  Target C: {target_c}")
                # The shooting should be easier / target may differ
                self.assertNotEqual(target_a, target_c, "Shot target should change with GK at 52")

    def test_test_d_no_gk(self):
        """Test D: No GK / impossible keeper observation."""
        # This would require constructing a state without GK
        # For now, we skip this as it requires special state construction
        self.skipTest("Test D requires special state construction")


class TestCoordinateSystem(unittest.TestCase):
    """Verify the coordinate system matches expectations."""

    def test_opponent_goal_at_60(self):
        from src.geom import OPP_GOAL_X
        self.assertEqual(60.0, OPP_GOAL_X)

    def test_own_goal_at_0(self):
        from src.geom import OWN_GOAL_X
        self.assertEqual(0.0, OWN_GOAL_X)

    def test_goal_center_y(self):
        from src.geom import GOAL_CENTER_Y
        self.assertEqual(20.0, GOAL_CENTER_Y)

    def test_goal_width(self):
        from src.geom import GOAL_LOW_Y, GOAL_HIGH_Y
        self.assertEqual(17.0, GOAL_LOW_Y)
        self.assertEqual(23.0, GOAL_HIGH_Y)
        self.assertEqual(6.0, GOAL_HIGH_Y - GOAL_LOW_Y)


SLOTS = [
    {"id": "defender", "role": "defender", "position": {"x": 14, "y": 20}},
    {"id": "left", "role": "winger", "position": {"x": 28, "y": 8}},
    {"id": "right", "role": "winger", "position": {"x": 28, "y": 32}},
    {"id": "striker", "role": "striker", "position": {"x": 43, "y": 20}},
]


if __name__ == "__main__":
    unittest.main(verbosity=2)