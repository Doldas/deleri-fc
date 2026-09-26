import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.tactics import (
    TacticalState,
    assign_roles,
    detect,
    plan_press,
    PressPlan,
    de_isolated,
)
from src.config import default_genome
from src.state import GameState, WorldModel


def observation_with(ball_pos, possess, our_x, their_x):
    o = {
        "protocolVersion": "1.0",
        "gameId": "t-test",
        "sequence": 1,
        "simulationTick": 1,
        "applyAtTick": 1,
        "timeRemainingSeconds": 300,
        "phase": "openPlay",
        "score": {"us": 0, "them": 0},
        "ball": {
            "position": {"x": ball_pos[0], "y": ball_pos[1]},
            "velocity": {"x": 0, "y": 0},
            "possessingTeam": possess,
            "possessedBy": ("st" if possess == "us" else "st"),
        },
        "us": [
            {"id": "gk", "role": "goalkeeper", "position": {"x": 5, "y": 20}, "velocity": {"x": 0, "y": 0}, "facingRadians": 0, "canAct": True},
            {"id": "cd", "role": "outfield", "position": {"x": our_x, "y": 20}, "velocity": {"x": 0, "y": 0}, "facingRadians": 0, "canAct": True},
            {"id": "am", "role": "outfield", "position": {"x": our_x + 11, "y": 16}, "velocity": {"x": 0, "y": 0}, "facingRadians": 0, "canAct": True},
            {"id": "w", "role": "outfield", "position": {"x": our_x + 9, "y": 6}, "velocity": {"x": 0, "y": 0}, "facingRadians": 0, "canAct": True},
            {"id": "st", "role": "outfield", "position": {"x": our_x + 16, "y": 24}, "velocity": {"x": 0, "y": 0}, "facingRadians": 0, "canAct": True},
        ],
        "them": [
            {"id": "tgk", "role": "goalkeeper", "position": {"x": 56, "y": 20}, "velocity": {"x": 0, "y": 0}, "facingRadians": 0, "canAct": True},
            {"id": "cd", "role": "outfield", "position": {"x": their_x, "y": 20}, "velocity": {"x": 0, "y": 0}, "facingRadians": 0, "canAct": True},
            {"id": "am", "role": "outfield", "position": {"x": their_x - 6, "y": 16}, "velocity": {"x": 0, "y": 0}, "facingRadians": 0, "canAct": True},
            {"id": "w", "role": "outfield", "position": {"x": their_x - 4, "y": 33}, "velocity": {"x": 0, "y": 0}, "facingRadians": 0, "canAct": True},
            {"id": "st", "role": "outfield", "position": {"x": their_x + 8, "y": 18}, "velocity": {"x": 0, "y": 0}, "facingRadians": 0, "canAct": True},
        ],
    }
    return o


class TacticsTests(unittest.TestCase):
    def test_assign_roles_covers_all_slots(self):
        slots = [
            {"id": "defender", "role": "defender", "position": {"x": 14, "y": 20}},
            {"id": "left", "role": "winger", "position": {"x": 28, "y": 8}},
            {"id": "right", "role": "winger", "position": {"x": 28, "y": 32}},
            {"id": "striker", "role": "striker", "position": {"x": 43, "y": 20}},
        ]
        state = GameState.from_observation(observation_with((30, 20), "us", 18, 38))
        roles = assign_roles(state, slots)
        self.assertEqual(
            set(roles.values()),
            {"DEFENDER", "WIDE_LEFT", "WIDE_RIGHT", "STRIKER"},
        )
        self.assertEqual(len(roles), 4)

    def test_detect_attacking_with_control(self):
        state = GameState.from_observation(observation_with((52, 20), "us", 40, 20))
        world = WorldModel.build(state)
        st = detect(state, world, default_genome(), prev_had_control=True)
        self.assertIn(
            st,
            (TacticalState.ATTACK, TacticalState.PROGRESSION, TacticalState.FINAL_ATTACK),
        )

    def test_detect_transition_no_control(self):
        state = GameState.from_observation(observation_with((18, 20), "them", 30, 35))
        world = WorldModel.build(state)
        st = detect(state, world, default_genome(), prev_had_control=True)
        self.assertIn(
            st,
            (
                TacticalState.DEFENSIVE_TRANSITION,
                TacticalState.COUNTERPRESS,
                TacticalState.MID_BLOCK,
                TacticalState.LOW_BLOCK,
                TacticalState.HIGH_PRESS,
            ),
        )

    def test_plan_press_assigns_roles(self):
        state = GameState.from_observation(observation_with((22, 18), "them", 26, 34))
        world = WorldModel.build(state)
        roles = assign_roles(state, [
            {"id": "defender", "role": "defender", "position": {"x": 14, "y": 20}},
            {"id": "left", "role": "winger", "position": {"x": 28, "y": 8}},
            {"id": "right", "role": "winger", "position": {"x": 28, "y": 32}},
            {"id": "striker", "role": "striker", "position": {"x": 43, "y": 20}},
        ])
        plan = plan_press(state, world, default_genome(), roles)
        self.assertIsInstance(plan, PressPlan)
        roles_out = {plan.role_for(pid) for pid in ("cd", "am", "w", "st")}
        self.assertTrue(roles_out & {"PRESS", "PRESS_SUPPORT", "COVER"})

    def test_de_isolated(self):
        obs = observation_with((20, 20), "them", 26, 34)
        state = GameState.from_observation(obs)
        st = next(p for p in state.outfield_them() if p.id == "st")
        self.assertIsInstance(de_isolated(state, st), bool)


if __name__ == "__main__":
    unittest.main()