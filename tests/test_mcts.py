import copy
import math
import sys
import unittest
from pathlib import Path
import random

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.config import (
    BALL_CONTROL_MAX_SPEED,
    DECISION_INTERVAL,
    REWARD_DEFAULTS,
    RuntimeConfig,
    default_genome,
)
from src.mcts import MCTSPlanner
from src.tactics import TacticalState, assign_roles, PressPlan
from src.policy import ActionCandidate, PlayerIntent, PolicyController, PolicyInput
from src.state import GameState, WorldModel
from src.search_state import (
    SearchState,
    advance_search_state,
    is_within_pitch,
    transition_candidate,
)
from src.teamplan import TeamPlan
from src.physics import MIN_PASS_TRAVEL

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


class SearchStateBoundaryTests(unittest.TestCase):
    def _production_carry(self, inp, controller=None, team_plan=None):
        controller = controller or PolicyController()
        possessor = inp.state.our_possessor()
        assert possessor is not None
        candidates = controller._build_attack_candidates(inp, possessor, team_plan)
        return next(candidate for candidate in candidates if candidate.reason == "carry_forward")

    def _manual_candidate(self, possessor, action_type, target=None, power=None):
        tx, ty = (possessor.x + 5.0, possessor.y + 2.0) if action_type == "none" else (possessor.x, possessor.y)
        intent = PlayerIntent(
            pid=possessor.id,
            tx=tx,
            ty=ty,
            speed=1.0 if action_type == "none" else 0.0,
            face_x=target[0] if target is not None else tx,
            face_y=target[1] if target is not None else ty,
            action_type=action_type,
            action_target=target,
            action_power=power,
        )
        return ActionCandidate(0.0, intent, f"test_{action_type}")

    def _assert_valid_sim_state(self, state):
        self.assertTrue(is_within_pitch(state))
        values = [state.time_remaining, state.plan.time, state.plan.ball.vx, state.plan.ball.vy]
        for player in state.plan.players:
            values.extend((player.x, player.y, player.vx, player.vy, player.facing))
        self.assertTrue(all(math.isfinite(value) for value in values))
        if state.plan.ball.possessing_team is None:
            self.assertIsNone(state.plan.ball.possessing_player)
        else:
            self.assertIsNotNone(
                state.plan.player(
                    state.plan.ball.possessing_team,
                    state.plan.ball.possessing_player or "",
                )
            )

    def test_production_candidate_transitions_without_live_side_effects(self):
        inp = build_inp()
        controller = PolicyController()
        controller._pending_pass_receiver = "live_receiver"
        controller._pending_pass_collection = (12.0, 8.0)
        controller.log.record({"existing": "entry"})
        team_plan = TeamPlan(
            shot_expected=True,
            rebound_zone_x=48.0,
            rebound_zone_y=17.0,
            far_post_target_y=23.0,
        )
        plan_before = copy.deepcopy(team_plan)
        log_before = copy.deepcopy(controller.log)
        pending_before = (
            controller._pending_pass_receiver,
            controller._pending_pass_collection,
        )
        state_before = inp.state
        world_before = copy.deepcopy(inp.world)
        candidate = self._production_carry(inp, controller, team_plan)
        candidate_before = copy.deepcopy(candidate)

        source = SearchState.from_policy_input(inp)
        self.assertEqual(source.plan.ball.x, inp.state.ball.x)
        self.assertEqual(source.plan.ball.y, inp.state.ball.y)
        next_state = transition_candidate(source, candidate)

        self.assertEqual(candidate, candidate_before)
        self.assertEqual(inp.state, state_before)
        self.assertEqual(inp.world, world_before)
        self.assertEqual(team_plan, plan_before)
        self.assertEqual(controller.log, log_before)
        self.assertEqual(
            (controller._pending_pass_receiver, controller._pending_pass_collection),
            pending_before,
        )
        self.assertEqual(next_state.plan.ball.possessing_team, "us")
        self.assertEqual(next_state.plan.ball.possessing_player, candidate.intent.pid)
        moved_carrier = next_state.plan.player("us", candidate.intent.pid)
        source_carrier = source.plan.player("us", candidate.intent.pid)
        assert moved_carrier is not None and source_carrier is not None
        self.assertGreater(moved_carrier.x, source_carrier.x)
        self.assertAlmostEqual(
            next_state.time_remaining,
            source.time_remaining - DECISION_INTERVAL,
        )
        self._assert_valid_sim_state(next_state)

    def test_snapshot_mutation_and_sibling_transitions_are_isolated(self):
        inp = build_inp()
        possessor = inp.state.our_possessor()
        assert possessor is not None
        source = SearchState.from_policy_input(inp)
        source_before = copy.deepcopy(source.plan)
        carry = self._production_carry(inp)
        pass_candidate = self._manual_candidate(
            possessor, "pass", target=(50.0, 20.0), power=0.0
        )

        branch_a = source.clone()
        branch_b = source.clone()
        branch_a_player = branch_a.plan.player("us", possessor.id)
        assert branch_a_player is not None
        branch_a_player.x += 1.0
        self.assertEqual(branch_b.plan, source_before)

        branch_a = transition_candidate(source, carry)
        branch_b = transition_candidate(source, pass_candidate)
        branch_b_before = copy.deepcopy(branch_b.plan)
        branch_a_player = branch_a.plan.player("us", possessor.id)
        assert branch_a_player is not None
        branch_a_player.y += 1.0

        self.assertNotEqual(branch_a.plan, branch_b.plan)
        self.assertEqual(branch_b.plan, branch_b_before)
        self.assertEqual(source.plan, source_before)
        self.assertEqual(inp.state.ball.x, 35.0)
        self.assertEqual(inp.state.our_possessor(), possessor)

    def test_same_state_and_candidate_produce_deterministic_successors(self):
        inp = build_inp()
        candidate = self._production_carry(inp)
        source = SearchState.from_policy_input(inp)

        first = transition_candidate(source, candidate)
        second = transition_candidate(source, candidate)

        self.assertEqual(first, second)
        self._assert_valid_sim_state(first)

    def test_carry_pass_and_shoot_use_lightengine_action_semantics(self):
        inp = build_inp()
        possessor = inp.state.our_possessor()
        assert possessor is not None
        source = SearchState.from_policy_input(inp)
        candidates = (
            self._production_carry(inp),
            self._manual_candidate(possessor, "pass", target=(50.0, 20.0), power=0.0),
            self._manual_candidate(possessor, "shoot", target=(60.0, 18.0), power=0.5),
        )

        carry_state, pass_state, shot_state = (
            transition_candidate(source, candidate) for candidate in candidates
        )

        self.assertEqual(carry_state.plan.ball.possessing_team, "us")
        self.assertEqual(carry_state.plan.ball.possessing_player, possessor.id)
        self.assertIn("kick:pass", pass_state.plan.events)
        self.assertIsNone(pass_state.plan.ball.possessing_team)
        self.assertGreater(math.hypot(pass_state.plan.ball.vx, pass_state.plan.ball.vy), 5.0)
        self.assertIn("kick:shoot", shot_state.plan.events)
        self.assertIsNone(shot_state.plan.ball.possessing_team)
        shot_player = shot_state.plan.player("us", possessor.id)
        assert shot_player is not None
        self.assertAlmostEqual(
            shot_player.facing,
            math.atan2(18.0 - possessor.y, 60.0 - possessor.x),
        )

        for next_state in (carry_state, pass_state, shot_state):
            self._assert_valid_sim_state(next_state)

    def test_short_pass_uses_lightengine_minimum_kick_and_collection_physics(self):
        inp = build_inp()
        possessor = inp.state.our_possessor()
        assert possessor is not None
        source = SearchState.from_policy_input(inp)
        short_pass = self._manual_candidate(
            possessor,
            "pass",
            target=(possessor.x + 1.0, possessor.y),
            power=0.0,
        )

        branch = transition_candidate(source, short_pass)
        self.assertIn("kick:pass", branch.plan.events)
        self.assertIsNone(branch.plan.ball.possessing_team)
        for _ in range(30):
            speed = math.hypot(branch.plan.ball.vx, branch.plan.ball.vy)
            if speed <= BALL_CONTROL_MAX_SPEED or branch.plan.ball.possessing_team is not None:
                break
            branch = advance_search_state(branch)

        speed = math.hypot(branch.plan.ball.vx, branch.plan.ball.vy)
        self.assertLessEqual(speed, BALL_CONTROL_MAX_SPEED)
        self.assertGreater(
            branch.plan.ball.x - possessor.x,
            MIN_PASS_TRAVEL - 2.0,
            "minimum-power pass should roll well beyond its one-metre aim point",
        )
        self._assert_valid_sim_state(branch)


if __name__ == "__main__":
    unittest.main()
