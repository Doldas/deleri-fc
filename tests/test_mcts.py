import copy
import math
import sys
import unittest
from pathlib import Path
import random
from dataclasses import replace

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
from src.teamplan import TeamPlan, TeamPlanManager
from src.opponent import OpponentModel
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


class MCTSCoreTests(unittest.TestCase):
    def _goal_position_input(self):
        inp = build_inp()
        inp.state = replace(
            inp.state,
            us=tuple(
                replace(player, x=55.0, y=20.0)
                if player.id == "am"
                else player
                for player in inp.state.us
            ),
            them=tuple(
                replace(player, y=38.0)
                if player.role == "goalkeeper"
                else player
                for player in inp.state.them
            ),
            ball=replace(
                inp.state.ball,
                x=55.0,
                y=20.0,
                possessing_team="us",
                possessing_player="am",
            ),
        )
        inp.world = WorldModel.build(inp.state)
        return inp

    def test_root_board_and_result_are_original_production_candidates(self):
        inp = build_inp()
        controller = PolicyController()
        carrier = inp.state.our_possessor()
        assert carrier is not None
        expected = controller._rank_attack_candidates(
            controller._filter_attack_candidates(
                carrier,
                controller._build_attack_candidates(inp, carrier),
            )
        )

        planner = MCTSPlanner(iterations=8)
        selected = planner.search(inp, controller)

        self.assertEqual(planner.root_candidates, tuple(expected))
        self.assertTrue(any(selected is candidate for candidate in planner.root_candidates))
        self.assertIn(selected, expected)
        self.assertEqual(
            tuple(stat.candidate for stat in planner.stats.root_actions),
            tuple(expected),
        )

    def test_controlling_production_candidate_board_controls_mcts_actions(self):
        class NarrowBoardController(PolicyController):
            def _build_attack_candidates(self, inp, p, team_plan=None):
                return super()._build_attack_candidates(inp, p, team_plan)[:1]

        inp = build_inp()
        controller = NarrowBoardController()
        planner = MCTSPlanner(iterations=4)

        selected = planner.search(inp, controller)

        self.assertEqual(len(planner.root_candidates), 1)
        self.assertIs(selected, planner.root_candidates[0])
        self.assertEqual(planner.stats.root_actions[0].candidate, selected)

    def test_deep_search_uses_candidate_pipeline_at_future_states(self):
        class RecordingController(PolicyController):
            def __init__(self):
                super().__init__()
                self.observed = []

            def _build_attack_candidates(self, inp, p, team_plan=None):
                self.observed.append((inp.state.ball.x, inp.state.ball.possessing_team))
                return super()._build_attack_candidates(inp, p, team_plan)

        inp = build_inp()
        controller = RecordingController()
        planner = MCTSPlanner(iterations=12, max_decision_depth=3)

        selected = planner.search(inp, controller)

        self.assertIsNotNone(selected)
        self.assertGreater(planner.stats.max_depth, 1)
        self.assertGreater(len(controller.observed), 1)
        self.assertTrue(
            any(abs(ball_x - inp.state.ball.x) > 1e-6 for ball_x, _ in controller.observed[1:]),
            "deeper production candidate generation must observe simulated positions",
        )

    def test_expanded_sibling_states_and_team_plans_are_isolated(self):
        inp = build_inp()
        controller = PolicyController()
        team_plan = TeamPlan(shot_expected=True, rebound_zone_x=48.0, rebound_zone_y=17.0)
        team_plan_before = copy.deepcopy(team_plan)
        state_before = copy.deepcopy(inp.state)
        world_before = copy.deepcopy(inp.world)
        planner = MCTSPlanner(iterations=4)

        planner.search(inp, controller, team_plan=team_plan)

        root = planner._root
        assert root is not None
        self.assertGreaterEqual(len(root.children), 2)
        siblings = [root.children[index] for index in sorted(root.children)[:2]]
        root_before = copy.deepcopy(root.state.plan)
        sibling_before = copy.deepcopy(siblings[1].state.plan)
        branch_plan_a = siblings[0].team_plan_manager.current_plan
        branch_plan_b = siblings[1].team_plan_manager.current_plan
        if branch_plan_a is None:
            branch_plan_a = TeamPlan(rebound_zone_x=48.0)
            siblings[0].team_plan_manager.current_plan = branch_plan_a
        if branch_plan_b is None:
            branch_plan_b = TeamPlan(rebound_zone_x=49.0)
            siblings[1].team_plan_manager.current_plan = branch_plan_b
        mutated_player = siblings[0].state.plan.player("us", "am")
        assert mutated_player is not None
        mutated_player.x += 2.0
        siblings[0].state.plan.ball.x += 1.0
        branch_plan_a.rebound_zone_x = 3.0

        self.assertEqual(root.state.plan, root_before)
        self.assertEqual(siblings[1].state.plan, sibling_before)
        self.assertNotEqual(
            branch_plan_a.rebound_zone_x,
            branch_plan_b.rebound_zone_x,
        )
        self.assertEqual(inp.state, state_before)
        self.assertEqual(inp.world, world_before)
        self.assertEqual(team_plan, team_plan_before)

    def test_same_input_and_budget_have_identical_search_and_statistics(self):
        inp = build_inp()
        first = MCTSPlanner(iterations=12, horizon=1.0, cpuct=1.1)
        second = MCTSPlanner(iterations=12, horizon=1.0, cpuct=1.1)

        first_choice = first.search(inp, PolicyController())
        second_choice = second.search(inp, PolicyController())

        self.assertEqual(first_choice, second_choice)
        self.assertEqual(first.stats, second.stats)

    def test_uct_handles_unvisited_exploration_and_stable_ties(self):
        unvisited = MCTSPlanner._uct_score(4, 0, 0.0, 1.4)
        low_parent = MCTSPlanner._uct_score(1, 1, 0.0, 1.0)
        high_parent = MCTSPlanner._uct_score(16, 1, 0.0, 1.0)
        self.assertTrue(math.isfinite(unvisited))
        self.assertTrue(math.isfinite(low_parent))
        self.assertTrue(math.isfinite(high_parent))
        self.assertGreater(high_parent, low_parent)

        planner = MCTSPlanner(iterations=2)
        planner.search(build_inp(), PolicyController())
        root = planner._root
        assert root is not None and len(root.children) >= 2
        first_index, second_index = sorted(root.children)[:2]
        first_child, second_child = root.children[first_index], root.children[second_index]
        first_child.visits = 1
        second_child.visits = 0
        first_child.value = 1e6
        root.visits = 8
        self.assertIs(planner._select(root), second_child)
        first_child.visits = second_child.visits = 1
        first_child.value = second_child.value = 0.5
        root.visits = 2
        self.assertIs(planner._select(root), first_child)
        self.assertIs(planner._select(root), first_child)

    def test_backpropagation_updates_every_ancestor_once(self):
        planner = MCTSPlanner(iterations=8)
        planner.search(build_inp(), PolicyController())
        root = planner._root
        assert root is not None

        leaf = next(
            node
            for child in root.children.values()
            for node in (child, *child.children.values())
            if node.depth >= 2
        )
        path = []
        current = leaf
        while current is not None:
            path.append(current)
            current = current.parent
        path.reverse()
        for node in path:
            node.visits = 0
            node.value = 0.0

        planner._backpropagate(path, 0.75)

        for node in path:
            self.assertEqual(node.visits, 1)
            self.assertEqual(node.value, 0.75)

    def test_loose_ball_no_possessor_empty_board_and_no_opponent_terminate(self):
        loose = build_inp()
        loose.state = replace(
            loose.state,
            ball=replace(
                loose.state.ball,
                possessing_team=None,
                possessing_player=None,
                vx=12.0,
            ),
        )
        loose.world = WorldModel.build(loose.state)
        planner = MCTSPlanner(iterations=6, max_advance_steps=3)
        self.assertIsNone(planner.search(loose, PolicyController()))
        self.assertEqual(planner.stats.iterations, 0)

        no_possessor = build_inp()
        no_possessor.state = replace(
            no_possessor.state,
            ball=replace(
                no_possessor.state.ball,
                possessing_team="them",
                possessing_player="am",
            ),
        )
        no_possessor.world = WorldModel.build(no_possessor.state)
        self.assertIsNone(planner.search(no_possessor, PolicyController()))

        class EmptyBoardController(PolicyController):
            def _build_attack_candidates(self, inp, p, team_plan=None):
                return []

        self.assertIsNone(planner.search(build_inp(), EmptyBoardController()))

        empty_them = build_inp()
        empty_them.state = replace(empty_them.state, them=())
        empty_them.world = WorldModel.build(empty_them.state)
        self.assertIsNotNone(planner.search(empty_them, PolicyController()))
        self.assertLessEqual(planner.stats.simulated_steps, planner.iterations * planner.max_simulated_steps)

    def test_goal_future_beats_non_goal_future_in_real_lightengine_branches(self):
        inp = self._goal_position_input()
        planner = MCTSPlanner(iterations=2, horizon=0.2, max_advance_steps=1)

        selected = planner.search(inp, PolicyController())

        root = planner._root
        assert root is not None
        goal_child = next(
            child for child in root.children.values()
            if child.candidate is not None and child.candidate.intent.action_type == "shoot"
        )
        non_goal_child = next(
            child for child in root.children.values()
            if child.candidate is not None and child.candidate.intent.action_type != "shoot"
        )
        self.assertEqual(goal_child.state.plan.score_us, inp.su + 1)
        self.assertEqual(non_goal_child.state.plan.score_us, inp.su)
        self.assertIs(selected, goal_child.candidate)
        self.assertGreater(goal_child.value / goal_child.visits, non_goal_child.value / non_goal_child.visits)

    def test_conceding_is_strictly_worse_than_comparable_non_conceding_state(self):
        root = SearchState.from_policy_input(build_inp())
        safe = root.clone()
        conceded = root.clone()
        conceded.plan.score_them += 1

        self.assertGreater(MCTSPlanner._evaluate(safe, root), MCTSPlanner._evaluate(conceded, root))

    def test_search_has_no_live_policy_state_side_effects(self):
        inp = build_inp()
        inp.opp = OpponentModel(samples=4, side_bias=1.5, press_samples=2.0, press_accum=5.0)
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
        controller._team_plan_manager = TeamPlanManager()
        controller._team_plan_manager.current_plan = copy.deepcopy(team_plan)
        state_before = copy.deepcopy(inp.state)
        world_before = copy.deepcopy(inp.world)
        opponent_before = copy.deepcopy(inp.opp)
        log_before = copy.deepcopy(controller.log)
        plan_before = copy.deepcopy(team_plan)
        manager_before = copy.deepcopy(controller._team_plan_manager)
        pending_before = (
            controller._pending_pass_receiver,
            controller._pending_pass_collection,
        )

        planner = MCTSPlanner(iterations=10)
        planner.search(inp, controller, team_plan)

        self.assertEqual(inp.state, state_before)
        self.assertEqual(inp.world, world_before)
        self.assertEqual(inp.opp, opponent_before)
        self.assertEqual(controller.log, log_before)
        self.assertEqual(team_plan, plan_before)
        self.assertEqual(
            controller._team_plan_manager.current_plan,
            manager_before.current_plan,
        )
        self.assertEqual(
            controller._team_plan_manager.previous_plan,
            manager_before.previous_plan,
        )
        self.assertEqual(
            (controller._pending_pass_receiver, controller._pending_pass_collection),
            pending_before,
        )

    def test_diagnostics_prove_strict_bounded_search_work(self):
        planner = MCTSPlanner(
            iterations=10,
            horizon=0.8,
            max_decision_depth=2,
            max_advance_steps=3,
        )

        planner.search(build_inp(), PolicyController())

        self.assertEqual(planner.stats.iterations, 10)
        self.assertLessEqual(planner.stats.nodes_created, 1 + planner.iterations)
        self.assertLessEqual(planner.stats.max_depth, planner.max_decision_depth)
        self.assertLessEqual(
            planner.stats.simulated_steps,
            planner.iterations * planner.max_simulated_steps,
        )


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
