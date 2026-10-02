import copy
import math
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.config import RuntimeConfig, default_genome
from src.mcts import RootActionStats, SearchDiagnostics
from src.opponent import OpponentModel
from src.policy import ActionCandidate, PlayerIntent, PolicyController, PolicyInput
from src.shadow import candidate_action_type, candidate_signature, evaluate_shadow
from src.state import GameState, WorldModel
from src.tactics import PressPlan, TacticalState, assign_roles
from src.teamplan import TeamPlan, TeamPlanManager

SLOTS = [
    {"id": "defender", "role": "defender", "position": {"x": 14, "y": 20}},
    {"id": "left", "role": "winger", "position": {"x": 28, "y": 8}},
    {"id": "right", "role": "winger", "position": {"x": 28, "y": 32}},
    {"id": "striker", "role": "striker", "position": {"x": 43, "y": 20}},
]


def build_input():
    obs = {
        "protocolVersion": "1.0",
        "gameId": "shadow-test",
        "sequence": 9,
        "simulationTick": 90,
        "applyAtTick": 90,
        "timeRemainingSeconds": 210,
        "phase": "openPlay",
        "score": {"us": 1, "them": 0},
        "ball": {
            "position": {"x": 35.0, "y": 20.0},
            "velocity": {"x": 0.0, "y": 0.0},
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
    state = GameState.from_observation(obs)
    return PolicyInput(
        state=state,
        world=WorldModel.build(state),
        config=RuntimeConfig(genome=default_genome()),
        tactical_state=TacticalState.PROGRESSION,
        press_plan=PressPlan(),
        roles=assign_roles(state, SLOTS),
        time_remaining=210.0,
        score_us=1,
        score_them=0,
        match_duration=300.0,
    )


def board_for(inp):
    player = inp.state.our_possessor()
    assert player is not None
    carry = ActionCandidate(
        30.0,
        PlayerIntent(player.id, 40.0, 20.0, 0.8, 40.0, 20.0, "none"),
        "carry label A",
    )
    passed = ActionCandidate(
        20.0,
        PlayerIntent(
            player.id, player.x, player.y, 0.0, 53.0, 20.0,
            "pass", (53.0, 20.0), 0.4, "st", (53.0, 20.0),
        ),
        "pass label",
    )
    shot = ActionCandidate(
        10.0,
        PlayerIntent(
            player.id, player.x, player.y, 0.0, 60.0, 18.0,
            "shoot", (60.0, 18.0), 0.6,
        ),
        "shot label",
    )
    return [carry, passed, shot]


class ControlledPlanner:
    selections = {}
    seen_boards = []

    def __init__(self, *, iterations, **kwargs):
        self.iterations = iterations
        self.stats = SearchDiagnostics()

    def search(self, inp, controller=None, team_plan=None, *, root_candidates=None):
        assert root_candidates is not None
        type(self).seen_boards.append(root_candidates)
        index = type(self).selections.get(self.iterations, 0)
        stats = tuple(
            RootActionStats(candidate, visits=1, value=float(position))
            for position, candidate in enumerate(root_candidates)
        )
        self.stats = SearchDiagnostics(
            iterations=self.iterations,
            nodes_created=1 + self.iterations,
            max_depth=2,
            simulated_steps=2 * self.iterations,
            root_actions=stats,
        )
        return root_candidates[index]


class ShadowEvaluationTests(unittest.TestCase):
    def setUp(self):
        ControlledPlanner.selections = {}
        ControlledPlanner.seen_boards = []

    def _controlled(self, inp, controller=None, team_plan=None, budgets=(2, 5)):
        with patch("src.shadow.MCTSPlanner", ControlledPlanner):
            return evaluate_shadow(
                inp,
                controller=controller,
                team_plan=team_plan,
                budgets=budgets,
            )

    def test_mcts_and_production_choices_share_exact_same_root_board(self):
        inp = build_input()
        result = self._controlled(inp)

        self.assertTrue(result.eligible)
        self.assertEqual(len(ControlledPlanner.seen_boards), 2)
        first_board, second_board = ControlledPlanner.seen_boards
        self.assertIs(first_board, second_board)
        self.assertEqual(result.root_candidate_count, len(first_board))
        self.assertEqual(result.production_signature, candidate_signature(first_board[0]))
        self.assertIs(first_board[0], first_board[result.production_index])
        self.assertIs(first_board[result.mcts_index], first_board[0])

    def test_agreement_when_mcts_returns_production_top_candidate(self):
        result = self._controlled(build_input())
        self.assertTrue(result.mcts_ran)
        self.assertTrue(result.agreement)
        self.assertEqual(result.production_index, result.mcts_index)

    def test_disagreement_and_action_types_are_reported(self):
        ControlledPlanner.selections = {2: 1, 5: 1}
        inp = build_input()
        candidates = board_for(inp)

        class FixedBoardController(PolicyController):
            def _build_attack_candidates(self, inp, p, team_plan=None):
                return candidates

        result = self._controlled(inp, FixedBoardController())
        self.assertFalse(result.agreement)
        self.assertEqual(result.production_action_type, "carry")
        self.assertEqual(result.mcts_action_type, "pass")
        self.assertEqual(result.mcts_production_value, 20.0)
        self.assertEqual(result.production_ev_difference, 10.0)

    def test_signature_ignores_reason_text_and_tracks_action_semantics(self):
        candidate = board_for(build_input())[0]
        renamed = replace(candidate, reason="unrelated explanation")
        moved = replace(
            candidate,
            intent=replace(candidate.intent, tx=candidate.intent.tx + 0.00001),
        )
        self.assertEqual(candidate_signature(candidate), candidate_signature(renamed))
        self.assertNotEqual(candidate_signature(candidate), candidate_signature(moved))

    def test_budget_stability_and_unstability_are_explicit(self):
        self.assertTrue(self._controlled(build_input()).budget_stable)
        ControlledPlanner.selections = {2: 1, 5: 0}
        unstable = self._controlled(build_input())
        self.assertFalse(unstable.budget_stable)
        self.assertEqual([row.selected_index for row in unstable.budget_results], [1, 0])
        self.assertEqual(unstable.mcts_index, 0, "headline choice uses the largest budget")

    def test_shadow_isolation_includes_policy_state_plans_world_and_candidates(self):
        inp = build_input()
        controller = PolicyController()
        controller._pending_pass_receiver = "pending"
        controller._pending_pass_collection = (12.0, 7.0)
        controller._current_game_id = inp.state.game_id
        controller._last_sequence = inp.state.sequence
        controller.log.record({"existing": "log"})
        controller._team_plan_manager = TeamPlanManager()
        controller._team_plan_manager.current_plan = TeamPlan(
            shot_expected=True,
            shot_committed=True,
            rebound_zone_x=48.0,
            rebound_zone_y=17.0,
        )
        inp.opp = OpponentModel(samples=5, side_bias=2.0, press_samples=4, press_accum=6.0)
        candidates = board_for(inp)

        class FixedBoardController(PolicyController):
            def _build_attack_candidates(self, inp, p, team_plan=None):
                return candidates

        live_controller = FixedBoardController()
        live_controller._pending_pass_receiver = controller._pending_pass_receiver
        live_controller._pending_pass_collection = controller._pending_pass_collection
        live_controller._current_game_id = controller._current_game_id
        live_controller._last_sequence = controller._last_sequence
        live_controller.log = copy.deepcopy(controller.log)
        live_controller._team_plan_manager = copy.deepcopy(controller._team_plan_manager)
        live_inp = copy.deepcopy(inp)
        before = {
            "input": copy.deepcopy(live_inp),
            "state": copy.deepcopy(live_inp.state),
            "world": copy.deepcopy(live_inp.world),
            "opp": copy.deepcopy(live_inp.opp),
            "controller": copy.deepcopy(live_controller),
            "candidates": copy.deepcopy(candidates),
            "plan": copy.deepcopy(live_controller._team_plan_manager),
        }

        self._controlled(live_inp, live_controller, live_controller._team_plan_manager.current_plan)

        self.assertEqual(live_inp.state, before["state"])
        self.assertEqual(live_inp.world, before["world"])
        self.assertEqual(live_inp.opp, before["opp"])
        self.assertEqual(live_inp.config, before["input"].config)
        self.assertEqual(live_inp.press_plan, before["input"].press_plan)
        self.assertEqual(live_inp.roles, before["input"].roles)
        self.assertEqual(live_controller.log, before["controller"].log)
        self.assertEqual(live_controller._current_game_id, before["controller"]._current_game_id)
        self.assertEqual(live_controller._last_sequence, before["controller"]._last_sequence)
        self.assertEqual(
            (live_controller._pending_pass_receiver, live_controller._pending_pass_collection),
            (before["controller"]._pending_pass_receiver, before["controller"]._pending_pass_collection),
        )
        self.assertEqual(
            live_controller._team_plan_manager.current_plan,
            before["plan"].current_plan,
        )
        self.assertEqual(
            live_controller._team_plan_manager.previous_plan,
            before["plan"].previous_plan,
        )
        self.assertEqual(candidates, before["candidates"])
        self.assertTrue(all(a is b for a, b in zip(candidates, ControlledPlanner.seen_boards[0])))

        expected_controller = copy.deepcopy(live_controller)
        expected = expected_controller.decide(copy.deepcopy(live_inp))
        actual = live_controller.decide(copy.deepcopy(live_inp))
        self.assertEqual(actual, expected, "shadow evaluation cannot alter later production output")

    def test_deterministic_records_for_same_root_and_budgets(self):
        inp = build_input()
        first = evaluate_shadow(inp, budgets=(2, 5))
        second = evaluate_shadow(inp, budgets=(2, 5))
        self.assertEqual(first, second)
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(first.opponent_model, "stationary_opponents_optimistic")

    def test_ineligible_and_empty_boards_fail_closed(self):
        inp = build_input()
        no_possessor = copy.deepcopy(inp)
        no_possessor.state = replace(
            inp.state,
            ball=replace(inp.state.ball, possessing_team=None, possessing_player=None),
        )
        no_possessor.world = WorldModel.build(no_possessor.state)
        self.assertEqual(evaluate_shadow(no_possessor).skip_reason, "no_our_possessor")

        not_open = copy.deepcopy(inp)
        not_open.state = replace(inp.state, phase="kickoff")
        self.assertEqual(evaluate_shadow(not_open).skip_reason, "not_open_play")

        unable = copy.deepcopy(inp)
        unable.state = replace(
            inp.state,
            us=tuple(replace(player, can_act=False) if player.id == "am" else player for player in inp.state.us),
        )
        self.assertEqual(evaluate_shadow(unable).skip_reason, "possessor_unable_to_act")

        class EmptyController(PolicyController):
            def _build_attack_candidates(self, inp, p, team_plan=None):
                return []

        empty = evaluate_shadow(inp, EmptyController())
        self.assertFalse(empty.mcts_ran)
        self.assertEqual(empty.skip_reason, "empty_production_candidate_board")

    def test_carry_pass_shoot_none_classes_and_budget_cap(self):
        inp = build_input()
        board = board_for(inp)
        self.assertEqual(
            [candidate_action_type(candidate, inp) for candidate in board],
            ["carry", "pass", "shoot"],
        )
        stationary = replace(board[0], intent=replace(board[0].intent, tx=35.0, ty=20.0))
        self.assertEqual(candidate_action_type(stationary, inp), "none")
        with self.assertRaises(ValueError):
            evaluate_shadow(inp, budgets=(257,))


if __name__ == "__main__":
    unittest.main()
