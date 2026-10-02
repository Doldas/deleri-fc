import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.config import DECISION_INTERVAL, MAX_RUN_SPEED
from src.geom import PITCH_LENGTH, distance
from src.light import make_state
from src.mcts import MCTSPlanner
from src.opponent import OpponentModel
from src.policy import (
    ActionCandidate,
    PlayerIntent,
    PolicyController,
    loose_ball_meeting_point,
)
from src.search_opponent import OPPONENT_RESPONSE_MODEL, branch_opponent_intents
from src.search_state import SearchState, advance_search_state, transition_candidate
from src.teamplan import TeamPlan, TeamPlanManager


def _search_state(*, carrier=True, ball=(20.6, 20.0), velocity=(0.0, 0.0)):
    plan = make_state(
        [
            ("gk", "goalkeeper", 3.0, 20.0),
            ("carrier", "outfield", 20.0, 20.0),
            ("support", "outfield", 13.0, 10.0),
        ],
        [
            ("tgk", "goalkeeper", 57.0, 20.0),
            ("near", "outfield", 24.0, 20.0),
            ("cover", "outfield", 39.0, 11.0),
            ("wide", "outfield", 42.0, 31.0),
        ],
        ball=ball,
        ball_v=velocity,
        possess=("us", "carrier") if carrier else None,
    )
    return SearchState(plan, 90.0)


def _candidate(state, action="none", *, target=None, collection=None, tx=23.0, ty=20.0):
    holder = state.plan.player("us", "carrier")
    assert holder is not None
    intent = PlayerIntent(
        pid="carrier",
        tx=tx if action == "none" else holder.x,
        ty=ty if action == "none" else holder.y,
        speed=1.0 if action == "none" else 0.0,
        face_x=target[0] if target else tx,
        face_y=target[1] if target else ty,
        action_type=action,
        action_target=target,
        action_power=0.0 if action in {"pass", "shoot"} else None,
        receiver_id="support" if action == "pass" else None,
        collection_point=collection,
    )
    return ActionCandidate(0.0, intent, f"test_{action}")


class SearchOpponentTests(unittest.TestCase):
    def test_carrier_branch_moves_pressing_defender_and_other_defenders_recover(self):
        state = _search_state()
        candidate = _candidate(state)
        before = copy.deepcopy(state.plan)
        intents = branch_opponent_intents(state, candidate)

        self.assertEqual(OPPONENT_RESPONSE_MODEL, "deterministic_press_shape_v1")
        self.assertNotIn(("them", "tgk"), intents)
        self.assertEqual(intents[("them", "near")].tx, 20.0)
        self.assertNotEqual(
            (intents[("them", "cover")].tx, intents[("them", "cover")].ty),
            (39.0, 11.0),
        )
        branch = transition_candidate(state, candidate, background_intents=intents)
        near_before = before.player("them", "near")
        near_after = branch.plan.player("them", "near")
        cover_before = before.player("them", "cover")
        cover_after = branch.plan.player("them", "cover")
        assert near_before is not None and near_after is not None
        assert cover_before is not None and cover_after is not None
        self.assertLess(distance(near_after.x, near_after.y, 20.0, 20.0), 4.0)
        self.assertLessEqual(
            distance(near_after.x, near_after.y, near_before.x, near_before.y),
            MAX_RUN_SPEED * DECISION_INTERVAL + 1e-9,
        )
        self.assertNotEqual((cover_after.x, cover_after.y), (cover_before.x, cover_before.y))
        self.assertEqual(state.plan, before)

    def test_identical_state_and_candidate_have_identical_responses_and_successors(self):
        state = _search_state()
        candidate = _candidate(state)
        intents_a = branch_opponent_intents(state, candidate)
        intents_b = branch_opponent_intents(state, candidate)
        successor_a = transition_candidate(state, candidate, background_intents=intents_a)
        successor_b = transition_candidate(state, candidate, background_intents=intents_b)

        self.assertEqual(intents_a, intents_b)
        self.assertEqual(successor_a, successor_b)

    def test_sibling_branches_have_independent_opponent_physics(self):
        state = _search_state()
        carry = _candidate(state, tx=25.0, ty=20.0)
        passed = _candidate(
            state,
            "pass",
            target=(48.0, 20.0),
            collection=(43.0, 20.0),
        )
        carry_branch = transition_candidate(
            state, carry, background_intents=branch_opponent_intents(state, carry)
        )
        pass_branch = transition_candidate(
            state, passed, background_intents=branch_opponent_intents(state, passed)
        )
        pass_before = copy.deepcopy(pass_branch.plan)
        carry_near = carry_branch.plan.player("them", "near")
        source_near = state.plan.player("them", "near")
        assert carry_near is not None and source_near is not None
        carry_near.x += 1.0

        self.assertNotEqual(carry_branch.plan, pass_branch.plan)
        self.assertEqual(pass_branch.plan, pass_before)
        self.assertEqual(state.plan.ball.possessing_team, "us")
        self.assertEqual(source_near.x, 24.0)

    def test_responses_do_not_mutate_live_policy_opponent_teamplan_or_logs(self):
        state = _search_state()
        from tests.test_mcts import build_inp

        inp = build_inp()
        controller = PolicyController()
        controller._pending_pass_receiver = "live"
        controller._pending_pass_collection = (9.0, 6.0)
        controller.log.record({"kept": True})
        controller._team_plan_manager = TeamPlanManager()
        controller._team_plan_manager.current_plan = TeamPlan(reason="live")
        opponent = OpponentModel(samples=3, side_bias=1.0, press_accum=7.0, press_samples=2.0)
        inp.opp = opponent
        before = {
            "state": copy.deepcopy(state.plan),
            "controller": copy.deepcopy(controller),
            "opponent": copy.deepcopy(opponent),
            "input": copy.deepcopy(inp),
        }

        branch_opponent_intents(state, _candidate(state))
        planner = MCTSPlanner(iterations=4, horizon=0.4, max_advance_steps=2)
        planner.search(inp, controller)

        self.assertEqual(state.plan, before["state"])
        self.assertEqual(inp.state, before["input"].state)
        self.assertEqual(inp.opp, before["opponent"])
        self.assertEqual(controller.log, before["controller"].log)
        self.assertEqual(controller._pending_pass_receiver, "live")
        self.assertEqual(controller._pending_pass_collection, (9.0, 6.0))
        self.assertEqual(
            controller._team_plan_manager.current_plan,
            before["controller"]._team_plan_manager.current_plan,
        )
        self.assertEqual(
            controller._team_plan_manager.previous_plan,
            before["controller"]._team_plan_manager.previous_plan,
        )

    def test_nearest_appropriate_defender_closes_on_carrier(self):
        state = _search_state()
        intents = branch_opponent_intents(state)
        self.assertEqual(intents[("them", "near")].tx, 20.0)
        self.assertEqual(intents[("them", "near")].ty, 20.0)
        self.assertEqual(intents[("them", "near")].speed, 1.0)
        self.assertNotEqual(intents[("them", "cover")].tx, 20.0)

    def test_pass_candidate_sends_nearest_defender_to_collection_then_tracks_ball(self):
        state = _search_state()
        pass_candidate = _candidate(
            state,
            "pass",
            target=(48.0, 20.0),
            collection=(43.0, 20.0),
        )
        intents = branch_opponent_intents(state, pass_candidate)
        receiver = intents[("them", "cover")]
        self.assertAlmostEqual(receiver.tx, 43.0)
        self.assertAlmostEqual(receiver.ty, 20.0)
        self.assertEqual(receiver.action_type, "none")

        branch = transition_candidate(state, pass_candidate, background_intents=intents)
        before_positions = {
            player.pid: (player.x, player.y)
            for player in branch.plan.outfield("them")
        }
        self.assertIsNone(branch.plan.ball.possessing_team)
        for _ in range(3):
            movement = branch_opponent_intents(branch)
            branch = advance_search_state(branch, movement)
        moved = [
            player
            for player in branch.plan.outfield("them")
            if (player.x, player.y) != before_positions[player.pid]
        ]
        self.assertTrue(moved, "outfield defenders froze while the pass travelled")
        self.assertGreater(branch.plan.ball.x, state.plan.ball.x)
        self.assertGreaterEqual(branch.time_remaining, 0.0)

    def test_loose_ball_pursuit_targets_physics_meeting_point(self):
        state = _search_state(carrier=False, ball=(30.0, 20.0), velocity=(12.0, 0.0))
        expected_x, expected_y, _ = loose_ball_meeting_point(30.0, 20.0, 12.0, 0.0)
        intents = branch_opponent_intents(state)
        presser = next(
            intent
            for (team, _pid), intent in intents.items()
            if team == "them"
            and abs(intent.tx - expected_x) < 1e-9
            and abs(intent.ty - expected_y) < 1e-9
        )

        self.assertAlmostEqual(presser.tx, expected_x)
        self.assertAlmostEqual(presser.ty, expected_y)
        before = next(
            player
            for player in state.plan.outfield("them")
            if player.pid == presser.pid
        )
        branch = advance_search_state(state, intents)
        after = branch.plan.player("them", presser.pid)
        assert before is not None and after is not None
        self.assertGreater(after.x, before.x)

    def test_shot_candidate_assigns_an_outfielder_to_the_shot_lane(self):
        state = _search_state()
        shot = _candidate(state, "shoot", target=(60.0, 20.0))
        intents = branch_opponent_intents(state, shot)
        interceptor = intents[("them", "near")]
        self.assertAlmostEqual(interceptor.ty, 20.0)
        self.assertGreaterEqual(interceptor.tx, state.plan.ball.x)
        self.assertLessEqual(interceptor.tx, 60.0)
        self.assertEqual(interceptor.action_type, "none")

    def test_defensive_response_mirrors_when_attack_direction_is_reversed(self):
        state = _search_state()
        original = branch_opponent_intents(state)
        mirror_plan = make_state(
            [
                ("tgk", "goalkeeper", 3.0, 20.0),
                ("near", "outfield", PITCH_LENGTH - 24.0, 20.0),
                ("cover", "outfield", PITCH_LENGTH - 39.0, 11.0),
                ("wide", "outfield", PITCH_LENGTH - 42.0, 31.0),
            ],
            [
                ("gk", "goalkeeper", PITCH_LENGTH - 3.0, 20.0),
                ("carrier", "outfield", PITCH_LENGTH - 20.0, 20.0),
                ("support", "outfield", PITCH_LENGTH - 13.0, 10.0),
            ],
            ball=(PITCH_LENGTH - 20.6, 20.0),
            possess=("them", "carrier"),
        )
        mirrored_state = SearchState(mirror_plan, 90.0)
        mirrored = branch_opponent_intents(mirrored_state, defending_team="us")

        for pid in ("near", "cover", "wide"):
            a, b = original[("them", pid)], mirrored[("us", pid)]
            self.assertAlmostEqual(a.tx + b.tx, PITCH_LENGTH)
            self.assertAlmostEqual(a.ty, b.ty)
            self.assertAlmostEqual(a.speed, b.speed)
            self.assertEqual(a.action_type, b.action_type)

    def test_goalkeeper_is_not_given_an_intent_and_lightengine_still_catches(self):
        state = _search_state(carrier=False, ball=(4.0, 20.0), velocity=(0.0, 0.0))
        keeper_before = copy.deepcopy(state.plan.player("us", "gk"))
        intents = branch_opponent_intents(state)

        self.assertNotIn(("them", "tgk"), intents)
        self.assertNotIn(("us", "gk"), intents)
        after = advance_search_state(state, intents)
        keeper_after = after.plan.player("us", "gk")
        assert keeper_before is not None and keeper_after is not None
        self.assertEqual((keeper_after.x, keeper_after.y), (keeper_before.x, keeper_before.y))
        self.assertEqual(after.plan.ball.possessing_team, "us")
        self.assertEqual(after.plan.ball.possessing_player, "gk")

    def test_search_diagnostics_remain_within_existing_limits(self):
        from tests.test_mcts import build_inp

        planner = MCTSPlanner(
            iterations=8,
            horizon=0.5,
            max_decision_depth=2,
            max_advance_steps=2,
        )
        planner.search(build_inp(), PolicyController())

        self.assertEqual(planner.stats.iterations, 8)
        self.assertLessEqual(planner.stats.max_depth, 2)
        self.assertLessEqual(planner.stats.simulated_steps, 8 * planner.max_simulated_steps)
        self.assertLessEqual(planner.stats.nodes_created, 1 + 8)

if __name__ == "__main__":
    unittest.main()
