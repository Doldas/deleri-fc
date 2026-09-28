"""Tests for ExpectedValueCalculator and the EV candidate board.

These cover the four defects the calculator was rewritten for, and the
regressions that surfaced once the planner was actually pricing its candidates
in goal-equivalents instead of invented constants:

1. ``p_shot_blocked`` used to be a constant, so every shot looked equally
   likely to be walked into.
2. ``p_shot_goal`` used to be effectively binary -- "beats the keeper" was
   worth a goal, "does not" was worth nothing, regardless of how wide the
   margin was.
3. The shot outcomes were not an explicit distribution, so they did not sum to
   one and ``ev_shot`` had to renormalise a set of overlapping terms.
4. ``p_pass_complete`` ignored where the ball actually stops, so it raced the
   receiver to the aim point instead of to the collection point.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.physics import GK_LATERAL_REACH
from src.policy import (
    ExpectedValueCalculator,
    PolicyController,
    _shot_geometry_target,
)
from src.state import GameState
from tests.test_policy import make_inp, obs

OPP_GOAL_X = 60.0
GOAL_CENTER_Y = 20.0


def place_opponents(observation, positions):
    """Put outfield opponents at exact positions, leaving the keeper put."""
    them = [p for p in observation["them"] if p["role"] == "goalkeeper"]
    for i, (x, y) in enumerate(positions):
        them.append(
            {
                "id": f"o{i}",
                "role": "outfield",
                "position": {"x": x, "y": y},
                "velocity": {"x": 0, "y": 0},
                "facingRadians": 0,
                "canAct": True,
            }
        )
    observation["them"] = them
    return observation


def carrier(inp):
    """The ball carrier, asserting there is one."""
    state = inp.state if hasattr(inp, "state") else inp
    p = state.our_possessor()
    assert p is not None, "fixture has no carrier"
    return p


def in_box(our_x=52.0, keeper=(58.0, 20.0), opponents=((10.0, 20.0),)):
    observation = place_opponents(
        obs((our_x, 20.0), "us", our_st_x=our_x, them_x=10), list(opponents)
    )
    for pl in observation["them"]:
        if pl["role"] == "goalkeeper":
            pl["position"]["x"], pl["position"]["y"] = keeper
    return make_inp(observation)


class ShotBlockTests(unittest.TestCase):
    """A block is a physical intercept, not a fixed penalty."""

    def test_clear_lane_is_not_blocked(self):
        inp = in_box(opponents=((5.0, 4.0), (5.0, 36.0)))
        ev = ExpectedValueCalculator(inp)
        p = carrier(inp)
        self.assertEqual(ev.p_shot_blocked(carrier(inp), (OPP_GOAL_X, 17.6), 0.9), 0.0)

    def test_a_defender_on_the_segment_blocks(self):
        inp = in_box(opponents=((56.0, 18.8),))
        ev = ExpectedValueCalculator(inp)
        p = carrier(inp)
        blocked = ev.p_shot_blocked(carrier(inp), (OPP_GOAL_X, 17.6), 0.9)
        self.assertGreater(blocked, 0.2)

    def test_block_probability_grows_with_the_wall(self):
        inp1 = in_box(opponents=((56.0, 18.8),))
        one = ExpectedValueCalculator(inp1).p_shot_blocked(
            carrier(inp1), (OPP_GOAL_X, 17.6), 0.9
        )

        inp3 = in_box(opponents=((56.0, 18.8), (56.2, 19.6), (56.2, 18.0)))
        three = ExpectedValueCalculator(inp3).p_shot_blocked(
            carrier(inp3), (OPP_GOAL_X, 17.6), 0.9
        )
        self.assertGreater(three, one)

    def test_a_defender_behind_the_shooter_does_not_block(self):
        inp = in_box(opponents=((48.0, 20.0),))
        ev = ExpectedValueCalculator(inp)
        p = carrier(inp)
        self.assertEqual(ev.p_shot_blocked(carrier(inp), (OPP_GOAL_X, 17.6), 0.9), 0.0)


class ShotDistributionTests(unittest.TestCase):
    """The four outcomes are exclusive and exhaustive."""

    def _distribution(self, inp):
        ev = ExpectedValueCalculator(inp)
        p = carrier(inp)
        target, power = _shot_geometry_target(inp, p)
        return ev, p, target, power

    def test_outcomes_sum_to_one_when_the_lane_is_clear(self):
        inp = in_box(opponents=((5.0, 4.0), (5.0, 36.0)))
        ev, p, target, power = self._distribution(inp)
        self.assertAlmostEqual(
            ev.p_shot_blocked(p, target, power)
            + ev.p_shot_wide(p, target, power)
            + ev.p_shot_saved(p, target, power)
            + ev.p_shot_goal(p, target, power),
            1.0,
            places=9,
        )

    def test_outcomes_sum_to_one_when_blocked(self):
        """The case the old renormalisation was hiding."""
        inp = in_box(opponents=((56.0, 18.8), (56.2, 19.6), (56.2, 18.0)))
        ev, p, target, power = self._distribution(inp)
        total = (
            ev.p_shot_blocked(p, target, power)
            + ev.p_shot_wide(p, target, power)
            + ev.p_shot_saved(p, target, power)
            + ev.p_shot_goal(p, target, power)
        )
        self.assertAlmostEqual(total, 1.0, places=9)

    def test_every_outcome_is_a_probability(self):
        inp = in_box(opponents=((56.0, 18.8),))
        ev, p, target, power = self._distribution(inp)
        for name in ("p_shot_blocked", "p_shot_wide", "p_shot_saved", "p_shot_goal"):
            value = getattr(ev, name)(p, target, power)
            with self.subTest(outcome=name):
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)


class ShotGoalTests(unittest.TestCase):
    """xG is continuous in the geometry, not a beats-the-keeper flag."""

    def test_a_wide_open_keeper_makes_the_same_shot_a_goal(self):
        """Dragging the keeper off his line is the whole lever.

        From the middle, 8 m out, both posts are about 2.4 m from a set keeper
        and he has time to shift across; aimed at the far post with him pulled
        the other way, the ball is out of reach. The old model returned 0.0 and
        1.0 for those two pictures.
        """
        tight = ExpectedValueCalculator(in_box(keeper=(58.0, 20.0)))
        p = tight.state.our_possessor()
        central = tight.p_shot_goal(p, (OPP_GOAL_X, 22.4), 0.95)

        wide = ExpectedValueCalculator(in_box(keeper=(56.0, 13.0)))
        p2 = wide.state.our_possessor()
        dragged = wide.p_shot_goal(p2, (OPP_GOAL_X, 22.4), 0.95)

        self.assertLess(central, 0.10)
        self.assertGreater(dragged, 0.30)

    def test_xg_falls_off_with_range(self):
        ev = ExpectedValueCalculator(in_box(our_x=40, keeper=(58.0, 20.0)))
        state = ev.state
        near = ExpectedValueCalculator(in_box(our_x=52.0, keeper=(58.0, 20.0)))
        xg_near = near.p_shot_goal(carrier(near.state), (OPP_GOAL_X, 22.4), 0.9)
        xg_far = ev.p_shot_goal(state.our_possessor(), (OPP_GOAL_X, 22.4), 0.9)
        self.assertGreater(xg_near, xg_far)

    def test_xg_is_never_a_step_function(self):
        """Ammunition of the old defect: small geometry changes must move xG.

        The old model answered a boolean, so every aim point either scored or
        did not and this sweep was two values. It has to vary continuously, and
        stay strictly inside (0, 1): a keeper who covers a post gets 0, a
        ball past him is not a certainty.
        """
        ev = ExpectedValueCalculator(in_box(keeper=(58.0, 20.0)))
        p = carrier(ev)
        values = [
            ev.p_shot_goal(p, (OPP_GOAL_X, 17.0 + 0.5 * i), 0.95) for i in range(13)
        ]
        self.assertGreaterEqual(len({round(v, 6) for v in values}), 6, values)
        for v in values:
            self.assertGreater(v, 0.0, "no shot should be a certainty")
            self.assertLess(v, 1.0, "no shot should be worth exactly zero")


    def test_p_goal_is_bounded_by_reaching_the_frame(self):
        ev = ExpectedValueCalculator(inbox := in_box(keeper=(58.0, 20.0)))
        p = carrier(inbox)
        target, power = _shot_geometry_target(inbox, p)
        self.assertLessEqual(
            ev.p_shot_goal(p, target, power), ev.p_shot_on_target(p, target, power)
        )


class PassCompletionTests(unittest.TestCase):
    """The race is to the collection point, not to the aim point."""

    def test_pass_plan_solves_for_the_intended_spot(self):
        inp = in_box(our_x=30, opponents=((50.0, 4.0),))
        ev = ExpectedValueCalculator(inp)
        p = carrier(inp)
        power, cx, cy, _t_ball, overshoot = ev.pass_plan(p, (48.0, 20.0))
        self.assertLess(overshoot, 0.5)
        self.assertGreaterEqual(power, 0.0)
        self.assertLessEqual(power, 1.0)

    def test_an_closer_opponent_makes_the_pass_worse(self):
        """Same pass, same receiver, only the defender's position differs."""
        clear = in_box(our_x=30, opponents=((52.0, 34.0),))
        ev_clear = ExpectedValueCalculator(clear)
        p = clear.state.our_possessor()
        open_pass = ev_clear.p_pass_complete(p, (48.0, 20.0), receiver_id="am")

        covered = in_box(our_x=30, opponents=((45.0, 20.0),))
        ev_cov = ExpectedValueCalculator(covered)
        p2 = carrier(covered)
        covered_pass = ev_cov.p_pass_complete(p2, (48.0, 20.0), receiver_id="am")

        self.assertGreater(open_pass, covered_pass)

    def test_a_short_pass_overshoots_and_is_penalised(self):
        """Every kick leaves at KICK_MIN_SPEED, so it cannot stop at 3 m."""
        inp = in_box(our_x=30, opponents=((52.0, 34.0),))
        ev = ExpectedValueCalculator(inp)
        p = carrier(inp)
        power, cx, cy, _t, overshoot = ev.pass_plan(p, (33.0, 20.0))
        self.assertGreater(overshoot, 10.0)
        self.assertLess(ev.p_pass_complete(p, (33.0, 20.0), power), 0.5)

    def test_completion_is_a_probability(self):
        inp = in_box(our_x=30, opponents=((45.0, 20.0),))
        ev = ExpectedValueCalculator(inp)
        p = carrier(inp)
        value = ev.p_pass_complete(p, (48.0, 20.0), receiver_id="am")
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 1.0)


class CandidateScaleTests(unittest.TestCase):
    """Everything on the candidate board is a goal-equivalent EV.

    Each of these was a regression where an invented constant on a different
    scale outbid a correctly-priced shot, so the team passed or dribbled
    instead of shooting.
    """

    def test_a_backwards_pass_is_worth_less_than_a_forward_one(self):
        inp = in_box(our_x=30, opponents=((52.0, 34.0),))
        ev = ExpectedValueCalculator(inp)
        p = carrier(inp)
        self.assertGreater(
            ev.value_pass_completed(p, (44.0, 20.0)),
            ev.value_pass_completed(p, (18.0, 20.0)),
        )

    def test_retreating_from_the_box_has_negative_value(self):
        inp = in_box(our_x=52.0, opponents=((10.0, 20.0),))
        ev = ExpectedValueCalculator(inp)
        p = carrier(inp)
        self.assertLess(ev.value_pass_completed(p, (30.0, 6.0)), 0.0)

    def test_dribbling_into_the_keeper_is_not_a_good_carry(self):
        """The keeper is not in outfield_them(), so he used to be invisible."""
        inp = in_box(keeper=(58.0, 20.0))
        ev = ExpectedValueCalculator(inp)
        p = carrier(inp)
        self.assertLess(ev.p_carry_success(p, 58.5, 18.2), 0.3)
        self.assertLess(ev.ev_carry(p, 58.5, 18.2), 0.0)

    def test_dribbling_away_from_the_keeper_is_a_fine_carry(self):
        inp = in_box(keeper=(58.0, 20.0))
        ev = ExpectedValueCalculator(inp)
        p = carrier(inp)
        self.assertGreater(ev.p_carry_success(p, 40.0, 20.0), 0.8)

    def test_a_carry_ends_inside_keeper_reach_is_a_giveaway(self):
        inp = in_box(keeper=(58.0, 20.0))
        ev = ExpectedValueCalculator(inp)
        p = carrier(inp)
        self.assertLessEqual(ev.p_carry_success(p, 58.0, 20.0), 0.05)

    def test_shooting_beats_passing_away_from_an_open_goal(self):
        """8 m out with the keeper set: shoot rather than roll it sideways."""
        inp = in_box(keeper=(58.0, 20.0), opponents=((10.0, 20.0), (4.0, 16.0), (18.0, 24.0)))
        intent = PolicyController().decide(inp)["st"]
        self.assertEqual(intent.action_type, "shoot")

    def test_no_candidate_emits_an_illegal_action_type(self):
        """The engine only accepts none/pass/shoot/clear/tackle/slap.

        A wall shot used to be emitted as the invented action type
        "wall_shot", which is not in that set.
        """
        legal = {"none", "pass", "shoot", "clear", "tackle", "slap"}
        for our_x in (44.0, 48.0, 52.0):
            for y in (4.0, 20.0, 36.0):
                with self.subTest(pos=(our_x, y)):
                    observation = obs((our_x, y), "us", our_st_x=our_x, them_x=10)
                    intents = PolicyController().decide(make_inp(observation))
                    for pid, intent in intents.items():
                        self.assertIn(intent.action_type, legal, f"{pid} at {our_x},{y}")


class PuckOutOfPlayTests(unittest.TestCase):
    """A miss is not a turnover: there are no goal kicks in this ruleset."""

    def test_a_wide_shot_is_not_priced_as_losing_the_ball(self):
        ev = ExpectedValueCalculator(in_box())
        self.assertGreaterEqual(ev.value_shot_wide(), 0.0)

    def test_a_dive_is_credited_with_the_grounded_keeper_window(self):
        """A dive leaves the keeper grounded for GK_GROUNDED_SECONDS."""
        inp = in_box(keeper=(57.0, 16.0))
        ev = ExpectedValueCalculator(inp)
        p = carrier(inp)
        self.assertGreater(ev.p_shot_forces_dive(p, (OPP_GOAL_X, 22.4), 0.95), 0.0)

    def test_a_clean_goal_is_not_also_credited_as_a_dive(self):
        ev = ExpectedValueCalculator(in_box(keeper=(40.0, 20.0)))
        p = carrier(ev.state)
        self.assertEqual(ev.p_shot_forces_dive(p, (OPP_GOAL_X, 22.4), 0.95), 0.0)


if __name__ == "__main__":
    unittest.main()
