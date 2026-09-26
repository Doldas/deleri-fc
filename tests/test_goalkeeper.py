"""Goalkeeper positioning tests.

Measured against Vanguard FC (elite), whose keeper in
`opponents/counter-elite/strategy.py::_gk_plan` is the strongest in the
generated set. Two things it does that ours did not:

* stands on the bisector between the ball and the goal centre
  (`_lerp(by, GOAL_CENTER_Y, 0.62)`) instead of clamping himself into the
  goal mouth, so a cross from a wide angle goes round him;
* scales his depth with the ball (`bx * (0.22 + 0.16 * aggression)`) instead of
  sitting on the goal line at x=2.0 for the whole match, so the angle an
  attacker has to hit narrows as he advances.

Both are geometric, not tuning: a keeper confined to y=17.5..22.4 in a 6 m
goal cannot cover a ball crossed from the touchline, and a keeper on the line
always presents the full angle however far away the ball is.

Each test below was checked to fail against the pre-change keeper, so a
regression of any of this is actually caught rather than merely asserted.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.policy import PolicyController  # noqa: E402

from tests.test_policy import obs  # noqa: E402
from tests.test_vanguard_fixes import make_inp  # noqa: E402

# The old keeper clamped himself into the goal mouth, inset 0.6 m from each
# post, so he could never leave this band no matter where the ball was.
OLD_BAND = (17.6, 22.4)

# Spelled out rather than imported from src.policy on purpose: these tests have
# to be runnable against the *old* keeper to prove they discriminate, and an
# import of a constant that only exists after the change would abort the module
# before a single positional assertion runs.
DEFENSIVE_FIFTH = 12.0   # RULES.md: first 20% of a 60 m pitch
BALL_GOAL_BIAS = 0.62    # matches opponents/counter-elite


def gk_intent(ball_x, ball_y, possess="them"):
    observation = obs((ball_x, ball_y), possess, our_st_x=6, them_x=ball_x)
    observation["ball"]["possessedBy"] = "tst" if possess == "them" else "st"
    return PolicyController().decide(make_inp(observation))["gk"]


class LateralCoverageTests(unittest.TestCase):
    def test_covers_a_ball_crossed_from_the_touchline(self):
        # Old: ty=17.5, inside the goal mouth. New: 2 + (20-2)*0.62 = 13.2.
        it = gk_intent(50.0, 2.0)
        self.assertLess(it.ty, OLD_BAND[0] - 1.0)
        self.assertAlmostEqual(it.ty, 2.0 + (20.0 - 2.0) * BALL_GOAL_BIAS, places=1)

    def test_covers_a_ball_crossed_from_the_far_touchline(self):
        it = gk_intent(50.0, 38.0)
        self.assertGreater(it.ty, OLD_BAND[1] + 1.0)

    def test_stays_central_when_the_ball_is_central(self):
        it = gk_intent(50.0, 20.0)
        self.assertAlmostEqual(it.ty, 20.0, places=1)

    def test_covers_both_flanks_symmetrically(self):
        low = gk_intent(50.0, 2.0).ty
        high = gk_intent(50.0, 38.0).ty
        self.assertAlmostEqual(low - 20.0, -(high - 20.0), places=1)


class DepthTests(unittest.TestCase):
    def test_advances_to_meet_a_ball_played_in_from_range(self):
        # Old: tx=2.0 whatever the ball did.
        it = gk_intent(50.0, 20.0)
        self.assertGreater(it.tx, 8.0)

    def test_depth_scales_with_the_ball(self):
        near = gk_intent(25.0, 20.0).tx
        far = gk_intent(50.0, 20.0).tx
        self.assertGreater(far, near)

    def test_never_leaves_the_defensive_fifth(self):
        # RULES.md: automatic handling only applies inside the first 20% from
        # his own goal line, so going past it buys nothing and risks the ball
        # running through him.
        self.assertAlmostEqual(DEFENSIVE_FIFTH, 12.0, places=6)
        for ball_x in (5.0, 15.0, 30.0, 45.0, 58.0):
            it = gk_intent(ball_x, 20.0)
            self.assertLessEqual(it.tx, DEFENSIVE_FIFTH - 1.0 + 1e-6)

    def test_stays_deep_when_the_ball_is_in_his_own_box(self):
        # A ball 5 m away leaves no angle to narrow, so he should not come out.
        it = gk_intent(5.0, 20.0)
        self.assertLess(it.tx, 4.0)

    def test_stays_on_his_line_at_the_very_edge_of_the_fifth(self):
        it = gk_intent(11.0, 20.0)
        self.assertLessEqual(it.tx, DEFENSIVE_FIFTH - 1.0 + 1e-6)


class SafetyTests(unittest.TestCase):
    def test_never_asks_to_tackle(self):
        # RULES.md:20 a goalkeeper cannot use `tackle`.
        for ball_x, ball_y in ((6.0, 20.0), (20.0, 4.0), (20.0, 36.0)):
            it = gk_intent(ball_x, ball_y)
            self.assertNotEqual(it.action_type, "tackle")

    def test_intents_are_finite_and_in_range(self):
        for ball_x in (1.0, 12.0, 30.0, 59.0):
            for ball_y in (1.0, 20.0, 39.0):
                it = gk_intent(ball_x, ball_y)
                self.assertTrue(0.0 <= it.tx <= 60.0)
                self.assertTrue(0.0 <= it.ty <= 40.0)
                self.assertTrue(0.0 <= it.speed <= 1.0)

    def test_does_not_act_on_a_ball_it_does_not_have(self):
        it = gk_intent(50.0, 20.0, possess="us")
        self.assertNotIn(it.action_type, ("pass", "shoot", "clear"))


if __name__ == "__main__":
    unittest.main()
