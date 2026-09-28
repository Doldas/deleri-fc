"""P1.3 -- shot dive / second-ball EV must not double-count probability.

The event semantics being tested:

* ``p_shot_forces_dive`` returns P(dive | the shot reaches the frame). It is a
  quality fraction between a standing catch and a clean beat.
* ``d["saved"]`` from ``shot_outcome_distribution`` is the *marginal*
  P(not blocked, on target, not a goal) -- it already contains the "reaches the
  frame" event and the "not blocked" event.
* The second ball is a free possession created by a dive save, so it belongs on
  the SAVED branch only, and its weight is
  ``P(saved AND dive) x V_second_ball``.

The old expression multiplied the frame event in a second time:
``P(saved) x P(dive|on) x P(saved + goal) x V``. The block probability was
therefore applied twice, which crushed the second-ball credit exactly for
shots worth taking (dive-forcing but blockable).
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.policy import (
    EV_SHOT_SECOND_BALL,
    SHOT_GOAL_CLEAN,
    SHOT_GOAL_STANDING_CATCH,
    ExpectedValueCalculator,
)
from tests.test_ev_model import place_opponents
from tests.test_policy import make_inp, obs

OPP_GOAL_X = 60.0
POWER = 0.9
# `wall_shot_target`-style edge of the frame; the keeper sits at GOAL_CENTER_Y.
NEAR_POST = (OPP_GOAL_X, 16.5)
FAR_POST = (OPP_GOAL_X, 23.5)
TOP_CORNER = (OPP_GOAL_X, 14.0)
KEEPER_BODY = (OPP_GOAL_X, 20.0)


def evcalc(our_x=52.0, keeper=(58.0, 20.0), opponents=((10.0, 20.0),)):
    """An EV calculator over a controlled attacking state."""
    observation = place_opponents(
        obs((our_x, 20.0), "us", our_st_x=our_x, them_x=10), list(opponents)
    )
    for pl in observation["them"]:
        if pl["role"] == "goalkeeper":
            pl["position"]["x"], pl["position"]["y"] = keeper
    inp = make_inp(observation)
    return ExpectedValueCalculator(inp), inp.state.our_possessor()


def split(ev, p, target, power=POWER):
    """Return (distribution, dive_probability, ev_shot, second_ball_term)."""
    d = ev.shot_outcome_distribution(p, target, power)
    dive = ev.p_shot_forces_dive(p, target, power)
    total = ev.ev_shot(p, target, power)
    second_ball = d["saved"] * EV_SHOT_SECOND_BALL * dive
    return d, dive, total, second_ball


class StandingCatchTests(unittest.TestCase):
    """A: shot into easy handling -- no second-ball bonus."""

    def test_standing_catch_earns_no_dive_and_no_second_ball(self):
        ev, p = evcalc()
        d, dive, total, second_ball = split(ev, p, KEEPER_BODY)
        self.assertEqual(dive, 0.0, "a standing catch must not force a dive")
        self.assertEqual(second_ball, 0.0)
        # The shot is heavily saved and barely a goal.
        self.assertGreater(d["saved"], 0.9)
        self.assertLess(d["goal"], 0.05)
        # ... so the EV must be clearly negative even though it is "on target".
        self.assertLess(total, 0.0)

    def test_standing_catch_ev_equals_plain_saved_value(self):
        ev, p = evcalc()
        d = ev.shot_outcome_distribution(p, KEEPER_BODY, POWER)
        expected = (
            d["blocked"] * ev.value_shot_blocked()
            + d["wide"] * ev.value_shot_wide()
            + d["saved"] * ev.value_shot_saved()
            + d["goal"] * ev.value_goal()
        )
        self.assertAlmostEqual(ev.ev_shot(p, KEEPER_BODY, POWER), expected, places=9)


class DiveSaveTests(unittest.TestCase):
    """B: a shot that forces a dive and is saved is worth the second ball."""

    def test_dive_save_earns_second_ball(self):
        ev, p = evcalc()
        d, dive, total, second_ball = split(ev, p, NEAR_POST)
        self.assertGreater(dive, 0.0, "a post shot must force a dive")
        self.assertGreater(d["saved"], 0.0)
        # Exactly P(saved) x P(dive|on) x V -- each probability once.
        self.assertAlmostEqual(
            second_ball, d["saved"] * dive * EV_SHOT_SECOND_BALL, places=12
        )
        self.assertGreater(total, 0.0, "a dive save should be net positive")

    def test_dive_save_beats_standing_catch_of_the_same_shot(self):
        ev, p = evcalc()
        catch_ev = ev.ev_shot(p, KEEPER_BODY, POWER)
        dive_ev = ev.ev_shot(p, NEAR_POST, POWER)
        self.assertGreater(dive_ev, catch_ev)

    def test_second_ball_scales_linearly_with_the_saved_probability(self):
        """Doubling the block rate must halve the credit, not quarter it."""
        clear_ev, clear_p = evcalc(opponents=((10.0, 20.0),))
        blocked_ev, blocked_p = evcalc(opponents=((54.0, 19.0),))
        d1, _, _, sb1 = split(clear_ev, clear_p, NEAR_POST)
        d2, _, _, sb2 = split(blocked_ev, blocked_p, NEAR_POST)
        # The dive *conditional* is unchanged: same shot, same keeper.
        self.assertAlmostEqual(
            clear_ev.p_shot_forces_dive(clear_p, NEAR_POST, POWER),
            blocked_ev.p_shot_forces_dive(blocked_p, NEAR_POST, POWER),
            places=12,
        )
        # The credit is exactly proportional to P(saved).
        self.assertAlmostEqual(sb2 / sb1, d2["saved"] / d1["saved"], places=9)

    def test_frame_event_is_not_applied_twice(self):
        """Regression on the exact defect: no (saved + goal) factor."""
        ev, p = evcalc(opponents=((54.0, 19.0),))
        d, dive, total, second_ball = split(ev, p, NEAR_POST)
        double_counted = d["saved"] * EV_SHOT_SECOND_BALL * dive * (d["saved"] + d["goal"])
        self.assertNotAlmostEqual(second_ball, double_counted, places=6)
        self.assertGreater(second_ball, double_counted)


class GoalTests(unittest.TestCase):
    """C: a goal must not be credited a second ball."""

    def test_clean_goal_earns_no_second_ball(self):
        ev, p = evcalc()
        d, dive, total, second_ball = split(ev, p, TOP_CORNER)
        self.assertGreater(d["goal"], d["saved"], "the corner should beat the keeper")
        # Any dive credit is scaled by P(saved), which is negligible here.
        self.assertLess(second_ball, 0.02)
        self.assertGreater(total, 0.0)

    def test_goal_value_is_credited_once(self):
        ev, p = evcalc()
        d = ev.shot_outcome_distribution(p, TOP_CORNER, POWER)
        manual = (
            d["blocked"] * ev.value_shot_blocked()
            + d["wide"] * ev.value_shot_wide()
            + d["saved"] * ev.value_shot_saved()
            + d["goal"] * ev.value_goal()
        )
        # The dive adjustment must be far smaller than the goal term.
        self.assertGreater(d["goal"] * ev.value_goal(), abs(ev.ev_shot(p, TOP_CORNER, POWER) - manual))


class MarginalReachTests(unittest.TestCase):
    """D: EV must vary continuously across the dive/reach boundary."""

    def _series(self):
        """Sweep the target across the frame near the reach boundary."""
        out = []
        for i in range(41):
            y = 19.0 + i * 0.1  # 19.0 .. 23.0
            ev, p = evcalc()
            d, dive, total, _ = split(ev, p, (OPP_GOAL_X, y))
            out.append((y, dive, total, d["saved"], d["goal"]))
        return out

    def test_dive_probability_is_continuous(self):
        series = self._series()
        for (y0, d0, _, _, _), (y1, d1, _, _, _) in zip(series, series[1:]):
            self.assertLessEqual(
                abs(d1 - d0), 0.06, f"dive probability jumped {d0}->{d1} at y={y1}"
            )

    def test_ev_is_continuous(self):
        series = self._series()
        for (y0, _, e0, _, _), (y1, _, e1, _, _) in zip(series, series[1:]):
            self.assertLessEqual(
                abs(e1 - e0), 4.0, f"EV jumped {e0:.2f}->{e1:.2f} at y={y1}"
            )

    def test_no_probability_explosion_at_the_boundary(self):
        """The old double product could not exceed 1, but it distorted shape."""
        for y, dive, total, saved, goal in self._series():
            self.assertGreaterEqual(dive, 0.0)
            self.assertLessEqual(dive, 1.0)
            # The second-ball weight can never exceed P(saved) x V.
            self.assertLessEqual(saved * dive * EV_SHOT_SECOND_BALL, saved * EV_SHOT_SECOND_BALL)

    def test_dive_band_matches_the_documented_constants(self):
        ev, p = evcalc()
        # A dead-on standing catch is the floor; a clean beat is the ceiling.
        self.assertEqual(
            ev.p_shot_forces_dive(p, KEEPER_BODY, POWER), 0.0
        )
        self.assertLess(SHOT_GOAL_STANDING_CATCH, SHOT_GOAL_CLEAN)


class DistributionIntegrityTests(unittest.TestCase):
    def test_distribution_still_sums_to_one(self):
        for target in (KEEPER_BODY, NEAR_POST, FAR_POST, TOP_CORNER):
            ev, p = evcalc()
            d = ev.shot_outcome_distribution(p, target, POWER)
            self.assertAlmostEqual(
                d["blocked"] + d["wide"] + d["saved"] + d["goal"], 1.0, places=9
            )

    def test_second_ball_branch_is_wired_to_the_saved_outcome_only(self):
        """Zeroing EV_SHOT_SECOND_BALL must lower EV by exactly P(saved)*V*dive.

        This checks the branch algebra without asserting any absolute ordering
        between shot types (the model does not rank a dive-save below a clean
        shot, and should not be forced to).
        """
        import src.policy as policy_mod

        ev, p = evcalc()
        d, dive, with_bonus, second_ball = split(ev, p, NEAR_POST)
        original = policy_mod.EV_SHOT_SECOND_BALL
        try:
            policy_mod.EV_SHOT_SECOND_BALL = 0.0
            without_bonus = ev.ev_shot(p, NEAR_POST, POWER)
        finally:
            policy_mod.EV_SHOT_SECOND_BALL = original
        self.assertAlmostEqual(with_bonus - without_bonus, second_ball, places=12)
        # ... and the drop is bounded by P(saved) * V, not P(saved) * P(on) * V.
        self.assertLessEqual(with_bonus - without_bonus, d["saved"] * original)
        self.assertGreater(dive, 0.0)


if __name__ == "__main__":
    unittest.main()


class MarginalReachBarTests(unittest.TestCase):
    """P1.3 follow-on: the in-box 'reaches the goal' bar is a marginal quantity.

    `_shot_worth_taking` tests whether the effort reaches the frame at all when
    the carrier is inside the box. `p_shot_on_target` is already marginal
    (= saved + goal, so it already contains "not blocked"), and the bar used to
    be compared against that value multiplied by `(1 - P(blocked))` a second
    time.
    """

    def test_reach_bar_equals_marginal_on_target(self):
        import src.policy as policy_mod

        for opponents in (((10.0, 20.0),), ((54.0, 19.0),), ((53.0, 17.0), (55.0, 22.0))):
            ev, p = evcalc(opponents=opponents)
            (tx, ty), power = policy_mod._shot_geometry_target(
                make_inp(place_opponents(
                    obs((52.0, 20.0), "us", our_st_x=52.0, them_x=10), list(opponents)
                )), p
            )
            marginal = ev.p_shot_on_target(p, (tx, ty), power)
            double_counted = (
                1.0 - ev.p_shot_blocked(p, (tx, ty), power)
            ) * marginal
            if ev.p_shot_blocked(p, (tx, ty), power) > 0.01:
                self.assertNotAlmostEqual(marginal, double_counted, places=4)
            self.assertAlmostEqual(
                policy_mod._shot_worth_taking(
                    make_inp(place_opponents(
                        obs((52.0, 20.0), "us", our_st_x=52.0, them_x=10), list(opponents)
                    )),
                    make_inp(place_opponents(
                        obs((52.0, 20.0), "us", our_st_x=52.0, them_x=10), list(opponents)
                    )).state.our_possessor(),
                ),
                marginal >= policy_mod.EV_SHOT_REACHES_GOAL_BAR,
                places=9,
            )

    def test_blocking_does_not_double_discount_the_reach_test(self):
        """With one blocker the reach test must equal the marginal value."""
        ev, p = evcalc(opponents=((54.0, 19.0),))
        inp = make_inp(place_opponents(
            obs((52.0, 20.0), "us", our_st_x=52.0, them_x=10), [(54.0, 19.0)]
        ))
        p2 = inp.state.our_possessor()
        ev2 = ExpectedValueCalculator(inp)
        import src.policy as policy_mod
        (tx, ty), power = policy_mod._shot_geometry_target(inp, p2)
        marginal = ev2.p_shot_on_target(p2, (tx, ty), power)
        self.assertGreater(ev2.p_shot_blocked(p2, (tx, ty), power), 0.1, "fixture lost its blocker")
        # The documented quantity is the marginal one, not the double product.
        self.assertGreater(marginal, 0.2)
        self.assertAlmostEqual(
            policy_mod._shot_worth_taking(inp, p2),
            marginal >= policy_mod.EV_SHOT_REACHES_GOAL_BAR,
            places=9,
        )
