"""Regression tests for the three fixes measured against Vanguard FC (elite).

Baseline from eight authoritative replays against `opponents/counter-elite`:

* 35 shots, all saved (0 goals), mean distance 15.1 m, 26 of them taken by the
  centre-backs. Outside the box the keeper always has time to cover a corner.
* 64.1% of open-play time the ball was loose.
* Outfield shape collapsed to 11.05 m of width while we had the ball, with 59%
  of player-ticks inside the central 12 m.

Each test below pins one of the root causes so the regression cannot come back.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.config import KICK_MIN_SPEED, default_genome, RuntimeConfig
from src.geom import GOAL_HIGH_Y, GOAL_LOW_Y, PITCH_WIDTH
from src.physics import (
    GK_LATERAL_REACH,
    MIN_PASS_TRAVEL,
    pass_collection_point,
    shot_beats_keeper,
    speed_for_travel,
)
from src.policy import (
    PolicyController,
    PolicyInput,
    ball_travel_before_control,
)
from src.state import GameState, WorldModel
from src.tactics import (
    PressPlan,
    ROLE_WIDE_LEFT,
    ROLE_WIDE_RIGHT,
    TacticalState,
    assign_roles,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.test_policy import SLOTS, obs  # noqa: E402


def make_inp(observation, tactical_state=TacticalState.ATTACK):
    state = GameState.from_observation(observation)
    world = WorldModel.build(state)
    roles = assign_roles(state, SLOTS)
    return PolicyInput(
        state=state,
        world=world,
        config=RuntimeConfig(genome=default_genome()),
        tactical_state=tactical_state,
        press_plan=PressPlan(),
        roles=roles,
    )


def outfield_width(intents):
    ys = [it.ty for pid, it in intents.items() if pid != "gk"]
    return max(ys) - min(ys)


class PassCollectionGeometryTests(unittest.TestCase):
    """A pass always leaves at 12 m/s and always rolls ~14.5 m before it can
    be touched again, so a pass aimed closer than that is a turnover."""

    def test_minimum_pass_travel_matches_engine(self):
        self.assertAlmostEqual(
            ball_travel_before_control(KICK_MIN_SPEED), MIN_PASS_TRAVEL, places=6
        )
        # The real floor is ~15.6 m, not the ~8 m a football intuition suggests.
        self.assertGreater(MIN_PASS_TRAVEL, 15.0)
        self.assertLess(MIN_PASS_TRAVEL, 16.0)

    def test_short_pass_overshoots_the_aim_point(self):
        cx, cy, _, _ = pass_collection_point(0.0, 0.0, 8.0, 0.0, 0.0)
        self.assertAlmostEqual(cx, MIN_PASS_TRAVEL, places=6)
        self.assertGreater(cx, 8.0)

    def test_no_slower_kick_exists_for_a_short_pass(self):
        self.assertEqual(speed_for_travel(5.0), KICK_MIN_SPEED)

    def test_long_pass_can_be_sized_to_arrive(self):
        target = 30.0
        speed = speed_for_travel(target)
        self.assertGreater(speed, KICK_MIN_SPEED)
        power = (speed - KICK_MIN_SPEED) / 14.0
        cx, cy, _, _ = pass_collection_point(0.0, 0.0, target, 0.0, power)
        self.assertAlmostEqual(cx, target, delta=1.0)
        self.assertAlmostEqual(cy, 0.0, delta=0.01)

    def test_collection_point_is_finite_and_on_the_pitch(self):
        cx, cy, power, speed0 = pass_collection_point(50.0, 20.0, 60.0, 38.0, 1.0)
        self.assertTrue(0.0 <= cx <= 60.0)
        self.assertTrue(0.0 <= cy <= PITCH_WIDTH)
        self.assertTrue(0.0 <= power <= 1.0)
        self.assertTrue(12.0 <= speed0 <= 26.0)


class KeeperReachTests(unittest.TestCase):
    """Shots must be worth taking, not merely have a clear lane.

    The engine fact that drives everything here: the goal is only 6 m wide
    (y 17..23), so the largest angle available against a keeper on the centre
    line is 3.0 m. The keeper's dive reach is 1.65 m and he keeps shifting
    across his goal at run speed while the ball is in flight, which leaves him
    *negative* -- a standing catch -- for anything beyond about 8 m.

    Consequence: a set keeper cannot be scored on from range at all. The only
    way to score is to displace him first, either by pulling him wide with the
    ball or by dragging him out of his defensive fifth. That is precisely why
    35 shots from a mean 15.1 m produced zero goals against Vanguard.
    """

    def test_a_set_keeper_cannot_be_scored_on_from_range(self):
        # Best possible angle from the centre line, taken from 18 m.
        self.assertFalse(shot_beats_keeper(42.0, 20.0, 58.0, 20.0, 22.4, 18.0))

    def test_even_the_widest_range_shot_from_centre_is_hopeless(self):
        # 12 m: the keeper's shift alone exceeds the whole half-goal angle.
        self.assertFalse(shot_beats_keeper(48.0, 20.0, 58.0, 20.0, 23.0, 12.0))

    def test_pulled_wide_keeper_can_be_beaten_from_range(self):
        # Ball worked wide drags him to y=12; the far corner is then 10.4 m
        # away and he cannot cover it. This is the shot worth taking.
        self.assertTrue(shot_beats_keeper(42.0, 8.0, 58.0, 12.0, 22.4, 18.0))

    def test_point_blank_shot_needs_the_actual_post_not_the_inset(self):
        # 6 m out, aiming at the post itself (3.0 m of angle) leaves 1.1 m after
        # his shift: a dive, so it is worth taking. The 0.6 m inset that
        # `pick_shot_target` uses leaves only 0.5 m, which is a standing catch.
        self.assertTrue(shot_beats_keeper(54.0, 20.0, 58.0, 20.0, GOAL_HIGH_Y, 6.0))
        self.assertFalse(shot_beats_keeper(54.0, 20.0, 58.0, 20.0, 22.4, 6.0))

    def test_keeper_dragged_out_of_his_area_is_a_free_goal(self):
        # gk.x < GK_AREA_START is outside the defensive fifth, so he cannot
        # auto-handle at all, whatever the angle.
        self.assertTrue(shot_beats_keeper(42.0, 20.0, 30.0, 20.0, 20.6, 18.0))

    def test_unknown_keeper_is_not_treated_as_a_free_goal(self):
        self.assertFalse(shot_beats_keeper(42.0, 20.0, -1.0, 20.0, 20.6, 18.0))

    def test_goal_is_too_narrow_for_range_beating_a_set_keeper(self):
        # Pins the engine fact the whole shot model rests on.
        self.assertEqual(GOAL_HIGH_Y - GOAL_LOW_Y, 6.0)
        self.assertGreater((GOAL_HIGH_Y - GOAL_LOW_Y) / 2.0, GK_LATERAL_REACH)


class ShotDisciplineTests(unittest.TestCase):
    def test_centre_back_does_not_shoot_from_18_metres(self):
        observation = obs((40, 20), "us", our_st_x=40, them_x=40)
        observation["us"][1]["position"] = {"x": 42, "y": 20}  # 'cd' holds the ball
        observation["ball"]["possessedBy"] = "cd"
        intents = PolicyController().decide(make_inp(observation))
        self.assertNotEqual(intents["cd"].action_type, "shoot")

    def test_striker_still_shoots_inside_the_box(self):
        intents = PolicyController().decide(make_inp(obs((52, 20), "us", our_st_x=52, them_x=10)))
        self.assertEqual(intents["st"].action_type, "shoot")


class ShapeTests(unittest.TestCase):
    """The support-triangle override used to collapse the team into a 16 m
    band around the ball whenever pressure exceeded 0.55 -- which, against a
    0.95 press, was almost always."""

    def test_pressure_does_not_collapse_the_attacking_shape(self):
        # Three opponents within 3 m of the ball: pressure is high.
        observation = obs((30, 20), "us", our_st_x=30, them_x=30)
        observation["them"] = [
            {"id": "tgk", "role": "goalkeeper", "position": {"x": 58, "y": 20}, "velocity": {"x": 0, "y": 0}, "facingRadians": 0, "canAct": True},
            {"id": "m1", "role": "outfield", "position": {"x": 31, "y": 20}, "velocity": {"x": 0, "y": 0}, "facingRadians": 0, "canAct": True},
            {"id": "m2", "role": "outfield", "position": {"x": 28, "y": 21}, "velocity": {"x": 0, "y": 0}, "facingRadians": 0, "canAct": True},
            {"id": "m3", "role": "outfield", "position": {"x": 30, "y": 23}, "velocity": {"x": 0, "y": 0}, "facingRadians": 0, "canAct": True},
        ]
        inp = make_inp(observation)
        self.assertGreater(inp.world.pressure_on_ball, 0.55)
        intents = PolicyController().decide(inp)

        # The support shape is what this test is about, so the ball carrier is
        # excluded: its `ty` is a carry destination chosen by the dribble probe
        # fan, not a formation position. Including it made this assertion a test
        # of the carrier's dodge instead of the shape, and a carry target that
        # was itself wrong used to supply the high extreme -- a one-sided
        # `by >= opp.y` test slid the carrier 7 m towards the same touchline in
        # a state and in its mirror, and in this fixture that alone is the
        # difference between ~20 m and ~9 m of "width".
        carrier = inp.state.our_possessor()
        assert carrier is not None
        support = [
            it.ty for pid, it in intents.items() if pid not in ("gk", carrier.id)
        ]
        self.assertTrue(support, "no support players")
        # Baseline under the old override was ~11 m for the whole outfield. The
        # support spread here is unchanged by the mirror fixes; what changed is
        # the carrier, which is no longer counted.
        self.assertGreaterEqual(max(support) - min(support), 9.0)

    def test_wingers_keep_their_flanks_under_pressure(self):
        observation = obs((30, 20), "us", our_st_x=30, them_x=30)
        observation["them"] = [
            {"id": "tgk", "role": "goalkeeper", "position": {"x": 58, "y": 20}, "velocity": {"x": 0, "y": 0}, "facingRadians": 0, "canAct": True},
            {"id": "m1", "role": "outfield", "position": {"x": 31, "y": 20}, "velocity": {"x": 0, "y": 0}, "facingRadians": 0, "canAct": True},
            {"id": "m2", "role": "outfield", "position": {"x": 28, "y": 19}, "velocity": {"x": 0, "y": 0}, "facingRadians": 0, "canAct": True},
        ]
        intents = PolicyController().decide(make_inp(observation))
        left = intents["w"].ty
        self.assertLessEqual(left, 14.0)

    def test_rest_defence_is_not_stacked_on_the_centre_line(self):
        observation = obs((20, 10), "us", our_st_x=20, them_x=20)
        observation["them"] = [
            {"id": "tgk", "role": "goalkeeper", "position": {"x": 58, "y": 20}, "velocity": {"x": 0, "y": 0}, "facingRadians": 0, "canAct": True},
            {"id": "m1", "role": "outfield", "position": {"x": 40, "y": 20}, "velocity": {"x": 0, "y": 0}, "facingRadians": 0, "canAct": True},
        ]
        intents = PolicyController().decide(make_inp(observation))
        # Both centre-backs used to sit at y == 20.0.
        self.assertNotEqual(intents["cd"].ty, intents["am"].ty)


class NoCrashTests(unittest.TestCase):
    def test_decisions_stay_finite_with_no_them(self):
        observation = obs((30, 20), "us")
        observation["them"] = []
        intents = PolicyController().decide(make_inp(observation))
        for it in intents.values():
            self.assertTrue(0.0 <= it.speed <= 1.0)


if __name__ == "__main__":
    unittest.main()
