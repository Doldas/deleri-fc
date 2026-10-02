import copy
import math
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.tactics import TacticalState, assign_roles, PressPlan
from src.config import RuntimeConfig, default_genome, KICK_MIN_SPEED, KICK_MAX_SPEED, MAX_RUN_SPEED
from src.policy import (
    PolicyController,
    PolicyInput,
    match_context,
    ball_travel_before_control,
    loose_ball_meeting_point,
    CARRY_CLEAN_VALUE,
    CARRY_HOLD_VALUE,
    PlayerIntent,
    ActionCandidate,
)
from src.opponent import OpponentModel
from src.state import GameState, WorldModel
from src import geom
from src.geom import PITCH_LENGTH, PITCH_WIDTH, GOAL_CENTER_Y, OPP_GOAL_X
from src.physics import shot_beats_keeper, MIN_PASS_TRAVEL
from src.teamplan import TeamPlan

SLOTS = [
    {"id": "defender", "role": "defender", "position": {"x": 14, "y": 20}},
    {"id": "left", "role": "winger", "position": {"x": 28, "y": 8}},
    {"id": "right", "role": "winger", "position": {"x": 28, "y": 32}},
    {"id": "striker", "role": "striker", "position": {"x": 43, "y": 20}},
]


def obs(ball_pos, possess, our_st_x=40, them_x=14):
    return {
        "protocolVersion": "1.0",
        "gameId": "p-test",
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
            {"id": "cd", "role": "outfield", "position": {"x": 18, "y": 20}, "velocity": {"x": 0, "y": 0}, "facingRadians": 0, "canAct": True},
            {"id": "am", "role": "outfield", "position": {"x": 32, "y": 16}, "velocity": {"x": 0, "y": 0}, "facingRadians": 0, "canAct": True},
            {"id": "w", "role": "outfield", "position": {"x": 30, "y": 6}, "velocity": {"x": 0, "y": 0}, "facingRadians": 0, "canAct": True},
            {"id": "st", "role": "outfield", "position": {"x": our_st_x, "y": 20}, "velocity": {"x": 0, "y": 0}, "facingRadians": 0, "canAct": True},
        ],
        "them": [
            {"id": "tgk", "role": "goalkeeper", "position": {"x": 58, "y": 20}, "velocity": {"x": 0, "y": 0}, "facingRadians": 0, "canAct": True},
            {"id": "cd", "role": "outfield", "position": {"x": them_x, "y": 20}, "velocity": {"x": 0, "y": 0}, "facingRadians": 0, "canAct": True},
            {"id": "am", "role": "outfield", "position": {"x": them_x - 6, "y": 16}, "velocity": {"x": 0, "y": 0}, "facingRadians": 0, "canAct": True},
            {"id": "st", "role": "outfield", "position": {"x": them_x + 8, "y": 24}, "velocity": {"x": 0, "y": 0}, "facingRadians": 0, "canAct": True},
        ],
    }


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


class PolicyTests(unittest.TestCase):
    def test_every_player_gets_a_valid_intent(self):
        inp = make_inp(obs((35, 20), "us"))
        intents = PolicyController().decide(inp)
        for pid in ("gk", "cd", "am", "w", "st"):
            self.assertIn(pid, intents)
            it = intents[pid]
            self.assertGreaterEqual(it.speed, 0.0)
            self.assertLessEqual(it.speed, 1.0)
            self.assertEqual(it.pid, pid)

    def test_possessor_near_open_goal_shoots(self):
        inp = make_inp(obs((52, 20), "us", our_st_x=52, them_x=10))
        intents = PolicyController().decide(inp)
        self.assertEqual(intents["st"].action_type, "shoot")
        self.assertIsNotNone(intents["st"].action_target)

    def test_no_shot_is_aimed_at_the_keeper_body(self):
        """A shot at the goal centre is a shot at the keeper.

        The engine treats a save needing less than GK_DIVE_LATERAL_MIN (0.8 m)
        of lateral reach as a standing catch, and a keeper holding his line
        sits on the centre. So an intent aimed at GOAL_CENTER_Y is only a real
        chance when shot_beats_keeper agrees.

        This is the invariant a "rebound setup" candidate broke: it aimed at
        GOAL_CENTER_Y from up to 20 m, never consulted shot_beats_keeper, and
        was worth 45.0 -- tying the best genuine shot in the planner, so the
        striker could prefer feeding the keeper over actually scoring.
        """
        for st_x in (36, 40, 44, 48, 52):
            with self.subTest(dist_goal=OPP_GOAL_X - st_x):
                inp = make_inp(obs((st_x, 20), "us", our_st_x=st_x, them_x=10))
                intents = PolicyController().decide(inp)
                gk = inp.state.goalkeeper_them()
                if gk is None:
                    self.fail("test needs an opponent keeper on the pitch")
                dist_goal = OPP_GOAL_X - st_x
                for pid, it in intents.items():
                    if it.action_type != "shoot" or it.action_target is None:
                        continue
                    _, ty = it.action_target
                    if abs(ty - GOAL_CENTER_Y) > 0.01:
                        continue  # aimed at a corner, not at the keeper
                    self.assertTrue(
                        shot_beats_keeper(
                            st_x, 20.0, gk.x, gk.y, ty, dist_goal,
                            it.action_power or 0.9,
                        ),
                        f"{pid} was chosen to shoot at the keeper's body from "
                        f"{dist_goal:.1f} m, a shot the keeper can hold",
                    )

    def test_carry_value_scores_progress_not_distance_remaining(self):
        """carry_value was max(5.0, (OPP_GOAL_X - p.x) * 0.1) -- "progress toward goal".

        (OPP_GOAL_X - p.x) is the distance *still to run*, so the value fell as
        the carrier advanced and the 5.0 floor then bound for every x >= 10: the
        same number at the halfway line and five metres out. That made the
        lowest-valued candidate in the planner the one the loose-ball physics
        note in src/policy.py calls the only reliable way to keep possession,
        and it lost to safe_pass's 10.0 in every state where a pass merely
        happened to be legal.
        """
        c = PolicyController()
        # Identical 8 m of forward ground; the only difference is a marker
        # standing in the lane we picked.
        clean_inp = make_inp(obs((30, 20), "us", our_st_x=30, them_x=10))
        busy_inp = make_inp(obs((30, 20), "us", our_st_x=30, them_x=34))
        p_clean = [q for q in clean_inp.state.outfield_us() if q.id == "st"][0]
        p_busy = [q for q in busy_inp.state.outfield_us() if q.id == "st"][0]

        clean = c._carry_value(clean_inp, p_clean, 38.0, 20.0)
        busy = c._carry_value(busy_inp, p_busy, 38.0, 20.0)
        self.assertAlmostEqual(clean, CARRY_CLEAN_VALUE, places=6)
        self.assertLess(busy, clean, "a contested lane must be worth less than a clean one")

        # Forward ground is what pays, so a dribble cut short by half is worth
        # proportionally less.
        self.assertLess(c._carry_value(clean_inp, p_clean, 34.0, 20.0), clean)

        # Going nowhere forward is the old floor, not zero: still keep the ball.
        self.assertAlmostEqual(
            c._carry_value(clean_inp, p_clean, 30.0, 20.0), CARRY_HOLD_VALUE, places=6)
        self.assertAlmostEqual(
            c._carry_value(clean_inp, p_clean, 26.0, 20.0), CARRY_HOLD_VALUE, places=6)

        # And it must be able to beat a collectable pass (10.0), or the old bug
        # returns wearing a different constant.
        self.assertGreater(clean, 10.0)

    def test_carry_beats_a_collectable_pass_on_a_clean_lane(self):
        """Behavioural half of the carry valuation, not just the arithmetic.

        A pass is only collectable once it has slowed below the control limit,
        so a collectable pass is still a race to win the ball back, while a
        dribble keeps it glued 0.65 m in front. The carrier here has a clean
        forward lane *and* a collectable pass available; it should carry.
        """
        inp = make_inp(obs((30, 20), "us", our_st_x=30, them_x=10))
        c = PolicyController()
        p = [q for q in inp.state.outfield_us() if q.id == "st"][0]

        # Guard the premise: if no pass were available the test would prove
        # nothing about beating one.
        self.assertIsNotNone(c._collectable_pass(inp, p), "fixture must offer a pass")
        tx, ty, _ = c._dribble_target(inp, p)
        self.assertEqual(c._segment_contest(inp.state, p.x, p.y, tx, ty), 0.0)

        c.decide(inp)
        reason = next(
            (e["reason"] for e in c.log.possessions_decided if e["player"] == "st"), None)
        self.assertEqual(reason, "carry_forward")

    def _box_carrier_obs(self, cx, cy):
        """One non-striker carrier, a central keeper, and three markers.

        The carrier is deliberately not the natural striker, so the main shoot
        path refuses -- it needs shot_beats_keeper, or a striker inside the box
        -- and only the "shoot on sight" rule can produce a shot here. That
        isolates the rule under test from IN_BOX_SHOT_FLOOR.
        """
        return {
            "protocolVersion": "1.0", "gameId": "sos", "sequence": 1,
            "simulationTick": 1, "applyAtTick": 1, "timeRemainingSeconds": 300,
            "phase": "openPlay", "score": {"us": 0, "them": 0},
            "ball": {"position": {"x": cx, "y": cy}, "velocity": {"x": 0, "y": 0},
                     "possessingTeam": "us", "possessedBy": "am"},
            "us": [
                {"id": "gk", "role": "goalkeeper", "position": {"x": 5, "y": 20}, "velocity": {"x": 0, "y": 0}, "facingRadians": 0, "canAct": True},
                {"id": "cd", "role": "outfield", "position": {"x": 18, "y": 20}, "velocity": {"x": 0, "y": 0}, "facingRadians": 0, "canAct": True},
                {"id": "am", "role": "outfield", "position": {"x": cx, "y": cy}, "velocity": {"x": 0, "y": 0}, "facingRadians": 0, "canAct": True},
                {"id": "w", "role": "outfield", "position": {"x": 44, "y": 6}, "velocity": {"x": 0, "y": 0}, "facingRadians": 0, "canAct": True},
                {"id": "st", "role": "outfield", "position": {"x": 50, "y": 26}, "velocity": {"x": 0, "y": 0}, "facingRadians": 0, "canAct": True},
            ],
            "them": [
                {"id": "tgk", "role": "goalkeeper", "position": {"x": 58, "y": 20}, "velocity": {"x": 0, "y": 0}, "facingRadians": 0, "canAct": True},
                {"id": "m1", "role": "outfield", "position": {"x": 54, "y": 16}, "velocity": {"x": 0, "y": 0}, "facingRadians": 0, "canAct": True},
                {"id": "m2", "role": "outfield", "position": {"x": 55, "y": 24}, "velocity": {"x": 0, "y": 0}, "facingRadians": 0, "canAct": True},
                {"id": "m3", "role": "outfield", "position": {"x": 54, "y": 20}, "velocity": {"x": 0, "y": 0}, "facingRadians": 0, "canAct": True},
            ],
        }

    def test_shoot_on_sight_applies_in_the_box_not_near_the_boundary(self):
        """"Shoot on sight in the box" was gated on is_near_wall(margin=4.0).

        wall.is_near_wall is a pitch-boundary test -- x <= m or x >= L - m or
        y <= m or y >= W - m -- so the rule fired only for a player hugging a
        touchline or the goal line and never for a central attacker in the box.
        Measured with this fixture, only the position varying, the behaviour
        was inverted: (52,20), 8 m out with 0.24 rad of goalmouth showing, got
        no shot at all, while (50,4), further out with a worse 0.17, shot for
        45.0.
        """
        from src.wall import is_near_wall
        from src.physics import shot_open_angle

        state = GameState.from_observation(self._box_carrier_obs(52.0, 20.0))
        # Guard the premise: the central carrier is inside the box and nowhere
        # near a boundary, so this cannot pass through the old gate.
        self.assertLessEqual(OPP_GOAL_X - 52.0, 11.0)
        self.assertFalse(is_near_wall(52.0, 20.0, margin=4.0))
        opps = [(q.x, q.y) for q in state.outfield_them()]
        self.assertGreaterEqual(shot_open_angle(52.0, 20.0, opps), 0.10)

        inp = make_inp(self._box_carrier_obs(52.0, 20.0))
        intents = PolicyController().decide(inp)
        self.assertEqual(intents["am"].action_type, "shoot",
                         "a non-striker 8 m out in the box with an open goal "
                         "must get the shoot-on-sight shot")

    def test_unreachable_pass_is_vetoed_so_the_ball_is_not_launched(self):
        """A pass aimed inside MIN_PASS_TRAVEL can never be collected.

        A kick leaves at KICK_MIN_SPEED (12 m/s) however little power is asked
        for, and the engine does not let anyone touch the ball until it has
        slowed to 5 m/s, so it unavoidably rolls MIN_PASS_TRAVEL (~15.6 m).
        Aiming at a point closer than that guarantees an overshoot, i.e. a
        turnover rather than a pass.

        Traced over 24 matches (186 executed passes): 66.7% were aimed inside
        MIN_PASS_TRAVEL, the p10/p25 aim distances were 0.00 m / 0.21 m, the
        median overshoot was 14.1 m, power sat at the floor in 68.8% of them,
        an opponent was closer to the landing point than any teammate in 58.1%,
        and 59.1% were intercepted. Our passes are 51% of every loose ball in a
        match, 97.1% of our possessions end with our own kick, and 67.8% of
        recoveries are re-lost within 0.5 s.

        With every pass target inside the minimum travel distance, the carrier
        must keep the ball rather than launch it out of reach.
        """
        from src.physics import MIN_PASS_TRAVEL

        def obs_with_mates(positions):
            o = self._box_carrier_obs(30.0, 20.0)
            o["ball"]["position"] = {"x": 30.0, "y": 20.0}
            for pid, (px, py) in positions.items():
                for p in o["us"]:
                    if p["id"] == pid:
                        p["position"] = {"x": px, "y": py}
            return o

        # Every teammate is within MIN_PASS_TRAVEL of the carrier at (30,20),
        # so no pass can be collected where it is aimed.
        close = {"cd": (24.0, 20.0), "w": (33.0, 14.0), "st": (34.0, 25.0)}
        for pid, (px, py) in close.items():
            self.assertLess(geom.distance(30.0, 20.0, px, py), MIN_PASS_TRAVEL)

        intents = PolicyController().decide(make_inp(obs_with_mates(close)))
        intent = intents["am"]
        if intent.action_type == "pass" and intent.action_target is not None:
            self.fail(
                "requested a pass aimed %.2f m away, inside the %.2f m minimum "
                "travel distance: the ball cannot be collected there"
                % (geom.distance(30.0, 20.0, intent.action_target[0],
                                 intent.action_target[1]), MIN_PASS_TRAVEL)
            )

    def test_wall_shot_is_range_bounded(self):
        """wall_shot had no upper distance bound and is_near_wall is a boundary test.

        `is_near_wall(x, y, m)` is `x <= m or x >= L - m or y <= m or y >= W - m`,
        so the wall-shot rule also fired for a defender hugging a touchline deep
        in our own defensive third. Traced: all 5 shot requests the team ever
        made were refused by the engine, and 2 were launched from x ~ 4-5 m --
        55 m from goal -- at power 0.95.
        """
        from src.wall import is_near_wall

        obs = self._box_carrier_obs(5.0, 4.0)
        obs["ball"]["position"] = {"x": 5.0, "y": 4.0}
        for p in obs["us"]:
            if p["id"] == "am":
                p["position"] = {"x": 5.0, "y": 4.0}
        # Premise: this carrier is deep in our own corner and next to a boundary.
        self.assertGreater(OPP_GOAL_X - 5.0, 40.0)
        self.assertTrue(is_near_wall(5.0, 4.0, margin=6.0))

        intents = PolicyController().decide(make_inp(obs))
        self.assertNotEqual(intents["am"].action_type, "shoot",
                            "must not request a shot from 55 m out just because "
                            "the carrier is near a touchline")

    def test_shoot_on_sight_no_longer_rewards_a_worse_angle(self):
        """The old gate preferred the boundary player on opening alone.

        (50,4) is 10 m out and only 0.17 rad open, yet it shot for 45.0 while
        the central (52,20) at 0.24 rad shot not at all. With the box as the
        gate, both shoot and the ordering no longer depends on which side of
        the pitch you happen to be standing on.
        """
        by_position = {}
        for cx, cy in ((52.0, 20.0), (50.0, 4.0)):
            inp = make_inp(self._box_carrier_obs(cx, cy))
            by_position[(cx, cy)] = PolicyController().decide(inp)["am"].action_type
            self.assertEqual(by_position[(cx, cy)], "shoot")
        self.assertEqual(by_position[(52.0, 20.0)], by_position[(50.0, 4.0)])

    def test_their_possession_no_crash_and_defensive(self):
        inp = make_inp(obs((22, 20), "them"), tactical_state=TacticalState.DEFENSIVE_TRANSITION)
        intents = PolicyController().decide(inp)
        for pid in ("gk", "cd", "am", "w", "st"):
            self.assertIn(pid, intents)

    def test_empty_them_handled(self):
        observation = obs((40, 20), "us")
        observation["them"] = []
        inp = make_inp(observation)
        intents = PolicyController().decide(inp)
        self.assertEqual(len(intents), 5)

    def test_intent_fields_are_finite(self):
        inp = make_inp(obs((48, 20), "us", our_st_x=46, them_x=8))
        intents = PolicyController().decide(inp)
        for it in intents.values():
            for v in (it.tx, it.ty, it.speed, it.face_x, it.face_y):
                self.assertEqual(v, v, "NaN in intent")
                self.assertNotEqual(v, float("inf"))
                self.assertNotEqual(v, float("-inf"))

    def test_opp_model_shades_rest_defence_toward_their_side(self):
        # They attack down their right (y high): the anchor must sit high.
        hi = OpponentModel()
        hi.samples = 10
        hi.side_bias = 8.0
        inp_hi = make_inp(obs((26, 20), "them"), tactical_state=TacticalState.DEFENSIVE_TRANSITION)
        inp_hi.opp = hi
        low = OpponentModel()
        low.samples = 10
        low.side_bias = -8.0
        inp_low = make_inp(obs((26, 20), "them"), tactical_state=TacticalState.DEFENSIVE_TRANSITION)
        inp_low.opp = low
        intents_hi = PolicyController().decide(inp_hi)
        intents_low = PolicyController().decide(inp_low)
        # A deep-outfield defender in rest defence lands on the anchor.
        anchor_hi = intents_hi["cd"]
        anchor_low = intents_low["cd"]
        self.assertGreater(anchor_hi.ty, 20.0)
        self.assertLess(anchor_low.ty, 20.0)

    def test_opp_model_press_intensity_deepens_line(self):
        """Keep the deep block against a high press.

        A high line between the ball and our own goal (x up to 40) was tried in
        place of this and measured far worse against Vanguard FC (elite):
        12 matches 1-0-11 and 1-13 goals, against 3-0-9 and 3-11 here. Their
        directness 0.75 and tempo 1.0 turn a high line into a ball over the
        top, and with four outfielders nobody is behind it. The empty middle
        third is cheaper than an exposed channel.
        """
        swarm = OpponentModel()
        swarm.press_samples = 10
        swarm.press_accum = 2.0  # min distance 2.0 -> intensity ~1.0
        inp = make_inp(obs((24, 20), "them"), tactical_state=TacticalState.DEFENSIVE_TRANSITION)
        inp.opp = swarm
        calm = OpponentModel()
        calm.press_samples = 10
        calm.press_accum = 100.0  # far away -> intensity ~0.0
        inp0 = make_inp(obs((24, 20), "them"), tactical_state=TacticalState.DEFENSIVE_TRANSITION)
        inp0.opp = calm
        line_swarm = PolicyController().decide(inp)["cd"]
        line_calm = PolicyController().decide(inp0)["cd"]
        self.assertLess(line_swarm.tx, line_calm.tx)

    def test_defensive_block_never_sinks_into_our_own_third(self):
        """Ceding the middle third is what lost these matches."""
        inp = make_inp(obs((8, 20), "them"), tactical_state=TacticalState.DEFENSIVE_TRANSITION)
        intent = PolicyController().decide(inp)["cd"]
        self.assertGreaterEqual(intent.tx, 20.0)

    def test_match_context_buckets(self):
        # Duration must be explicit: "late" is the last 20% of the configured
        # match, not a fixed number of seconds. See tests/test_match_context.py.
        self.assertEqual(match_context(300.0, 1, 0, 300.0), "leading")
        self.assertEqual(match_context(300.0, 0, 1, 300.0), "trailing")
        self.assertEqual(match_context(300.0, 0, 0, 300.0), "tied")
        self.assertEqual(match_context(60.0, 1, 0, 300.0), "leading_late")
        self.assertEqual(match_context(60.0, 0, 1, 300.0), "trailing_late")
        self.assertEqual(match_context(60.0, 0, 0, 300.0), "tied_late")
        # A 30 s match is never "late" before 6 s, whatever the old 120.0 said.
        self.assertEqual(match_context(30.0, 0, 0, 30.0), "tied")
        self.assertEqual(match_context(6.0, 0, 0, 30.0), "tied_late")

    def test_trailing_late_raises_risk_and_verticality(self):
        # 60 s remaining of a 300 s match is exactly the 20% late threshold.
        # This test only passed before because `late` was `tr <= 120.0`, which
        # made every state late; it now needs a real duration to be exercised.
        leading = make_inp(obs((35, 20), "us", them_x=20))
        leading.score_us = 2
        leading.score_them = 0
        leading.time_remaining = 60.0
        leading.match_duration = 300.0
        trailing = make_inp(obs((35, 20), "us", them_x=20))
        trailing.score_us = 0
        trailing.score_them = 2
        trailing.time_remaining = 60.0
        trailing.match_duration = 300.0
        self.assertEqual(leading.context(), "leading_late")
        self.assertEqual(trailing.context(), "trailing_late")
        # Estimate the modulated cone from the receivers of the ball carrier.
        cfg = default_genome()
        self.assertGreater(
            PolicyController()._cfg(trailing, "passing_risk", 0.4),
            PolicyController()._cfg(leading, "passing_risk", 0.4),
        )
        self.assertGreater(
            PolicyController()._cfg(trailing, "verticality", 0.55),
            PolicyController()._cfg(leading, "verticality", 0.55),
        )
        # Trailing late should be more aggressive (higher shooting threshold = longer range)
        # Leading late should be more conservative (lower shooting threshold)
        self.assertGreater(
            PolicyController()._cfg(trailing, "shooting_threshold", 0.5),
            PolicyController()._cfg(leading, "shooting_threshold", 0.5),
        )


class LooseBallPhysicsTests(unittest.TestCase):
    """Locks in the engine constraint behind the Round-4 rebuild.

    RULES.md: a free ball is collectable only below 5 m/s, but every "pass"
    action releases it at KICK_MIN_SPEED (12 m/s) or more. A pass therefore
    always rolls for metres before anybody may touch it, which is why the
    previous policy left the ball loose for 81% of every match.
    """

    def test_a_pass_always_rolls_farther_than_a_receiver_can_be_reached(self):
        roll = ball_travel_before_control(KICK_MIN_SPEED)
        self.assertGreater(roll, 10.0)
        # A runner covers at most MAX_RUN_SPEED * flight time, so the ball
        # outruns anyone starting from rest behind it.
        self.assertGreater(roll, MAX_RUN_SPEED * 1.0)

    def test_harder_passes_roll_even_further(self):
        speeds = [KICK_MIN_SPEED, 16.0, 20.0, KICK_MAX_SPEED]
        rolls = [ball_travel_before_control(s) for s in speeds]
        self.assertEqual(rolls, sorted(rolls))
        self.assertGreater(rolls[-1], rolls[0] * 2.0)

    def test_slow_ball_needs_no_travel(self):
        self.assertEqual(ball_travel_before_control(3.0), 0.0)
        self.assertEqual(ball_travel_before_control(0.0), 0.0)

    def test_meeting_point_is_downstream_of_the_ball(self):
        mx, my, seconds = loose_ball_meeting_point(30.0, 20.0, 12.0, 0.0)
        self.assertGreater(mx, 30.0)
        self.assertAlmostEqual(my, 20.0)
        self.assertGreater(seconds, 0.0)

    def test_meeting_point_never_exceeds_running_reach(self):
        mx, _, seconds = loose_ball_meeting_point(30.0, 20.0, 26.0, 0.0)
        self.assertLessEqual(mx - 30.0, MAX_RUN_SPEED * seconds + 1e-6)

    def test_already_slow_ball_is_collected_where_it_lies(self):
        self.assertEqual(loose_ball_meeting_point(12.0, 8.0, 1.0, 0.0), (12.0, 8.0, 0.0))


class PossessionStyleTests(unittest.TestCase):
    def test_carrier_dribbles_when_no_receiver_is_collectable(self):
        # Push every teammate out of collectable range: a pass could never be
        # touched, so the carrier must keep the ball.
        observation = obs((30, 20), "us", our_st_x=30, them_x=45)
        for player in observation["us"]:
            if player["id"] == "st":
                continue
            player["position"] = {"x": 6.0, "y": 3.0}
        inp = make_inp(observation)
        intents = PolicyController().decide(inp)
        self.assertEqual(intents["st"].action_type, "none")
        self.assertIsNone(intents["st"].action_target)

    def test_carried_dribble_target_is_reachable_not_a_dash_to_the_goal_line(self):
        inp = make_inp(obs((28, 20), "us", our_st_x=28, them_x=45))
        controller = PolicyController()
        carrier = next(p for p in inp.state.outfield_us() if p.id == "st")
        tx, ty, _ = controller._dribble_target(inp, carrier)
        self.assertLessEqual(geom.distance(carrier.x, carrier.y, tx, ty), 16.0)
        self.assertGreater(tx, carrier.x)

    def test_loose_ball_chaser_aims_at_the_meeting_point(self):
        observation = obs((30, 20), None)
        observation["ball"]["velocity"] = {"x": 12.0, "y": 0.0}
        inp = make_inp(observation)
        intents = PolicyController().decide(inp)
        mx, my, _ = loose_ball_meeting_point(30.0, 20.0, 12.0, 0.0)
        chasers = [it for it in intents.values() if it.tx > 30.5 and it.speed >= 0.9]
        self.assertTrue(chasers, "nobody chased the loose ball")
        for it in chasers:
            self.assertAlmostEqual(it.tx, mx, delta=1.0)
            self.assertAlmostEqual(it.ty, my, delta=1.0)

    def test_kickoff_carries_the_ball_instead_of_passing(self):
        observation = obs((30, 20), "us", our_st_x=30, them_x=45)
        inp = make_inp(observation, tactical_state=TacticalState.KICKOFF)
        intents = PolicyController().decide(inp)
        self.assertEqual(intents["st"].action_type, "none")
        self.assertGreater(intents["st"].tx, 30.0)

    def test_every_intent_target_is_inside_the_pitch(self):
        for possess in ("us", "them", None):
            inp = make_inp(obs((38, 20), possess, them_x=30))
            for it in PolicyController().decide(inp).values():
                self.assertGreaterEqual(it.tx, -0.5)
                self.assertLessEqual(it.tx, PITCH_LENGTH + 0.5)
                self.assertGreaterEqual(it.ty, -0.5)
                self.assertLessEqual(it.ty, PITCH_WIDTH + 0.5)

class AttackCandidatePipelineTests(unittest.TestCase):
    def _intent(
        self,
        *,
        action_type="none",
        action_target=None,
    ):
        return PlayerIntent(
            pid="st",
            tx=30.0,
            ty=20.0,
            speed=0.4,
            face_x=60.0,
            face_y=20.0,
            action_type=action_type,
            action_target=action_target,
        )

    def test_filter_removes_only_physically_short_passes(self):
        controller = PolicyController()

        inp = make_inp(obs((30, 20), "us", our_st_x=30, them_x=45))
        carrier = next(
            p for p in inp.state.outfield_us()
            if p.id == "st"
        )

        short_pass = ActionCandidate(
            10.0,
            self._intent(
                action_type="pass",
                action_target=(carrier.x + MIN_PASS_TRAVEL - 1.0, carrier.y),
            ),
            "short_pass",
        )

        legal_pass = ActionCandidate(
            9.0,
            self._intent(
                action_type="pass",
                action_target=(carrier.x + MIN_PASS_TRAVEL + 1.0, carrier.y),
            ),
            "legal_pass",
        )

        shot = ActionCandidate(
            8.0,
            self._intent(
                action_type="shoot",
                action_target=(OPP_GOAL_X, GOAL_CENTER_Y),
            ),
            "shot",
        )

        carry = ActionCandidate(
            7.0,
            self._intent(),
            "carry",
        )

        result = controller._filter_attack_candidates(
            carrier,
            [short_pass, legal_pass, shot, carry],
        )

        self.assertEqual(
            [candidate.reason for candidate in result],
            ["legal_pass", "shot", "carry"],
        )

    def test_rank_orders_by_descending_ev_without_mutating_input(self):
        controller = PolicyController()

        low = ActionCandidate(2.0, self._intent(), "low")
        high = ActionCandidate(9.0, self._intent(), "high")
        middle = ActionCandidate(5.0, self._intent(), "middle")

        candidates = [low, high, middle]
        original = list(candidates)

        ranked = controller._rank_attack_candidates(candidates)

        self.assertEqual(
            [candidate.value for candidate in ranked],
            [9.0, 5.0, 2.0],
        )

        self.assertEqual(candidates, original)
        self.assertIsNot(ranked, candidates)

    def test_rank_preserves_generation_order_for_equal_ev(self):
        controller = PolicyController()

        first = ActionCandidate(5.0, self._intent(), "first")
        second = ActionCandidate(5.0, self._intent(), "second")
        lower = ActionCandidate(1.0, self._intent(), "lower")

        ranked = controller._rank_attack_candidates(
            [first, second, lower]
        )

        self.assertEqual(
            [candidate.reason for candidate in ranked],
            ["first", "second", "lower"],
        )

    def test_candidate_generation_is_deterministic_and_structurally_valid(self):
        controller = PolicyController()
        inp = make_inp(obs((52, 20), "us", our_st_x=52, them_x=10))
        carrier = inp.state.our_possessor()
        assert carrier is not None

        first = controller._build_attack_candidates(inp, carrier)
        second = controller._build_attack_candidates(inp, carrier)

        self.assertTrue(first)
        self.assertEqual(first, second)
        for candidate in first:
            self.assertIsInstance(candidate, ActionCandidate)
            self.assertTrue(math.isfinite(candidate.value))
            self.assertIsInstance(candidate.intent, PlayerIntent)
            self.assertEqual(candidate.intent.pid, carrier.id)
            self.assertIn(candidate.intent.action_type, {"none", "pass", "shoot"})
            self.assertTrue(candidate.reason.strip())
            for value in (
                candidate.intent.tx,
                candidate.intent.ty,
                candidate.intent.speed,
                candidate.intent.face_x,
                candidate.intent.face_y,
            ):
                self.assertTrue(math.isfinite(value))
            if candidate.intent.action_target is not None:
                self.assertTrue(all(math.isfinite(value) for value in candidate.intent.action_target))
            if candidate.intent.action_power is not None:
                self.assertTrue(math.isfinite(candidate.intent.action_power))

    def test_generation_does_not_commit_shots_or_pending_passes(self):
        controller = PolicyController()
        shot_input = make_inp(obs((52, 20), "us", our_st_x=52, them_x=10))
        shot_carrier = shot_input.state.our_possessor()
        assert shot_carrier is not None
        plan = TeamPlan(
            shot_expected=True,
            rebound_zone_x=47.0,
            rebound_zone_y=18.0,
            far_post_target_y=22.0,
        )
        plan_before = copy.deepcopy(plan)
        state_before = copy.deepcopy(shot_input.state)
        world_before = copy.deepcopy(shot_input.world)
        log_before = copy.deepcopy(controller.log)

        shot_board = controller._build_attack_candidates(shot_input, shot_carrier, plan)

        self.assertTrue(any(candidate.intent.action_type == "shoot" for candidate in shot_board))
        self.assertEqual(plan, plan_before)
        self.assertFalse(plan.shot_committed)
        self.assertEqual(shot_input.state, state_before)
        self.assertEqual(shot_input.world, world_before)
        self.assertIsNone(controller._pending_pass_receiver)
        self.assertIsNone(controller._pending_pass_collection)
        self.assertEqual(controller.log, log_before)

        selected = controller._evaluate_attack_actions(shot_input, shot_carrier, plan)
        self.assertEqual(selected.action_type, "shoot")
        assert selected.action_target is not None
        self.assertTrue(plan.shot_committed)
        self.assertFalse(plan.shot_expected)
        self.assertEqual(plan.rebound_zone_x, selected.action_target[0])
        self.assertEqual(plan.rebound_zone_y, selected.action_target[1])

        pass_input = make_inp(obs((30, 20), "us", our_st_x=30, them_x=10))
        pass_carrier = pass_input.state.our_possessor()
        assert pass_carrier is not None
        pass_board = controller._build_attack_candidates(pass_input, pass_carrier)
        self.assertTrue(
            any(candidate.intent.action_type == "pass" for candidate in pass_board),
            "fixture must include an available pass to verify it is not scheduled",
        )
        self.assertIsNone(controller._pending_pass_receiver)
        self.assertIsNone(controller._pending_pass_collection)

    def test_raw_board_flows_through_filter_and_rank_without_mutation(self):
        controller = PolicyController()
        inp = make_inp(obs((30, 20), "us", our_st_x=30, them_x=10))
        carrier = inp.state.our_possessor()
        assert carrier is not None

        candidates = controller._build_attack_candidates(inp, carrier)
        original = list(candidates)
        filtered = controller._filter_attack_candidates(carrier, candidates)
        filtered_original = list(filtered)
        ranked = controller._rank_attack_candidates(filtered)

        self.assertTrue(ranked)
        self.assertEqual(candidates, original)
        self.assertEqual(filtered, filtered_original)
        self.assertIsNot(ranked, filtered)
        self.assertEqual(
            [candidate.value for candidate in ranked],
            sorted((candidate.value for candidate in filtered), reverse=True),
        )
        self.assertEqual(
            ranked,
            controller._rank_attack_candidates(
                controller._filter_attack_candidates(
                    carrier, controller._build_attack_candidates(inp, carrier)
                )
            ),
        )

    def test_own_goal_veto_prevents_commitment_to_replaced_pass(self):
        controller = PolicyController()
        inp = make_inp(obs((25, 20), "us", our_st_x=25, them_x=10))
        carrier = inp.state.our_possessor()
        assert carrier is not None
        plan = TeamPlan()
        invalid_pass = ActionCandidate(
            1.0,
            PlayerIntent(
                pid=carrier.id,
                tx=5.0,
                ty=20.0,
                speed=0.4,
                face_x=5.0,
                face_y=20.0,
                action_type="pass",
                action_target=(5.0, 20.0),
                action_power=0.5,
                receiver_id="w",
                collection_point=(7.0, 20.0),
            ),
            "invalid_own_goal_pass",
        )

        selected_intent = controller._commit_attack_candidate(
            inp, carrier, invalid_pass, plan
        )

        self.assertEqual(selected_intent.action_type, "none")
        self.assertIsNone(controller._pending_pass_receiver)
        self.assertIsNone(controller._pending_pass_collection)
        self.assertFalse(plan.shot_committed)


class ActiveMCTSIntegrationTests(unittest.TestCase):
    class FixedBoardController(PolicyController):
        def __init__(self, board):
            super().__init__()
            self.board = board
            self.commits = 0

        def _build_attack_candidates(self, inp, p, team_plan=None):
            return self.board

        def _commit_attack_candidate(self, inp, p, candidate, team_plan=None):
            self.commits += 1
            return super()._commit_attack_candidate(inp, p, candidate, team_plan)

    def _input_and_board(self, descriptions):
        inp = make_inp(obs((30, 20), "us", our_st_x=30, them_x=10))
        inp.config.enable_mcts = True
        inp.config.mcts_iterations = 4
        possessor = inp.state.our_possessor()
        assert possessor is not None
        board = []
        for action_type, value in descriptions:
            if action_type == "pass":
                intent = PlayerIntent(
                    pid=possessor.id,
                    tx=possessor.x,
                    ty=possessor.y,
                    speed=0.4,
                    face_x=possessor.x + 20.0,
                    face_y=possessor.y,
                    action_type="pass",
                    action_target=(possessor.x + 20.0, possessor.y),
                    action_power=0.2,
                    receiver_id="am",
                    collection_point=(possessor.x + 18.0, possessor.y),
                )
            elif action_type == "shoot":
                intent = PlayerIntent(
                    pid=possessor.id,
                    tx=possessor.x,
                    ty=possessor.y,
                    speed=0.4,
                    face_x=60.0,
                    face_y=17.0,
                    action_type="shoot",
                    action_target=(60.0, 17.0),
                    action_power=0.8,
                )
            else:
                intent = PlayerIntent(
                    pid=possessor.id,
                    tx=possessor.x + 5.0,
                    ty=possessor.y + 1.0,
                    speed=0.7,
                    face_x=possessor.x + 5.0,
                    face_y=possessor.y + 1.0,
                )
            board.append(ActionCandidate(value, intent, f"opaque_{action_type}_reason"))
        return inp, board

    def _decide_with_choice(self, descriptions, selected_index, *, root_means=None, visits=None):
        inp, board = self._input_and_board(descriptions)
        controller = self.FixedBoardController(board)

        def select_from_exact_board(
            planner, search_inp, *, controller, team_plan, root_candidates
        ):
            from src.mcts import RootActionStats, SearchDiagnostics

            self.assertIs(search_inp, inp)
            self.assertIs(controller, controller_arg)
            self.assertEqual(len(root_candidates), len(board))
            self.assertTrue(all(actual is expected for actual, expected in zip(root_candidates, board)))
            means = list(root_means or [0.0] * len(root_candidates))
            if root_means is None and selected_index != 0:
                means[selected_index] = 0.25
            visit_counts = list(visits or [4] * len(root_candidates))
            planner.stats = SearchDiagnostics(
                iterations=sum(visit_counts),
                root_actions=tuple(
                    RootActionStats(candidate, count, means[index] * count)
                    for index, (candidate, count) in enumerate(
                        zip(root_candidates, visit_counts)
                    )
                ),
            )
            return root_candidates[selected_index]

        controller_arg = controller
        with patch(
            "src.mcts.MCTSPlanner.search",
            autospec=True,
            side_effect=select_from_exact_board,
        ) as search:
            intents = controller.decide(inp)
        return inp, board, controller, intents, search

    def test_default_off_uses_top_ranked_candidate_without_search(self):
        inp, board = self._input_and_board((("pass", 20.0), ("carry", 10.0)))
        inp.config.enable_mcts = False
        controller = self.FixedBoardController(board)

        with patch("src.mcts.MCTSPlanner.search", autospec=True) as search:
            intents = controller.decide(inp)

        self.assertEqual(intents["st"], board[0].intent)
        self.assertEqual(controller.commits, 1)
        search.assert_not_called()
        self.assertEqual(controller.last_mcts_diagnostics["skip_reason"], "mcts_disabled")

    def test_active_selector_uses_exact_root_objects_and_can_choose_nonfirst(self):
        inp, board, controller, intents, search = self._decide_with_choice(
            (("pass", 20.0), ("carry", 10.0)), 1
        )

        self.assertIs(intents["st"], board[1].intent)
        self.assertEqual(controller.commits, 1)
        search.assert_called_once()
        diagnostics = controller.last_mcts_diagnostics
        self.assertTrue(diagnostics["mcts_ran"])
        self.assertEqual(diagnostics["production_candidate_index"], 0)
        self.assertEqual(diagnostics["mcts_candidate_index"], 1)
        self.assertFalse(diagnostics["agreement"])
        self.assertTrue(diagnostics["override_accepted"])
        self.assertFalse(diagnostics["override_rejected"])
        self.assertEqual(diagnostics["selected_candidate_index"], 1)
        possessor = inp.state.our_possessor()
        assert possessor is not None
        self.assertEqual(possessor.id, intents["st"].pid)

    def test_active_pass_commits_receiver_metadata_once(self):
        _, board, controller, intents, _ = self._decide_with_choice(
            (("carry", 20.0), ("pass", 10.0)), 1
        )

        self.assertEqual(intents["st"], board[1].intent)
        self.assertEqual((intents["am"].tx, intents["am"].ty), (48.0, 20.0))
        self.assertIsNone(controller._pending_pass_receiver)
        self.assertIsNone(controller._pending_pass_collection)
        self.assertEqual(controller.commits, 1)

    def test_weak_search_disagreement_keeps_production_winner(self):
        _, board, controller, intents, _ = self._decide_with_choice(
            (("pass", 20.0), ("carry", 10.0)),
            1,
            root_means=(0.0, 0.08),
        )

        self.assertIs(intents["st"], board[0].intent)
        diagnostics = controller.last_mcts_diagnostics
        self.assertEqual(diagnostics["mcts_candidate_index"], 1)
        self.assertEqual(diagnostics["selected_candidate_index"], 0)
        self.assertTrue(diagnostics["proposed_disagreement"])
        self.assertFalse(diagnostics["override_accepted"])
        self.assertTrue(diagnostics["override_rejected"])
        advantage = diagnostics["search_advantage"]
        assert isinstance(advantage, float)
        self.assertAlmostEqual(advantage, 0.08)
        self.assertEqual(diagnostics["required_override_margin"], 0.10)
        self.assertEqual(diagnostics["override_reason"], "below_required_search_advantage")

    def test_strong_search_disagreement_overrides_production_winner(self):
        _, board, controller, intents, _ = self._decide_with_choice(
            (("pass", 20.0), ("carry", 10.0)),
            1,
            root_means=(0.0, 0.15),
        )

        self.assertIs(intents["st"], board[1].intent)
        self.assertTrue(controller.last_mcts_diagnostics["override_accepted"])
        advantage = controller.last_mcts_diagnostics["search_advantage"]
        assert isinstance(advantage, float)
        self.assertAlmostEqual(advantage, 0.15)

    def test_weak_shoot_to_carry_proposal_uses_larger_margin(self):
        _, board, controller, intents, _ = self._decide_with_choice(
            (("shoot", 20.0), ("carry", 10.0)),
            1,
            root_means=(0.0, 0.15),
        )

        diagnostics = controller.last_mcts_diagnostics
        self.assertIs(intents["st"], board[0].intent)
        self.assertEqual(diagnostics["mcts_action_type"], "none")
        self.assertEqual(diagnostics["selected_action_type"], "shoot")
        self.assertEqual(diagnostics["required_override_margin"], 0.20)
        self.assertTrue(diagnostics["override_rejected"])

    def test_strong_search_can_still_override_production_shot(self):
        _, board, controller, intents, _ = self._decide_with_choice(
            (("shoot", 20.0), ("carry", 10.0)),
            1,
            root_means=(0.0, 0.25),
        )

        diagnostics = controller.last_mcts_diagnostics
        self.assertIs(intents["st"], board[1].intent)
        self.assertEqual(diagnostics["required_override_margin"], 0.20)
        self.assertTrue(diagnostics["override_accepted"])

    def test_agreement_does_not_trigger_override_gate(self):
        _, board, controller, intents, _ = self._decide_with_choice(
            (("pass", 20.0), ("carry", 10.0)), 0
        )

        diagnostics = controller.last_mcts_diagnostics
        self.assertIs(intents["st"], board[0].intent)
        self.assertTrue(diagnostics["agreement"])
        self.assertFalse(diagnostics["proposed_disagreement"])
        self.assertFalse(diagnostics["override_accepted"])
        self.assertFalse(diagnostics["override_rejected"])
        self.assertEqual(diagnostics["override_reason"], "mcts_agreed")

    def test_alternative_without_root_statistics_is_rejected(self):
        inp, board = self._input_and_board((("pass", 20.0), ("carry", 10.0)))
        controller = self.FixedBoardController(board)

        def select_without_stats(planner, search_inp, **kwargs):
            return kwargs["root_candidates"][1]

        with patch(
            "src.mcts.MCTSPlanner.search",
            autospec=True,
            side_effect=select_without_stats,
        ):
            intents = controller.decide(inp)

        diagnostics = controller.last_mcts_diagnostics
        self.assertIs(intents["st"], board[0].intent)
        self.assertTrue(diagnostics["proposed_disagreement"])
        self.assertFalse(diagnostics["override_accepted"])
        self.assertTrue(diagnostics["override_rejected"])
        self.assertIsNone(diagnostics["search_advantage"])
        self.assertEqual(diagnostics["override_reason"], "insufficient_root_statistics")

        _, board, controller, intents, _ = self._decide_with_choice(
            (("pass", 20.0), ("carry", 10.0)),
            1,
            root_means=(0.0, 0.25),
            visits=(0, 4),
        )
        self.assertIs(intents["st"], board[0].intent)
        self.assertTrue(controller.last_mcts_diagnostics["override_rejected"])
        self.assertEqual(
            controller.last_mcts_diagnostics["override_reason"],
            "insufficient_root_statistics",
        )

    def test_active_shot_commits_team_plan_once(self):
        _, board, controller, intents, _ = self._decide_with_choice(
            (("carry", 20.0), ("shoot", 10.0)), 1
        )

        manager = controller._team_plan_manager
        assert manager is not None
        plan = manager.current_plan
        assert plan is not None
        self.assertEqual(intents["st"], board[1].intent)
        self.assertTrue(plan.shot_committed)
        self.assertEqual((plan.rebound_zone_x, plan.rebound_zone_y), (60.0, 17.0))
        self.assertEqual(controller.commits, 1)

    def test_active_carry_preserves_production_intent_semantics(self):
        _, board, controller, intents, _ = self._decide_with_choice(
            (("pass", 20.0), ("carry", 10.0)), 1
        )

        actual = intents["st"]
        expected = board[1].intent
        self.assertEqual(
            (actual.pid, actual.tx, actual.ty, actual.speed, actual.face_x, actual.face_y,
             actual.action_type, actual.action_target, actual.action_power),
            (expected.pid, expected.tx, expected.ty, expected.speed, expected.face_x,
             expected.face_y, expected.action_type, expected.action_target, expected.action_power),
        )
        self.assertEqual(controller.commits, 1)

    def test_invalid_or_missing_search_result_falls_back_to_ranked_first(self):
        for unavailable in (True, False):
            with self.subTest(unavailable=unavailable):
                inp, board = self._input_and_board((("pass", 20.0), ("carry", 10.0)))
                controller = self.FixedBoardController(board)
                invalid = None if unavailable else ActionCandidate(
                    board[1].value, board[1].intent, board[1].reason
                )
                with patch(
                    "src.mcts.MCTSPlanner.search",
                    autospec=True,
                    return_value=invalid,
                ):
                    intents = controller.decide(inp)

                self.assertIs(intents["st"], board[0].intent)
                self.assertEqual(controller.commits, 1)
                self.assertEqual(
                    controller.last_mcts_diagnostics["fallback_reason"],
                    "search_returned_no_candidate" if unavailable else "candidate_not_in_root_board",
                )

    def test_unsupported_search_state_falls_back_to_ranked_first(self):
        inp, board = self._input_and_board((("pass", 20.0), ("carry", 10.0)))
        inp.state = replace(
            inp.state,
            ball=replace(inp.state.ball, x=-1.0),
        )
        inp.world = WorldModel.build(inp.state)
        controller = self.FixedBoardController(board)

        intents = controller.decide(inp)

        self.assertIs(intents["st"], board[0].intent)
        self.assertEqual(controller.commits, 1)
        self.assertEqual(
            controller.last_mcts_diagnostics["fallback_reason"],
            "search_returned_no_candidate",
        )

    def test_single_candidate_skips_search(self):
        inp, board = self._input_and_board((("carry", 10.0),))
        controller = self.FixedBoardController(board)
        with patch("src.mcts.MCTSPlanner.search", autospec=True) as search:
            intents = controller.decide(inp)

        self.assertIs(intents["st"], board[0].intent)
        search.assert_not_called()
        self.assertEqual(controller.last_mcts_diagnostics["skip_reason"], "no_meaningful_choice")

    def test_goalkeeper_possession_and_zero_budget_skip_search(self):
        inp = make_inp(obs((30, 20), "us", our_st_x=30, them_x=10))
        inp.config.enable_mcts = True
        inp.config.mcts_iterations = 4
        inp.state = replace(
            inp.state,
            ball=replace(inp.state.ball, possessing_player="gk"),
        )
        inp.world = WorldModel.build(inp.state)
        goalkeeper = inp.state.our_possessor()
        assert goalkeeper is not None
        board = [
            ActionCandidate(
                2.0,
                PlayerIntent(goalkeeper.id, goalkeeper.x, goalkeeper.y, 0.0,
                             30.0, 20.0, "none"),
                "opaque_a",
            ),
            ActionCandidate(
                1.0,
                PlayerIntent(goalkeeper.id, goalkeeper.x, goalkeeper.y, 0.0,
                             31.0, 20.0, "none"),
                "opaque_b",
            ),
        ]
        controller = self.FixedBoardController(board)
        with patch("src.mcts.MCTSPlanner.search", autospec=True) as search:
            controller.decide(inp)

        search.assert_not_called()
        self.assertEqual(
            controller.last_mcts_diagnostics["skip_reason"],
            "no_valid_outfield_possessor",
        )

        budget_input, budget_board = self._input_and_board((("pass", 20.0), ("carry", 10.0)))
        budget_input.config.mcts_iterations = 0
        budget_controller = self.FixedBoardController(budget_board)
        with patch("src.mcts.MCTSPlanner.search", autospec=True) as search:
            budget_controller.decide(budget_input)
        search.assert_not_called()
        self.assertEqual(
            budget_controller.last_mcts_diagnostics["skip_reason"],
            "non_positive_budget",
        )

    def test_real_active_selection_and_diagnostics_are_deterministic(self):
        decisions = []
        for _ in range(2):
            inp = make_inp(obs((30, 20), "us", our_st_x=30, them_x=10))
            inp.config.enable_mcts = True
            inp.config.mcts_iterations = 4
            controller = PolicyController()
            decisions.append((controller.decide(inp)["st"], controller.last_mcts_diagnostics))

        self.assertEqual(decisions[0], decisions[1])
        self.assertTrue(decisions[0][1]["mcts_ran"])

    def test_real_weak_shoot_downgrade_is_rejected_on_live_root_board(self):
        inp = make_inp(obs((58.349748, 1.705684), "us", our_st_x=58, them_x=10))
        us = {
            "gk": (2.5, 13.032385, -0.197539),
            "cd": (24.524329, 9.753869, -0.231505),
            "w": (18.685203, 9.722849, -0.198954),
            "am": (13.327374, 15.845790, -0.319273),
            "st": (58.121901, 2.039270, -0.948023),
        }
        them = {
            "tgk": (58.5, 20.0, 0.0),
            "cd": (45.278273, 14.721727, 0.0),
            "am": (36.0, 28.0, 0.0),
            "st": (41.714027, 18.285973, 0.0),
        }
        inp.state = replace(
            inp.state,
            sequence=30,
            simulation_tick=30,
            apply_at_tick=30,
            time_remaining=293.9,
            ball=replace(
                inp.state.ball,
                x=58.349748,
                y=1.705684,
                vx=-2.373563,
                vy=2.373563,
                possessing_team="us",
                possessing_player="st",
            ),
            us=tuple(
                replace(player, x=us[player.id][0], y=us[player.id][1], facing=us[player.id][2])
                for player in inp.state.us
            ),
            them=tuple(
                replace(player, x=them[player.id][0], y=them[player.id][1], facing=them[player.id][2])
                for player in inp.state.them
            ),
        )
        inp.world = WorldModel.build(inp.state)
        inp.time_remaining = 293.9
        inp.config.enable_mcts = True
        inp.config.mcts_iterations = 24
        class CountingController(PolicyController):
            def __init__(self):
                super().__init__()
                self.commit_count = 0

            def _commit_attack_candidate(self, inp, p, candidate, team_plan=None):
                self.commit_count += 1
                return super()._commit_attack_candidate(inp, p, candidate, team_plan)

        controller = CountingController()
        from src.mcts import MCTSPlanner

        original_search = MCTSPlanner.search
        captured = {}

        def capture_root(planner, search_input, *, controller, team_plan, root_candidates):
            selected = original_search(
                planner,
                search_input,
                controller=controller,
                team_plan=team_plan,
                root_candidates=root_candidates,
            )
            captured["board"] = tuple(root_candidates)
            captured["selected"] = selected
            return selected

        with patch(
            "src.mcts.MCTSPlanner.search", autospec=True, side_effect=capture_root
        ):
            intents = controller.decide(inp)

        board = captured["board"]
        selected = captured["selected"]
        diagnostics = controller.last_mcts_diagnostics
        self.assertEqual(diagnostics["production_action_type"], "shoot")
        self.assertEqual(diagnostics["mcts_candidate_index"], 1)
        self.assertEqual(diagnostics["selected_candidate_index"], 0)
        self.assertFalse(diagnostics["agreement"])
        self.assertTrue(diagnostics["override_rejected"])
        self.assertEqual(diagnostics["required_override_margin"], 0.20)
        self.assertIs(selected, board[1])
        self.assertIs(intents["st"], board[0].intent)
        self.assertEqual(controller.commit_count, 1)

    def test_real_search_has_no_live_side_effects_before_commit(self):
        inp = make_inp(obs((30, 20), "us", our_st_x=30, them_x=10))
        inp.config.enable_mcts = True
        inp.config.mcts_iterations = 3
        controller = PolicyController()
        controller._pending_pass_receiver = "previous_receiver"
        controller._pending_pass_collection = (11.0, 12.0)
        original_search = __import__("src.mcts", fromlist=["MCTSPlanner"]).MCTSPlanner.search
        observed = []

        def isolated_search(planner, search_inp, *, controller, team_plan, root_candidates):
            state_before = copy.deepcopy(search_inp.state)
            world_before = copy.deepcopy(search_inp.world)
            opp_before = copy.deepcopy(search_inp.opp)
            log_before = copy.deepcopy(controller.log)
            plan_before = copy.deepcopy(team_plan)
            pending_before = (
                controller._pending_pass_receiver,
                controller._pending_pass_collection,
            )
            candidates_before = copy.deepcopy(root_candidates)
            selected = original_search(
                planner,
                search_inp,
                controller=controller,
                team_plan=team_plan,
                root_candidates=root_candidates,
            )
            self.assertEqual(search_inp.state, state_before)
            self.assertEqual(search_inp.world, world_before)
            self.assertEqual(search_inp.opp, opp_before)
            self.assertEqual(controller.log, log_before)
            self.assertEqual(team_plan, plan_before)
            self.assertEqual(
                (controller._pending_pass_receiver, controller._pending_pass_collection),
                pending_before,
            )
            self.assertEqual(root_candidates, candidates_before)
            observed.append(tuple(root_candidates))
            return selected

        with patch("src.mcts.MCTSPlanner.search", autospec=True, side_effect=isolated_search):
            controller.decide(inp)

        self.assertTrue(observed)
        self.assertTrue(all(len(board) > 1 for board in observed))

if __name__ == "__main__":
    unittest.main()
