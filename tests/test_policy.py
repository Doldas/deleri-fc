import sys
import unittest
from pathlib import Path

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
)
from src.opponent import OpponentModel
from src.state import GameState, WorldModel
from src import geom
from src.geom import PITCH_LENGTH, PITCH_WIDTH, GOAL_CENTER_Y, OPP_GOAL_X
from src.physics import shot_beats_keeper

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
        self.assertEqual(match_context(300.0, 1, 0), "leading")
        self.assertEqual(match_context(300.0, 0, 1), "trailing")
        self.assertEqual(match_context(300.0, 0, 0), "tied")
        self.assertEqual(match_context(90.0, 1, 0), "leading_late")
        self.assertEqual(match_context(90.0, 0, 1), "trailing_late")
        self.assertEqual(match_context(90.0, 0, 0), "tied_late")

    def test_trailing_late_raises_risk_and_verticality(self):
        leading = make_inp(obs((35, 20), "us", them_x=20))
        leading.score_us = 2
        leading.score_them = 0
        leading.time_remaining = 60.0
        trailing = make_inp(obs((35, 20), "us", them_x=20))
        trailing.score_us = 0
        trailing.score_them = 2
        trailing.time_remaining = 60.0
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


if __name__ == "__main__":
    unittest.main()
