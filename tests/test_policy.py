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
)
from src.opponent import OpponentModel
from src.state import GameState, WorldModel
from src import geom
from src.geom import PITCH_LENGTH, PITCH_WIDTH

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
        self.assertLess(
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
