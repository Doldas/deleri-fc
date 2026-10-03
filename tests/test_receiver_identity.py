"""P1.2 -- the player a pass is generated for must be the player who is ordered
to the collection point.

The pass generators (``_through_ball_choice``, ``_cross_choice``, ... ) already
compute the intended ``receiver_id`` and hand it to ``p_pass_complete`` to price
the pass. That identity was then thrown away at selection time and rediscovered
with "first outfielder within 10 m of the target", which sends the collection
movement to whichever teammate happens to come first in the player list.

These tests pin the end-to-end identity for every pass type, plus the ambiguity
case where a decoy teammate is closer to the collection point than the real
receiver.
"""

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.policy import PolicyController
from tests.test_policy import make_inp, obs

# Positions found by searching for scenarios in which each generator wins its
# candidate board. Stored as coordinates rather than raw observations so the
# fixtures stay readable. Full float precision is required: several of these
# generators have knife-edge gates (e.g. `_through_ball_choice`'s `miss < 7.0`)
# and rounding to 2 dp flips which candidate wins the board.
PASS_SCENARIOS = {
    "through_ball": dict(
        ball=(34.198341896447374, 33.16156890780306), carrier="st",
        us=[("gk", 48.95638540002879, 16.79164649940514), ("cd", 52.068712655727616, 32.9146113568672), ("am", 44.40235355537375, 29.087173228531235), ("w", 47.0787072099662, 16.795347266138926), ("st", 45.46010545658366, 5.398756259004191)],
        them=[("tgk", 58, 20), ("cd", 30.985309120000878, 18.9405733006399), ("am", 18.402595011486337, 15.091664296982355), ("st", 42.27105894910377, 24.216689479987124)],
    ),
    "cross": dict(
        ball=(51.3022973492486, 29.709168098626286), carrier="st",
        us=[("gk", 26.006988414508697, 16.405337660449845), ("cd", 50.466322264646756, 24.822412324075657), ("am", 21.812644582987552, 36.63625771534586), ("w", 26.103248005945858, 11.781436967295095), ("st", 47.362208211367275, 14.184484468622818)],
        them=[("tgk", 58, 20), ("cd", 29.260340978759963, 5.495550815206421), ("am", 21.42445257255335, 22.81298313769677), ("st", 27.23449096522737, 23.44365068178592)],
    ),
    "cutback": dict(
        ball=(38.09018245533192, 5.090770212212024), carrier="st",
        us=[("gk", 50.39627388524801, 3.1813120729389586), ("cd", 34.445932693792166, 18.667887855628624), ("am", 53.19120470922643, 28.20148671875577), ("w", 38.76907423374422, 19.2941458293999), ("st", 47.9948835082502, 30.085847334850737)],
        them=[("tgk", 58, 20), ("cd", 50.364648985079775, 29.071110102493574), ("am", 54.75931581825047, 6.37336463931188), ("st", 50.219797369847655, 12.448158570070259)],
    ),
    "high_press_onetwo": dict(
        ball=(28.46574703110936, 4.868330321786736), carrier="st",
        us=[("gk", 41.92980784391967, 7.740611445016581), ("cd", 30.7723350467164, 18.26882694798989), ("am", 21.795166438409193, 36.664411901752636), ("w", 44.70630747705984, 23.252328988691286), ("st", 26.828145861411617, 12.043358352558878)],
        them=[("tgk", 58, 20), ("cd", 27.75908762389348, 3.8559903071647756), ("am", 20.683512058509027, 13.346519133033446), ("st", 19.99666488908533, 18.382058319446184)],
    ),
    "wall_pass": dict(
        ball=(37.23145648864116, 11.360847324241671), carrier="st",
        us=[("gk", 37.578814722284115, 4.095063626462468), ("cd", 41.8386088398359, 18.19170674429833), ("am", 40.54787817414163, 23.215435861471594), ("w", 53.40117454782096, 8.165592165716223), ("st", 35.80998115591157, 36.75763944315446)],
        them=[("tgk", 58, 20), ("cd", 46.043720943247436, 13.640788200512652), ("am", 45.996514825670246, 36.64020301966068), ("st", 33.279675319743816, 32.49411437598374)],
    ),
    "safe_pass_to_w": dict(
        ball=(47.94405012171376, 29.639630832866413), carrier="st",
        us=[("gk", 43.43568166415652, 10.855394394457425), ("cd", 25.01655659828264, 3.466324016767625), ("am", 40.25001243029236, 34.99415825967406), ("w", 54.87642752562226, 7.547436200922307), ("st", 44.128342898949896, 17.256323341781354)],
        them=[("tgk", 58, 20), ("cd", 41.99563972129983, 16.48839572926263), ("am", 53.46835356811771, 36.619442349060684), ("st", 19.850640997123644, 27.251008286399323)],
    ),
    "safe_pass_to_am": dict(
        ball=(48.149661352942225, 22.717024313062083), carrier="st",
        us=[("gk", 19.826277739213154, 9.791858746142594), ("cd", 34.397222382155995, 31.703499104671437), ("am", 53.85195832818917, 23.098536910562746), ("w", 27.859822705610632, 4.582469473733603), ("st", 39.23495263680243, 16.909706091326676)],
        them=[("tgk", 58, 20), ("cd", 53.36878402130752, 10.023431524336136), ("am", 25.78275489849826, 3.539500981644666), ("st", 19.08985717240712, 30.452744306266794)],
    ),
    "switch_play": dict(
        ball=(39.42001331560032, 27.10143098764504), carrier="st",
        us=[("gk", 20.646619400791995, 21.55813628319957), ("cd", 46.00620880491783, 33.606659962478886), ("am", 45.62939491895524, 26.580962179539064), ("w", 48.14413369155239, 34.110087704769356), ("st", 31.36969567259564, 26.294962481345735)],
        them=[("tgk", 58, 20), ("cd", 52.62486184807357, 32.71197487365064), ("am", 33.85181939513307, 29.878088020005585), ("st", 50.81196161357444, 22.47545494507332)],
    ),}

# Required pass types from P1.2; "normal progressive pass" is covered by the
# safe pass / switch play boards, which are the generic non-special passes.
REQUIRED_TYPES = (
    "through_ball",
    "cross",
    "cutback",
    "high_press_onetwo",
    "wall_pass",
    "safe_pass_to_w",
    "safe_pass_to_am",
    "switch_play",
)


def build(spec):
    """Build an observation from a compact position spec."""
    observation = obs(spec["ball"], "us")
    observation["ball"]["possessedBy"] = spec["carrier"]
    for team, key in (("us", "us"), ("them", "them")):
        for pl in observation[team]:
            for pid, x, y in spec[key]:
                if pl["id"] == pid:
                    pl["position"]["x"] = x
                    pl["position"]["y"] = y
    return observation


def selected_pass(controller, intents):
    """The carrier's pass intent, plus the reason it won the board."""
    for entry in controller.log.possessions_decided:
        if entry["action"] == "pass":
            return intents.get(entry["player"]), entry["reason"]
    return None, None


def legacy_receiver(inp, carrier_id, target):
    """The removed heuristic: first outfielder within 10 m of the target."""
    for t in inp.state.outfield_us():
        if t.id != carrier_id and t.can_act:
            if math.dist((t.x, t.y), target) < 10.0:
                return t.id
    return None


class ReceiverIdentityTests(unittest.TestCase):
    def _assert_identity(self, reason_key):
        spec = PASS_SCENARIOS[reason_key]
        controller = PolicyController()
        observation = build(spec)
        if reason_key == "cutback":
            # Block the keeper-pull dribble lane and the direct striker lane so
            # this fixture still isolates the cutback receiver identity after
            # the final-third keeper-displacement carry was added.
            for index, (x, y) in enumerate(((50.2, 29.0), (50.2, 34.0))):
                observation["them"].append({
                    "id": f"cutback_blocker_{index}", "role": "outfield",
                    "position": {"x": x, "y": y},
                    "velocity": {"x": 0.0, "y": 0.0},
                    "facingRadians": 0.0, "canAct": True,
                })
        inp = make_inp(observation)
        intents = controller.decide(inp)
        carrier_intent, reason = selected_pass(controller, intents)

        self.assertIsNotNone(carrier_intent, f"{reason_key}: no pass selected")
        self.assertEqual(carrier_intent.action_type, "pass")
        self.assertEqual(reason, reason_key)

        # 1. The generated receiver must be a real, acting teammate.
        rid = carrier_intent.receiver_id
        self.assertIsNotNone(rid, f"{reason_key}: receiver_id lost")
        self.assertNotEqual(rid, spec["carrier"], f"{reason_key}: passed to self")
        teammate = next((p for p in inp.state.outfield_us() if p.id == rid), None)
        self.assertIsNotNone(teammate, f"{reason_key}: {rid} is not a teammate")
        self.assertTrue(teammate.can_act, f"{reason_key}: {rid} cannot act")

        # 2. There must be a collection point to move to.
        self.assertIsNotNone(carrier_intent.collection_point, reason_key)

        # 3. The selected receiver is the one who receives the order, and the
        #    order goes to the collection point, not to the aim point.
        order = intents.get(rid)
        self.assertIsNotNone(order, f"{reason_key}: no intent for {rid}")
        cx, cy = carrier_intent.collection_point
        self.assertAlmostEqual(order.tx, cx, places=6, msg=f"{reason_key}: tx")
        self.assertAlmostEqual(order.ty, cy, places=6, msg=f"{reason_key}: ty")
        self.assertEqual(order.action_type, "none", f"{reason_key}: order not a move")

    def test_through_ball_receiver_identity(self):
        self._assert_identity("through_ball")

    def test_cross_receiver_identity(self):
        self._assert_identity("cross")

    def test_cutback_receiver_identity(self):
        self._assert_identity("cutback")

    def test_one_two_receiver_identity(self):
        self._assert_identity("high_press_onetwo")

    def test_wall_pass_receiver_identity(self):
        self._assert_identity("wall_pass")

    def test_progressive_pass_receiver_identity(self):
        self._assert_identity("safe_pass_to_w")

    def test_switch_play_receiver_identity(self):
        self._assert_identity("switch_play")

    def test_every_required_type_is_covered(self):
        self.assertEqual(set(REQUIRED_TYPES), set(PASS_SCENARIOS))


class TwoPlayerAmbiguityTests(unittest.TestCase):
    """Two teammates near one target must not confuse the receiver.

    The `switch_play` fixture is the discriminating case: the pass is generated
    for `w`, but `cd` stands 0.86 m from the collection point and comes *first*
    in the outfield player list. The removed "first outfielder within 10 m" scan
    therefore picked `cd` and sent the collection order to the wrong player.
    """

    AMBIGUOUS = "switch_play"

    def test_fixture_is_genuinely_ambiguous(self):
        controller = PolicyController()
        inp = make_inp(build(PASS_SCENARIOS[self.AMBIGUOUS]))
        intents = controller.decide(inp)
        carrier_intent, reason = selected_pass(controller, intents)
        self.assertEqual(reason, self.AMBIGUOUS)
        rid = carrier_intent.receiver_id
        cx, cy = carrier_intent.collection_point

        decoys = [
            p for p in inp.state.outfield_us()
            if p.id != rid and math.dist((p.x, p.y), (cx, cy)) < 10.0
        ]
        self.assertTrue(decoys, "fixture no longer has an ambiguous teammate")

        # At least one decoy must be BOTH closer to the collection point than the
        # true receiver AND earlier in the player list -- that is the case the
        # legacy first-within-10 m scan gets wrong.
        receiver = next(p for p in inp.state.outfield_us() if p.id == rid)
        order = [p.id for p in inp.state.outfield_us()]
        r_dist = math.dist((receiver.x, receiver.y), (cx, cy))
        discriminating = [
            d for d in decoys
            if math.dist((d.x, d.y), (cx, cy)) < r_dist
            and order.index(d.id) < order.index(rid)
        ]
        self.assertTrue(
            discriminating,
            f"no decoy beats {rid} on both closeness and list order; decoys={decoys}",
        )
        self.assertNotEqual(legacy_receiver(inp, carrier_intent.pid, (cx, cy)), rid)

    def test_decoy_does_not_steal_the_collection_order(self):
        controller = PolicyController()
        inp = make_inp(build(PASS_SCENARIOS[self.AMBIGUOUS]))
        intents = controller.decide(inp)
        carrier_intent, _ = selected_pass(controller, intents)
        rid = carrier_intent.receiver_id
        cx, cy = carrier_intent.collection_point

        order = intents[rid]
        self.assertAlmostEqual(order.tx, cx, places=6)
        self.assertAlmostEqual(order.ty, cy, places=6)
        self.assertEqual(order.action_type, "none")

        for d in inp.state.outfield_us():
            if d.id == rid:
                continue
            d_order = intents[d.id]
            self.assertFalse(
                abs(d_order.tx - cx) < 1e-6 and abs(d_order.ty - cy) < 1e-6,
                f"{d.id} was sent the collection order meant for {rid}",
            )

    def test_legacy_heuristic_diverges_somewhere_in_the_suite(self):
        """Keeps the ambiguity coverage honest as the board is recalibrated."""
        divergences = []
        for key in REQUIRED_TYPES:
            controller = PolicyController()
            inp = make_inp(build(PASS_SCENARIOS[key]))
            intents = controller.decide(inp)
            carrier_intent, _ = selected_pass(controller, intents)
            if carrier_intent is None or carrier_intent.receiver_id is None:
                continue
            legacy = legacy_receiver(
                inp, carrier_intent.pid, carrier_intent.collection_point
            )
            if legacy is not None and legacy != carrier_intent.receiver_id:
                divergences.append((key, carrier_intent.receiver_id, legacy))
        self.assertTrue(divergences, "no fixture distinguishes identity from the legacy scan")


class MetadataTests(unittest.TestCase):
    def test_wire_format_is_unchanged(self):
        """receiver_id / collection_point must not leak to the engine."""
        controller = PolicyController()
        inp = make_inp(build(PASS_SCENARIOS["cross"]))
        intents = controller.decide(inp)
        for intent in intents.values():
            wire = intent.to_wire()
            self.assertNotIn("receiver_id", wire)
            self.assertNotIn("collection_point", wire)
            self.assertIn("move", wire)
            self.assertIn("action", wire)

    def test_wire_uses_the_aim_target_not_the_collection_point(self):
        """The engine receives the pass target exactly as generated."""
        controller = PolicyController()
        inp = make_inp(build(PASS_SCENARIOS["through_ball"]))
        intents = controller.decide(inp)
        carrier_intent, _ = selected_pass(controller, intents)
        wire = carrier_intent.to_wire()
        self.assertAlmostEqual(wire["action"]["target"]["x"], carrier_intent.action_target[0], places=3)
        self.assertAlmostEqual(wire["action"]["target"]["y"], carrier_intent.action_target[1], places=3)

    def test_non_pass_intents_have_no_receiver(self):
        controller = PolicyController()
        inp = make_inp(build(PASS_SCENARIOS["cross"]))
        intents = controller.decide(inp)
        for intent in intents.values():
            if intent.action_type != "pass":
                self.assertIsNone(intent.receiver_id)


if __name__ == "__main__":
    unittest.main()
