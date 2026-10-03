"""End-to-end longitudinal orientation invariance through RuntimeManager."""

from __future__ import annotations

import copy
import math
import random
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.geom import PITCH_LENGTH
from src.runtime import RuntimeManager, _normalize_state
from src.state import GameState, WorldModel
from src.tactics import assign_roles, detect, plan_press
from src.teamplan import TeamPlanPhase

TOL = 1e-6
LENGTH = float(PITCH_LENGTH)


def _player(pid, role, x, y, *, vx=0.0, vy=0.0, facing=0.0, can_act=True):
    return {
        "id": pid,
        "role": role,
        "position": {"x": float(x), "y": float(y)},
        "velocity": {"x": float(vx), "y": float(vy)},
        "facingRadians": float(facing),
        "canAct": can_act,
    }


def observation(game_id="orientation", *, sequence=0, tick=0, ball=None,
                us=None, them=None, time_remaining=45.0):
    if us is None:
        us = [
            _player("gk", "goalkeeper", 6, 20),
            _player("cd", "outfield", 15, 20),
            _player("am", "outfield", 37, 20),
            _player("w", "outfield", 33, 8),
            _player("st", "outfield", 45, 25),
        ]
    if them is None:
        them = [
            _player("tgk", "goalkeeper", 54, 20),
            _player("t1", "outfield", 48, 12),
            _player("t2", "outfield", 43, 18),
            _player("t3", "outfield", 39, 27),
            _player("t4", "outfield", 31, 33),
        ]
    if ball is None:
        ball = {
            "position": {"x": 37, "y": 20},
            "velocity": {"x": 0, "y": 0},
            "possessingTeam": "us",
            "possessedBy": "am",
        }
    return {
        "protocolVersion": "1.0", "gameId": game_id, "sequence": sequence,
        "simulationTick": tick, "applyAtTick": tick,
        "timeRemainingSeconds": time_remaining, "phase": "openPlay",
        "score": {"us": 1, "them": 0}, "ball": copy.deepcopy(ball),
        "us": copy.deepcopy(us), "them": copy.deepcopy(them),
    }


def mirror_observation(obs, *, game_id=None):
    twin = copy.deepcopy(obs)
    if game_id is not None:
        twin["gameId"] = game_id
    for team in ("us", "them"):
        for p in twin[team]:
            p["position"]["x"] = LENGTH - p["position"]["x"]
            p["velocity"]["x"] = -p["velocity"].get("x", 0.0)
            p["facingRadians"] = math.pi - p.get("facingRadians", 0.0)
    twin["ball"]["position"]["x"] = LENGTH - twin["ball"]["position"]["x"]
    twin["ball"]["velocity"]["x"] = -twin["ball"]["velocity"].get("x", 0.0)
    return twin


def _normalized_wire(decision, attack_sign):
    result = {}
    for item in decision["intents"]:
        move = item["move"]["target"]
        face = item["face"]
        target = item["action"].get("target")
        result[item["playerId"]] = (
            move["x"] if attack_sign == 1 else LENGTH - move["x"], move["y"],
            face["x"] if attack_sign == 1 else LENGTH - face["x"], face["y"],
            item["move"]["speed"], item["action"]["type"],
            None if target is None else (
                target["x"] if attack_sign == 1 else LENGTH - target["x"], target["y"]
            ), item["action"].get("power"),
        )
    return result


def _assert_wire_equivalent(test, decision_a, decision_b):
    a, b = _normalized_wire(decision_a, 1), _normalized_wire(decision_b, -1)
    test.assertEqual(a.keys(), b.keys())
    for pid in a:
        test.assertEqual(a[pid][5], b[pid][5], f"{pid} action")
        for index in (0, 1, 2, 3, 4):
            test.assertAlmostEqual(a[pid][index], b[pid][index], delta=TOL,
                                   msg=f"{pid} intent component {index}: {a[pid]} != {b[pid]}")
        for index in (6,):
            if a[pid][index] is None or b[pid][index] is None:
                test.assertEqual(a[pid][index], b[pid][index], f"{pid} action target")
            else:
                test.assertAlmostEqual(a[pid][index][0], b[pid][index][0], delta=TOL)
                test.assertAlmostEqual(a[pid][index][1], b[pid][index][1], delta=TOL)
        if a[pid][7] is None or b[pid][7] is None:
            test.assertEqual(a[pid][7], b[pid][7], f"{pid} action power")
        else:
            test.assertAlmostEqual(a[pid][7], b[pid][7], delta=TOL)


def _assert_state_equivalent(test, state_a, state_b):
    for field in ("protocol_version", "sequence", "simulation_tick",
                  "apply_at_tick", "phase", "score_us", "score_them"):
        test.assertEqual(getattr(state_a, field), getattr(state_b, field), field)
    test.assertAlmostEqual(state_a.time_remaining, state_b.time_remaining, delta=TOL)
    for a, b in ((state_a.ball, state_b.ball),):
        for field in ("x", "y", "vx", "vy"):
            test.assertAlmostEqual(getattr(a, field), getattr(b, field), delta=TOL,
                                   msg=f"ball.{field}")
        test.assertEqual((a.possessing_team, a.possessing_player),
                         (b.possessing_team, b.possessing_player))
    for players_a, players_b in ((state_a.us, state_b.us), (state_a.them, state_b.them)):
        test.assertEqual(len(players_a), len(players_b))
        for a, b in zip(players_a, players_b):
            test.assertEqual((a.id, a.team, a.role, a.can_act),
                             (b.id, b.team, b.role, b.can_act))
            for field in ("x", "y", "vx", "vy", "facing"):
                test.assertAlmostEqual(getattr(a, field), getattr(b, field), delta=TOL,
                                       msg=f"{a.id}.{field}")


def _assert_plan_equal(test, a, b):
    test.assertEqual(a is None, b is None)
    if a is None:
        return
    for field in ("phase", "carrier_id", "created_tick", "expires_tick",
                  "primary_runner_id", "secondary_runner_id", "support_player_id",
                  "rest_defender_id", "shot_expected", "shot_committed", "reason",
                  "last_possession_tick"):
        test.assertEqual(getattr(a, field), getattr(b, field), field)
    for field in ("rebound_zone_x", "rebound_zone_y", "far_post_target_y",
                  "last_ball_x", "last_ball_y", "confidence"):
        av, bv = getattr(a, field), getattr(b, field)
        if av is None or bv is None:
            test.assertEqual(av, bv, field)
        elif field.endswith("_x"):
            test.assertAlmostEqual(av, bv, delta=TOL, msg=field)
        else:
            test.assertAlmostEqual(av, bv, delta=TOL, msg=field)
    test.assertEqual(len(a.targets), len(b.targets))
    for ta, tb in zip(a.targets, b.targets):
        test.assertEqual((ta.player_id, ta.role, ta.priority, ta.description),
                         (tb.player_id, tb.role, tb.priority, tb.description))
        test.assertAlmostEqual(ta.target_x, tb.target_x, delta=TOL)
        test.assertAlmostEqual(ta.target_y, tb.target_y, delta=TOL)


class RuntimeOrientationTests(unittest.TestCase):
    def _paired_runtime(self, obs_a, obs_b):
        # TeamPlan managers are globally keyed by gameId, so the mirrored twin
        # must be a distinct match context even though all football state is the
        # same. gameId is transport identity, not tactical input.
        if obs_a["gameId"] == obs_b["gameId"]:
            obs_b["gameId"] = f"{obs_b['gameId']}-orientation-mirror"
        a, b = RuntimeManager(), RuntimeManager()
        a.start_match({"gameId": obs_a["gameId"], "randomSeed": "orientation-seed"})
        b.start_match({"gameId": obs_b["gameId"], "randomSeed": "orientation-seed"})
        da, db = a.decide(obs_a), b.decide(obs_b)
        return a, b, da, db

    def _assert_pair(self, obs_a, obs_b):
        a, b, da, db = self._paired_runtime(obs_a, obs_b)
        ca = a.get_or_create(obs_a)
        cb = b.get_or_create(obs_b)
        self.assertTrue(ca.attack_direction_known)
        self.assertTrue(cb.attack_direction_known)
        self.assertEqual((ca.attack_sign, cb.attack_sign), (1, -1))
        sa = GameState.from_observation(obs_a)
        sb = GameState.from_observation(obs_b)
        na, nb = _normalize_state(sa, ca.attack_sign), _normalize_state(sb, cb.attack_sign)
        _assert_state_equivalent(self, na, nb)
        wa, wb = WorldModel.build(na), WorldModel.build(nb)
        self.assertEqual((wa.ball_zone, wa.ball_side, wa.ball_near_wall),
                         (wb.ball_zone, wb.ball_side, wb.ball_near_wall))
        for field in ("pressure_on_ball", "our_control", "their_control"):
            self.assertAlmostEqual(getattr(wa, field), getattr(wb, field), delta=TOL,
                                   msg=f"world.{field}")
        roles_a = assign_roles(na, ca.slots)
        roles_b = assign_roles(nb, cb.slots)
        self.assertEqual(roles_a, roles_b)
        self.assertEqual(detect(na, wa, ca.genome, False), detect(nb, wb, cb.genome, False))
        press_a = plan_press(na, wa, ca.genome, roles_a)
        press_b = plan_press(nb, wb, cb.genome, roles_b)
        self.assertEqual(press_a, press_b)
        self.assertEqual(ca.opp.summary(), cb.opp.summary())
        _assert_wire_equivalent(self, da, db)
        return a, b

    def test_orientation_is_explicitly_unknown_until_a_defensive_fifth_keeper_is_seen(self):
        manager = RuntimeManager()
        manager.start_match({"gameId": "unknown-orientation", "randomSeed": "s"})
        obs = observation("unknown-orientation", us=[
            _player("cd", "outfield", 15, 20),
            _player("am", "outfield", 37, 20),
            _player("w", "outfield", 33, 8),
            _player("st", "outfield", 45, 25),
        ])
        manager.decide(obs)
        ctx = manager.get_or_create(obs)
        self.assertFalse(ctx.attack_direction_known)
        self.assertEqual(ctx.attack_sign, 1)  # provisional only

        central_keeper = observation("unknown-orientation", sequence=1, tick=1,
                                     us=[_player("gk", "goalkeeper", 30, 20)] + obs["us"])
        manager.decide(central_keeper)
        self.assertFalse(ctx.attack_direction_known)

        later = observation("unknown-orientation", sequence=2, tick=2,
                            us=[_player("gk", "goalkeeper", 54, 20)] + obs["us"])
        manager.decide(later)
        self.assertTrue(ctx.attack_direction_known)
        self.assertEqual(ctx.attack_sign, -1)

    def test_full_runtime_pipeline_mirrors_for_controlled_action_situations(self):
        cases = {
            "carry": {},
            "pass": {"ball": {"position": {"x": 24, "y": 20}, "velocity": {"x": 0, "y": 0}, "possessingTeam": "us", "possessedBy": "cd"}},
            "through": {"ball": {"position": {"x": 36, "y": 20}, "velocity": {"x": 0, "y": 0}, "possessingTeam": "us", "possessedBy": "am"}},
            "switch": {"ball": {"position": {"x": 43, "y": 20}, "velocity": {"x": 0, "y": 0}, "possessingTeam": "us", "possessedBy": "st"}},
            "wall_pass": {"ball": {"position": {"x": 39, "y": 2}, "velocity": {"x": 0, "y": 0}, "possessingTeam": "us", "possessedBy": "w"}},
            "shot": {"ball": {"position": {"x": 53, "y": 20}, "velocity": {"x": 0, "y": 0}, "possessingTeam": "us", "possessedBy": "st"}},
            "loose": {"ball": {"position": {"x": 31, "y": 5}, "velocity": {"x": 7, "y": -3}, "possessingTeam": None, "possessedBy": None}},
            "press": {"ball": {"position": {"x": 42, "y": 15}, "velocity": {"x": 0, "y": 0}, "possessingTeam": "them", "possessedBy": "t2"}},
            "goalkeeper": {"ball": {"position": {"x": 6, "y": 20}, "velocity": {"x": 0, "y": 0}, "possessingTeam": "us", "possessedBy": "gk"}},
        }
        for label, changes in cases.items():
            with self.subTest(situation=label):
                obs_a = observation(f"orient-{label}", **changes)
                obs_b = mirror_observation(obs_a)
                obs_b["gameId"] = f"orient-{label}"
                manager_a, manager_b = self._assert_pair(obs_a, obs_b)
                if label == "press":
                    ctx = manager_a.get_or_create(obs_a)
                    state = _normalize_state(GameState.from_observation(obs_a), ctx.attack_sign)
                    plan = plan_press(state, WorldModel.build(state), ctx.genome,
                                      assign_roles(state, ctx.slots))
                    self.assertTrue(plan.pressers, "controlled defending fixture must exercise pressing")

    def test_runtime_mirrors_real_carry_pass_variants_shot_and_keeper_distribution(self):
        from tests.test_policy import obs as policy_obs
        from tests.test_receiver_identity import PASS_SCENARIOS, build as build_pass

        # The scenario fixtures exercise different candidate generators through
        # the production runtime, rather than inferring an action from a label.
        for name in ("through_ball", "switch_play", "wall_pass", "safe_pass_to_w"):
            with self.subTest(action=name):
                spec = PASS_SCENARIOS[name]
                base = build_pass(spec)
                game_id = f"runtime-action-{name}"
                base["gameId"] = game_id
                base["us"][0]["position"]["x"] = 6.0
                base["sequence"] = base["simulationTick"] = base["applyAtTick"] = 1
                mirror = mirror_observation(base)
                mirror["gameId"] = game_id
                a, b, da, db = self._paired_runtime(base, mirror)
                _assert_wire_equivalent(self, da, db)
                action_a = next(i for i in da["intents"] if i["playerId"] == spec["carrier"])["action"]["type"]
                action_b = next(i for i in db["intents"] if i["playerId"] == spec["carrier"])["action"]["type"]
                self.assertEqual((action_a, action_b), ("pass", "pass"))
                ctxa, ctxb = a.get_or_create(base), b.get_or_create(mirror)
                reason_a = ctxa.policy.log.possessions_decided[-1]["reason"]
                reason_b = ctxb.policy.log.possessions_decided[-1]["reason"]
                self.assertEqual((reason_a, reason_b), (name, name))

        shot = policy_obs((52, 20), "us", our_st_x=52, them_x=10)
        shot["them"][0]["position"] = {"x": 45.0, "y": 20.0}
        shot["gameId"] = "runtime-action-shot"
        shot["sequence"] = shot["simulationTick"] = shot["applyAtTick"] = 1
        shot_mirror = mirror_observation(shot)
        shot_mirror["gameId"] = shot["gameId"]
        a, b, da, db = self._paired_runtime(shot, shot_mirror)
        _assert_wire_equivalent(self, da, db)
        self.assertEqual(next(i["action"]["type"] for i in da["intents"] if i["playerId"] == "st"), "shoot")
        self.assertEqual(next(i["action"]["type"] for i in db["intents"] if i["playerId"] == "st"), "shoot")
        manager_a = a.get_or_create(shot).policy._team_plan_manager
        manager_b = b.get_or_create(shot_mirror).policy._team_plan_manager
        self.assertIsNotNone(manager_a)
        self.assertIsNotNone(manager_b)
        assert manager_a is not None and manager_b is not None
        plan_a = manager_a.current_plan
        plan_b = manager_b.current_plan
        self.assertIsNotNone(plan_a)
        self.assertIsNotNone(plan_b)
        assert plan_a is not None and plan_b is not None
        self.assertTrue(plan_a.shot_committed)
        self.assertTrue(plan_b.shot_committed)
        _assert_plan_equal(self, plan_a, plan_b)

        # The default middle-third state is a genuine carry decision: no ball
        # action is requested and the carrier receives a forward movement target.
        carry = observation("runtime-action-carry")
        ca, cb, da, db = self._paired_runtime(carry, mirror_observation(carry))
        _assert_wire_equivalent(self, da, db)
        for decision in (da, db):
            carrier_intent = next(i for i in decision["intents"] if i["playerId"] == "am")
            self.assertEqual(carrier_intent["action"]["type"], "none")
            target = carrier_intent["move"]["target"]
            self.assertNotEqual(target["x"], 37.0)

        # Goalkeeper distribution scenario reaches a real keeper pass in both
        # frames, and its action target must normalize to the same location.
        keeper = observation("runtime-action-goalkeeper", ball={
            "position": {"x": 6, "y": 20}, "velocity": {"x": 0, "y": 0},
            "possessingTeam": "us", "possessedBy": "gk",
        })
        keeper_mirror = mirror_observation(keeper)
        keeper_mirror["gameId"] = keeper["gameId"]
        _, _, da, db = self._paired_runtime(keeper, keeper_mirror)
        _assert_wire_equivalent(self, da, db)
        self.assertEqual(next(i["action"]["type"] for i in da["intents"] if i["playerId"] == "gk"), "pass")
        self.assertEqual(next(i["action"]["type"] for i in db["intents"] if i["playerId"] == "gk"), "pass")

    def test_runtime_denormalizes_pass_collection_point_metadata(self):
        from tests.test_receiver_identity import PASS_SCENARIOS, build as build_pass
        from src import runtime as runtime_module

        base = build_pass(PASS_SCENARIOS["through_ball"])
        base["gameId"] = "collection-orientation"
        base["us"][0]["position"]["x"] = 6.0
        mirror = mirror_observation(base)
        mirror["gameId"] = f"{base['gameId']}-mirror"
        a, b = RuntimeManager(), RuntimeManager()
        a.start_match({"gameId": base["gameId"], "randomSeed": "same"})
        b.start_match({"gameId": mirror["gameId"], "randomSeed": "same"})
        captured = []
        original = runtime_module._denormalize_intents

        def capture(intents, sign):
            denormalized = original(intents, sign)
            captured.append((sign, denormalized))
            return denormalized

        with patch.object(runtime_module, "_denormalize_intents", side_effect=capture):
            a.decide(base)
            b.decide(mirror)
        self.assertEqual([sign for sign, _ in captured], [1, -1])
        intent_a, intent_b = captured[0][1]["st"], captured[1][1]["st"]
        self.assertEqual(intent_a.receiver_id, intent_b.receiver_id)
        self.assertIsNotNone(intent_a.collection_point)
        self.assertIsNotNone(intent_b.collection_point)
        assert intent_a.collection_point is not None and intent_b.collection_point is not None
        self.assertAlmostEqual(intent_a.collection_point[0] + intent_b.collection_point[0], LENGTH, delta=TOL)
        self.assertAlmostEqual(intent_a.collection_point[1], intent_b.collection_point[1], delta=TOL)
        self.assertIsNotNone(intent_a.action_target)
        self.assertIsNotNone(intent_b.action_target)
        assert intent_a.action_target is not None and intent_b.action_target is not None
        self.assertAlmostEqual(intent_a.action_target[0] + intent_b.action_target[0], LENGTH, delta=TOL)
        self.assertAlmostEqual(intent_a.action_target[1], intent_b.action_target[1], delta=TOL)

    def test_keeper_orientation_latches_survives_advance_missing_observation_and_resets_per_match(self):
        for initial_x, sign in ((6.0, 1), (54.0, -1)):
            game_id = f"keeper-{initial_x}"
            manager = RuntimeManager()
            manager.start_match({"gameId": game_id, "randomSeed": "s"})
            obs = observation(game_id, us=[_player("gk", "goalkeeper", initial_x, 20)] + observation()["us"][1:])
            manager.decide(obs)
            ctx = manager.get_or_create(obs)
            self.assertTrue(ctx.attack_direction_known)
            self.assertEqual(ctx.attack_sign, sign)
            for tick, x in enumerate((20, 30, 35, 45), start=1):
                moving = copy.deepcopy(obs)
                moving["sequence"] = moving["simulationTick"] = moving["applyAtTick"] = tick
                moving["us"][0]["position"]["x"] = x
                manager.decide(moving)
                self.assertEqual(ctx.attack_sign, sign)
            missing = copy.deepcopy(obs)
            missing["sequence"] = missing["simulationTick"] = missing["applyAtTick"] = 6
            missing["us"] = [p for p in missing["us"] if p["role"] != "goalkeeper"]
            manager.decide(missing)
            self.assertEqual(ctx.attack_sign, sign)

        manager = RuntimeManager()
        manager.start_match({"gameId": "new-orientation", "randomSeed": "s"})
        manager.decide(observation("new-orientation"))
        self.assertEqual(manager.get_or_create(observation("new-orientation")).attack_sign, 1)
        manager.start_match({"gameId": "new-orientation", "randomSeed": "s2"})
        new_ctx = manager.get_or_create(observation("new-orientation"))
        self.assertFalse(new_ctx.attack_direction_known)
        self.assertEqual(new_ctx.attack_sign, 1)
        reversed_obs = mirror_observation(observation("new-orientation"))
        manager.decide(reversed_obs)
        self.assertEqual(new_ctx.attack_sign, -1)
        manager.start_match({"gameId": "distinct-new-game", "randomSeed": "s3"})
        distinct = observation("distinct-new-game")
        distinct_ctx = manager.get_or_create(distinct)
        self.assertFalse(distinct_ctx.attack_direction_known)
        distinct_mirror = mirror_observation(distinct)
        manager.decide(distinct_mirror)
        self.assertTrue(distinct_ctx.attack_direction_known)
        self.assertEqual(distinct_ctx.attack_sign, -1)

    def test_teamplan_persistence_and_shot_followup_are_canonically_equivalent(self):
        gid = "orientation-sequence"
        gid_b = f"{gid}-mirror"
        a, b = RuntimeManager(), RuntimeManager()
        a.start_match({"gameId": gid, "randomSeed": "same"})
        b.start_match({"gameId": gid_b, "randomSeed": "same"})
        steps = [
            ("controlled", "us", "am", 37, 20),
            ("carry", "us", "am", 39, 20),
            ("planned-pass", "us", "am", 41, 20),
            ("new-carrier", "us", "st", 47, 25),
        ]
        for tick, (_, team, carrier, x, y) in enumerate(steps):
            base = observation(gid, sequence=tick, tick=tick, ball={
                "position": {"x": x, "y": y}, "velocity": {"x": 0, "y": 0},
                "possessingTeam": team, "possessedBy": carrier,
            })
            mirror = mirror_observation(base)
            mirror["gameId"] = gid_b
            da, db = a.decide(base), b.decide(mirror)
            _assert_wire_equivalent(self, da, db)
            ca, cb = a.get_or_create(base), b.get_or_create(mirror)
            _assert_state_equivalent(
                self,
                _normalize_state(GameState.from_observation(base), ca.attack_sign),
                _normalize_state(GameState.from_observation(mirror), cb.attack_sign),
            )
            pma, pmb = ca.policy._team_plan_manager, cb.policy._team_plan_manager
            self.assertIsNotNone(pma)
            self.assertIsNotNone(pmb)
            assert pma is not None and pmb is not None
            _assert_plan_equal(self, pma.current_plan, pmb.current_plan)

        from tests.test_policy import obs as policy_obs
        shot = policy_obs((52, 20), "us", our_st_x=52, them_x=10)
        shot["them"][0]["position"] = {"x": 45.0, "y": 20.0}
        shot["gameId"] = gid
        shot["sequence"] = shot["simulationTick"] = shot["applyAtTick"] = 4
        shot_mirror = mirror_observation(shot)
        shot_mirror["gameId"] = gid_b
        da, db = a.decide(shot), b.decide(shot_mirror)
        _assert_wire_equivalent(self, da, db)
        self.assertEqual(next(i["action"]["type"] for i in da["intents"] if i["playerId"] == "st"), "shoot")
        self.assertEqual(next(i["action"]["type"] for i in db["intents"] if i["playerId"] == "st"), "shoot")
        ca, cb = a.get_or_create(shot), b.get_or_create(shot_mirror)
        pma, pmb = ca.policy._team_plan_manager, cb.policy._team_plan_manager
        self.assertIsNotNone(pma)
        self.assertIsNotNone(pmb)
        assert pma is not None and pmb is not None
        pa, pb = pma.current_plan, pmb.current_plan
        self.assertIsNotNone(pa)
        self.assertIsNotNone(pb)
        assert pa is not None and pb is not None
        self.assertTrue(pa.shot_committed)
        self.assertTrue(pb.shot_committed)
        for tick, poss_team in ((5, None), (6, None), (7, "them")):
            bx = 55.0 if poss_team is None else 48.0
            base = observation(gid, sequence=tick, tick=tick, ball={
                "position": {"x": bx, "y": 20}, "velocity": {"x": 16 if poss_team is None else 0, "y": 0},
                "possessingTeam": poss_team, "possessedBy": "t1" if poss_team else None,
            })
            mirror = mirror_observation(base)
            mirror["gameId"] = gid_b
            da, db = a.decide(base), b.decide(mirror)
            _assert_wire_equivalent(self, da, db)
            ca, cb = a.get_or_create(base), b.get_or_create(mirror)
            pma, pmb = ca.policy._team_plan_manager, cb.policy._team_plan_manager
            self.assertIsNotNone(pma)
            self.assertIsNotNone(pmb)
            assert pma is not None and pmb is not None
            _assert_plan_equal(self, pma.current_plan, pmb.current_plan)
            if tick in (5, 6):
                self.assertIsNotNone(pma.current_plan)
                assert pma.current_plan is not None and pmb.current_plan is not None
                self.assertEqual(pma.current_plan.phase, TeamPlanPhase.SHOT_FOLLOWUP)
                self.assertEqual(pmb.current_plan.phase, TeamPlanPhase.SHOT_FOLLOWUP)
                if tick == 5:
                    canonical = _normalized_wire(da, 1)
                    plan = pma.current_plan
                    self.assertIsNotNone(plan.primary_runner_id)
                    self.assertIsNotNone(plan.secondary_runner_id)
                    self.assertIsNotNone(plan.rest_defender_id)
                    self.assertIsNotNone(plan.rebound_zone_x)
                    self.assertIsNotNone(plan.rebound_zone_y)
                    self.assertIsNotNone(plan.far_post_target_y)
                    assert plan.primary_runner_id and plan.secondary_runner_id and plan.rest_defender_id
                    assert (plan.rebound_zone_x is not None and plan.rebound_zone_y is not None
                            and plan.far_post_target_y is not None)
                    self.assertAlmostEqual(canonical[plan.primary_runner_id][0],
                                           plan.rebound_zone_x - 2.0, delta=TOL)
                    self.assertAlmostEqual(canonical[plan.primary_runner_id][1],
                                           plan.rebound_zone_y, delta=TOL)
                    self.assertAlmostEqual(canonical[plan.secondary_runner_id][0],
                                           plan.rebound_zone_x - 5.0, delta=TOL)
                    self.assertAlmostEqual(canonical[plan.secondary_runner_id][1],
                                           plan.far_post_target_y, delta=TOL)
                    rest_target = next(t for t in plan.targets if t.player_id == plan.rest_defender_id)
                    self.assertLess(rest_target.target_x, 55.0,
                                    "rest defender must remain behind the rebound runners")
            else:
                self.assertIsNone(pma.current_plan)
                self.assertIsNone(pmb.current_plan)

    def test_shot_followup_support_cuts_back_under_both_orientations(self):
        game_a, game_b = "followup-support", "followup-support-mirror"
        a, b = RuntimeManager(), RuntimeManager()
        a.start_match({"gameId": game_a, "randomSeed": "support-seed"})
        b.start_match({"gameId": game_b, "randomSeed": "support-seed"})
        base = observation(game_a, ball={
            "position": {"x": 6, "y": 20}, "velocity": {"x": 0, "y": 0},
            "possessingTeam": "us", "possessedBy": "gk",
        })
        mirror = mirror_observation(base)
        mirror["gameId"] = game_b
        a.decide(base)
        b.decide(mirror)
        ctx_a, ctx_b = a.get_or_create(base), b.get_or_create(mirror)
        pm_a, pm_b = ctx_a.policy._team_plan_manager, ctx_b.policy._team_plan_manager
        self.assertIsNotNone(pm_a)
        self.assertIsNotNone(pm_b)
        assert pm_a is not None and pm_b is not None
        plan_a, plan_b = pm_a.current_plan, pm_b.current_plan
        self.assertIsNotNone(plan_a)
        self.assertIsNotNone(plan_b)
        assert plan_a is not None and plan_b is not None
        self.assertIsNotNone(plan_a.support_player_id)
        self.assertIsNotNone(plan_a.rest_defender_id)
        _assert_plan_equal(self, plan_a, plan_b)
        for plan in (plan_a, plan_b):
            plan.shot_committed = True
            plan.rebound_zone_x = 57.0
            plan.rebound_zone_y = 24.0
            plan.far_post_target_y = 16.0

        loose = observation(game_a, sequence=1, tick=1, ball={
            "position": {"x": 55, "y": 24}, "velocity": {"x": 16, "y": 0},
            "possessingTeam": None, "possessedBy": None,
        })
        loose_mirror = mirror_observation(loose)
        loose_mirror["gameId"] = game_b
        da, db = a.decide(loose), b.decide(loose_mirror)
        _assert_wire_equivalent(self, da, db)
        plan = pm_a.current_plan
        plan_mirror = pm_b.current_plan
        self.assertIsNotNone(plan)
        self.assertIsNotNone(plan_mirror)
        assert plan is not None and plan_mirror is not None
        self.assertEqual(plan.phase, TeamPlanPhase.SHOT_FOLLOWUP)
        self.assertEqual(plan.support_player_id, plan_mirror.support_player_id)
        canonical = _normalized_wire(da, 1)
        assert plan.support_player_id is not None and plan.rest_defender_id is not None
        self.assertAlmostEqual(canonical[plan.support_player_id][0], 47.0, delta=TOL)
        self.assertAlmostEqual(canonical[plan.support_player_id][1], 20.0, delta=TOL)
        rest_target = next(t for t in plan.targets if t.player_id == plan.rest_defender_id)
        self.assertLess(rest_target.target_x, canonical[plan.support_player_id][0])

    def test_wall_proximity_is_invariant_at_all_four_edges_and_corners(self):
        for x, y in ((30, 1), (30, 39), (1, 20), (59, 20), (1, 1), (59, 39)):
            with self.subTest(x=x, y=y):
                obs_a = observation(f"wall-{x}-{y}", ball={
                    "position": {"x": x, "y": y}, "velocity": {"x": 2, "y": -1},
                    "possessingTeam": None, "possessedBy": None,
                })
                obs_b = mirror_observation(obs_a)
                self._assert_pair(obs_a, obs_b)

    def test_seeded_randomized_full_runtime_property_1000_pairs(self):
        rng = random.Random(20261002)
        for i in range(1000):
            game_id = f"random-orientation-{i}"
            keeper_x = rng.uniform(2.0, 11.0)
            us = [_player("gk", "goalkeeper", keeper_x, rng.uniform(17, 23),
                          vx=rng.uniform(-2, 2), facing=rng.uniform(-math.pi, math.pi))]
            them = [_player("tgk", "goalkeeper", rng.uniform(49, 58), rng.uniform(17, 23),
                            vx=rng.uniform(-2, 2), facing=rng.uniform(-math.pi, math.pi))]
            for j in range(1, 5):
                us.append(_player(f"u{j}", "outfield", rng.uniform(2, 58), rng.uniform(1, 39),
                                  vx=rng.uniform(-6, 6), vy=rng.uniform(-6, 6),
                                  facing=rng.uniform(-math.pi, math.pi)))
                them.append(_player(f"t{j}", "outfield", rng.uniform(2, 58), rng.uniform(1, 39),
                                    vx=rng.uniform(-6, 6), vy=rng.uniform(-6, 6),
                                    facing=rng.uniform(-math.pi, math.pi)))
            possession = rng.choice(("us", "them", None))
            possessor = rng.choice([f"u{j}" for j in range(1, 5)]) if possession == "us" else (
                rng.choice([f"t{j}" for j in range(1, 5)]) if possession == "them" else None)
            if possession == "us":
                holder = next(p for p in us if p["id"] == possessor)
                bx, by = holder["position"]["x"], holder["position"]["y"]
            elif possession == "them":
                holder = next(p for p in them if p["id"] == possessor)
                bx, by = holder["position"]["x"], holder["position"]["y"]
            else:
                bx, by = rng.uniform(0, 60), rng.uniform(0, 40)
            base = observation(game_id, us=us, them=them, sequence=0, ball={
                "position": {"x": bx, "y": by},
                "velocity": {"x": rng.uniform(-15, 15), "y": rng.uniform(-15, 15)},
                "possessingTeam": possession, "possessedBy": possessor,
            })
            twin = mirror_observation(base)
            self._assert_pair(base, twin)


if __name__ == "__main__":
    unittest.main()
