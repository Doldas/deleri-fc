"""Tests for the opponent zoo, the engine brain, and the generated teams.

The regressions asserted here are ones that were actually found and fixed while
building the pool, not hypotheticals. Each of these was a silent behaviour
change that made the arena report confident nonsense:

* `LightEngine.can_act` latched false forever after a knockdown
* `plan_to_obs` reported `canAct: true` during the 0.3 s kick cooldown
* `_apply_bias` clamped `shoot_range` (a metre value) into 0..1
* the pass search would play a 7 m retreat to the corner
* the shared brain silently ignored `TRAINED_PARAMS` when module-copied
"""

import json
import pathlib
import random
import subprocess
import sys
import unittest

ROOT = pathlib.Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import src.sim as sim  # noqa: E402
from src.light import LPlayer, PlanState, make_state  # noqa: E402
from src.opponents import engine_brain  # noqa: E402
from src.opponents.learned import ACT_HOLD, FEATURES  # noqa: E402
from src.opponents.registry import (  # noqa: E402
    LEARNED_NAMES,
    LURE_NAMES,
    REGISTRY,
    opponent_names,
)
from src.opponents.spec import make_spec  # noqa: E402
from src.opponents.zoo import make_tunable  # noqa: E402,F401

from training.scripts.engine_probe import build_module, make_controller, mirror  # noqa: E402


def _moves_toward(brain, bx: float, by: float) -> bool:
    """True if some outfielder is actually closing the ball down.

    Used instead of asserting on the private `_pressers` set alone, because a
    ball-watcher is only useful if somebody is told to go and get it.
    """
    if not brain.outfield:
        return False
    return min(engine_brain._dist(q.x, q.y, bx, by) for q in brain.outfield) \
        < engine_brain.CONTEST_RADIUS


def observation(seq=0, bx=30.0, by=20.0, team: "str | None" = "us",
                held: "str | None" = "d2", game="g",
                us=None, them=None):
    def mk(pid, role, x, y):
        return {
            "id": pid, "role": role, "position": {"x": x, "y": y},
            "velocity": {"x": 0.0, "y": 0.0},
            "facingRadians": 0.0, "canAct": True,
        }

    return {
        "protocolVersion": "1.0", "gameId": game, "sequence": seq,
        "simulationTick": seq * 6, "applyAtTick": seq * 6,
        "timeRemainingSeconds": 300.0, "phase": "openPlay",
        "score": {"us": 0, "them": 0},
        "ball": {
            "id": "b", "position": {"x": bx, "y": by},
            "velocity": {"x": 0.0, "y": 0.0},
            "possessingTeam": team, "possessedBy": held,
        },
        "us": us or [
            mk("gk", "goalkeeper", 3.0, 20.0),
            mk("d1", "outfield", 26.0, 20.0),
            mk("d2", "outfield", 22.0, 13.0),
            mk("d3", "outfield", 22.0, 27.0),
            mk("d4", "outfield", 30.0, 20.0),
        ],
        "them": them or [
            mk("gk", "goalkeeper", 57.0, 20.0),
            mk("d1", "outfield", 40.0, 18.0),
            mk("d2", "outfield", 42.0, 20.0),
            mk("d3", "outfield", 42.0, 22.0),
            mk("d4", "outfield", 44.0, 20.0),
        ],
    }


class SpecTests(unittest.TestCase):
    def test_shoot_range_is_a_length_not_a_fraction(self):
        """Regression: bias clamping turned shoot_range into 1.0 m."""
        for difficulty in (0.0, 0.5, 1.0):
            spec = make_spec("t", "possession", difficulty)
            self.assertGreater(spec.shoot_range, 5.0)
            self.assertLessEqual(spec.shoot_range, 32.0)

    def test_harder_is_never_quieter(self):
        soft = make_spec("t", "possession", 0.2)
        hard = make_spec("t", "possession", 1.0)
        self.assertGreater(soft.noise, hard.noise,
                           "a harder side should make fewer errors, not more")
        self.assertLessEqual(soft.compactness, hard.compactness + 1e-9)
        self.assertGreaterEqual(hard.shoot_range, soft.shoot_range)

    def test_every_family_builds(self):
        for family in ("possession", "high_press", "low_block", "counter",
                       "direct", "wall", "tika", "overload", "physical",
                       "gk_hell", "chaos", "lure", "learned"):
            spec = make_spec("t", family, 0.8)
            self.assertEqual(spec.family, family)


class EngineModelRegressionTests(unittest.TestCase):
    def test_can_act_recovers_after_knockdown(self):
        """Regression: can_act latched false, so a slapped player never
        passed or shot again for the rest of the match."""
        from src.config import ACTION_COOLDOWN  # noqa: F401
        from src.light import LightEngine

        state = make_state(
            [("gk", "goalkeeper", 3.0, 20.0), ("d1", "outfield", 30.0, 20.0)],
            [("gk", "goalkeeper", 57.0, 20.0), ("d1", "outfield", 30.0, 20.5)],
            ball=(30.0, 20.0),
        )
        state.player("us", "d1").grounded = 0.4  # type: ignore[union-attr]
        engine = LightEngine()
        for _ in range(10):
            # `step` returns a new PlanState rather than mutating in place.
            state = engine.step(state, {})
        player = state.player("us", "d1")
        assert player is not None
        self.assertEqual(player.grounded, 0.0)
        self.assertTrue(player.can_act, "can_act must recover once grounded expires")

    def test_plan_to_obs_reports_cooldown_truthfully(self):
        """Regression: plan_to_obs hardcoded canAct=True, so every kick-happy
        policy re-requested kicks for the whole cooldown and the engine dropped
        them."""
        from src.config import ACTION_COOLDOWN

        state = make_state(
            [("gk", "goalkeeper", 3.0, 20.0), ("d1", "outfield", 30.0, 20.0)],
            [("gk", "goalkeeper", 57.0, 20.0)],
            ball=(30.0, 20.0),
        )
        player = state.player("us", "d1")
        assert player is not None
        self.assertTrue(sim.plan_to_obs(state, "t", 0)["us"][1]["canAct"])

        player.cooldown = ACTION_COOLDOWN
        self.assertFalse(sim.plan_to_obs(state, "t", 1)["us"][1]["canAct"],
                         "cooldown must show as canAct=False")

        player.cooldown = 0.0
        player.grounded = 0.5
        self.assertFalse(sim.plan_to_obs(state, "t", 2)["us"][1]["canAct"],
                         "grounded must show as canAct=False")


class RegistryTests(unittest.TestCase):
    def test_pool_is_large_and_diverse(self):
        names = opponent_names()
        self.assertGreaterEqual(len(names), 40)
        self.assertEqual(len(names), len(set(names)))

    def test_every_factory_builds_a_working_controller(self):
        for name in opponent_names():
            controller = REGISTRY[name](random.Random(3))
            self.assertTrue(hasattr(controller, "decide"), name)

    def test_registry_preserves_the_legacy_names(self):
        for legacy in ("possession", "press", "direct", "defensive",
                       "random", "counter", "parkbus", "wall"):
            self.assertIn(legacy, sim.OPPONENTS)

    def test_lure_and_learned_names_are_not_plain_presets(self):
        """Regression: `learned-rookie` used to be a hand-tuned preset, which
        made the arena ranking meaningless."""
        for name in opponent_names():
            if name.startswith("lure-"):
                self.assertTrue(name[len("lure-"):] in LURE_NAMES)
            if name.startswith("learned-"):
                self.assertTrue(name[len("learned-"):] in LEARNED_NAMES)

    def test_lure_variants_differ_behaviourally(self):
        modules = {
            n: build_module({"family": "lure", "deception": 0.9,
                             "noise": 0.02 + 0.05 * i})
            for i, n in enumerate(LURE_NAMES)
        }
        outs = {}
        for name, module in modules.items():
            decisions = [
                json.dumps(module.decide(observation(seq=s, game=name)), sort_keys=True)
                for s in range(20)
            ]
            outs[name] = decisions
        self.assertGreater(len(set(map(tuple, outs.values()))), 1,
                           "lure variants produced identical play")

    def test_learned_weights_move(self):
        module = build_module({"family": "learned", "learns": True,
                               "noise": 0.15, "shoot_range": 20.0})
        for seq in range(60):
            module.decide(observation(seq=seq, game="g", bx=20.0 + seq * 0.4))
        brain = module._BRAINS["g"]
        self.assertTrue(any(v != 0.0 for v in brain.w),
                        "online learner never updated a single weight")
        self.assertLessEqual(len(brain.w), 7 * FEATURES)


class EngineBrainTests(unittest.TestCase):
    def test_brain_is_self_contained(self):
        """It ships inside a bare Alpine container: no intra-package imports."""
        source = (ROOT / "src" / "opponents" / "engine_brain.py").read_text()
        self.assertNotIn("from .", source)
        self.assertNotIn("from ..", source)
        import ast
        mods = {n.names[0].name.split(".")[0]
                for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Import)}
        mods |= {n.module.split(".")[0]
                 for n in ast.walk(ast.parse(source))
                 if isinstance(n, ast.ImportFrom) and n.module}
        self.assertLessEqual(mods, {"hashlib", "math", "__future__"})

    def test_decision_shape(self):
        module = build_module({"family": "possession"})
        decision = module.decide(observation())
        self.assertEqual(decision["protocolVersion"], "1.0")
        self.assertEqual(decision["gameId"], "g")
        self.assertEqual(decision["sequence"], 0)
        self.assertEqual(len(decision["intents"]), 5)
        for intent in decision["intents"]:
            self.assertIn(intent["action"]["type"],
                          ("none", "pass", "shoot", "clear", "tackle", "slap"))
            speed = intent["move"]["speed"]
            self.assertGreaterEqual(speed, 0.0)
            self.assertLessEqual(speed, 1.0)
            self.assertGreaterEqual(intent["move"]["target"]["x"], 0.0)
            self.assertLessEqual(intent["move"]["target"]["x"], 60.0)

    def test_goalkeeper_never_leaves_the_defensive_fifth_when_positioning(self):
        """Automatic GK handling only applies inside its own fifth; a keeper
        that drifts out of it stops being a goalkeeper."""
        module = build_module({"family": "possession"})
        for bx in (5.0, 20.0, 40.0, 55.0):
            decision = module.decide(observation(seq=1, bx=bx, by=8.0,
                                                 team="them", held="d1"))
            gk = next(i for i in decision["intents"] if i["playerId"] == "gk")
            self.assertLessEqual(gk["move"]["target"]["x"], 0.2 * 60.0 + 1e-6,
                                 f"keeper left the fifth with ball at x={bx}")

    def test_shoots_from_inside_the_range(self):
        """Shooting range is measured from the *carrier*, not the ball."""
        module = build_module({"family": "direct", "shoot_range": 22.0})
        near = [
            {"id": "gk", "role": "goalkeeper", "position": {"x": 3.0, "y": 20.0},
             "velocity": {"x": 0.0, "y": 0.0}, "facingRadians": 0.0, "canAct": True},
            {"id": "d1", "role": "outfield", "position": {"x": 44.0, "y": 20.0},
             "velocity": {"x": 0.0, "y": 0.0}, "facingRadians": 0.0, "canAct": True},
            {"id": "d2", "role": "outfield", "position": {"x": 40.0, "y": 10.0},
             "velocity": {"x": 0.0, "y": 0.0}, "facingRadians": 0.0, "canAct": True},
            {"id": "d3", "role": "outfield", "position": {"x": 40.0, "y": 30.0},
             "velocity": {"x": 0.0, "y": 0.0}, "facingRadians": 0.0, "canAct": True},
            {"id": "d4", "role": "outfield", "position": {"x": 52.0, "y": 20.0},
             "velocity": {"x": 0.0, "y": 0.0}, "facingRadians": 0.0, "canAct": True},
        ]
        shots = 0
        for seq in range(40):
            decision = module.decide(observation(seq=seq, bx=52.0, by=20.0,
                                                 held="d4", us=near))
            d4 = next(i for i in decision["intents"] if i["playerId"] == "d4")
            shots += d4["action"]["type"] == "shoot"
        self.assertEqual(shots, 40, "8 m clear on goal must always be shot")

    def test_never_plays_the_ball_into_its_own_sixth(self):
        """Regression: the pass search would retreat to the corner or dump the
        ball in its own defensive sixth."""
        module = build_module({"family": "possession", "pass_power": 0.4})
        for seq in range(30):
            decision = module.decide(observation(seq=seq, bx=7.0, by=6.0, held="d2"))
            d2 = next(i for i in decision["intents"] if i["playerId"] == "d2")
            action = d2["action"]
            if action["type"] in ("pass", "clear"):
                target = action["target"]
                if action["type"] == "pass":
                    self.assertGreater(target["x"], 5.0,
                                       "passed backwards into our own goal area")

    def test_does_not_press_with_everyone(self):
        module = build_module({"family": "high_press", "press_intensity": 1.0})
        decision = module.decide(observation(seq=1, bx=12.0, by=20.0,
                                             team="them", held="d2"))
        tackling = [i for i in decision["intents"] if i["action"]["type"] == "tackle"]
        self.assertLessEqual(len(tackling), 4,
                             "all four outfielders committed to the press")

    def test_a_loose_ball_always_gets_one_chaser(self):
        """Regression: `_press_plan` returned an empty presser set whenever
        `they_have` was false, so the team assigned nobody to the ball.

        That looked like "pressing" and was not. The engine leaves
        `possessingTeam` unset for ~87% of a match (measured over 601 real
        replay frames), so for most of the game every outfielder stood in
        shape and watched the ball roll. In the arena the opponent parked all
        four outfielders on one x-coordinate and we walked the ball in for
        free -- the exact behaviour this was written to prevent.

        A chaser is only sent when the ball is within CONTEST_RADIUS of an
        outfielder (so we don't abandon shape for a ball in the far corner).
        """
        module = build_module({"family": "possession", "press_intensity": 0.2})
        for bx, by in ((30.0, 20.0), (24.0, 20.0), (18.0, 12.0), (22.0, 28.0)):
            decision = module.decide(observation(
                seq=int(bx), bx=bx, by=by, team=None, held=None))
            brain = module._BRAINS["g"]
            self.assertTrue(
                brain._pressers or _moves_toward(brain, bx, by),
                f"nobody assigned to a loose ball at ({bx}, {by})",
            )
            chasers = [i for i in decision["intents"]
                       if i["action"]["type"] in ("tackle", "slap")]
            self.assertLessEqual(len(chasers), 2,
                                 "a loose ball must not pull the whole team")

    def test_contest_plan_sends_one_man_and_respects_its_radius(self):
        """A loose ball is the *default* state, so the contest must be a
        single nearest man inside a tight radius. Sending two within 22 m
        (an earlier revision) meant two players permanently abandoning shape
        and the team got beaten 32-1."""
        module = build_module({"family": "high_press", "press_intensity": 0.9})
        brain = module._BRAINS["g"] if "g" in module._BRAINS else None
        module.decide(observation(seq=0, team=None, held=None))
        brain = module._BRAINS["g"]
        picked = brain._contest_plan(30.0, 20.0, 0.9)
        self.assertEqual(len(picked), 1, "expected exactly one ball-watcher")
        # A ball in their own corner is not worth leaving shape for.
        self.assertEqual(brain._contest_plan(1.5, 1.0, 0.9), set())

    def test_close_down_fires_even_when_the_press_trigger_has_not(self):
        """A defender stood right in front of a carrier used to be ignored
        unless the deep trigger also fired."""
        them = [
            {"id": "gk", "role": "goalkeeper", "position": {"x": 57.0, "y": 20.0},
             "velocity": {"x": 0.0, "y": 0.0}, "facingRadians": 0.0,
             "canAct": True},
            # Carrier at x=45, deep in the opponent's half. Use a press_trigger
            # of 0.7 (42m) so dangerous is False (holder.x >= 42 and bx >= 18).
            # Place our d4 5 m from the carrier -> should close down.
            {"id": "d1", "role": "outfield", "position": {"x": 45.0, "y": 20.0},
             "velocity": {"x": 0.0, "y": 0.0}, "facingRadians": 0.0,
             "canAct": True},
            {"id": "d2", "role": "outfield", "position": {"x": 40.0, "y": 8.0},
             "velocity": {"x": 0.0, "y": 0.0}, "facingRadians": 0.0,
             "canAct": True},
            {"id": "d3", "role": "outfield", "position": {"x": 40.0, "y": 32.0},
             "velocity": {"x": 0.0, "y": 0.0}, "facingRadians": 0.0,
             "canAct": True},
            {"id": "d4", "role": "outfield", "position": {"x": 40.0, "y": 20.0},
             "velocity": {"x": 0.0, "y": 0.0}, "facingRadians": 0.0,
             "canAct": True},
        ]
        us = [
            {"id": "gk", "role": "goalkeeper", "position": {"x": 3.0, "y": 20.0},
             "velocity": {"x": 0.0, "y": 0.0}, "facingRadians": 0.0,
             "canAct": True},
            # d1 far away
            {"id": "d1", "role": "outfield", "position": {"x": 26.0, "y": 20.0},
             "velocity": {"x": 0.0, "y": 0.0}, "facingRadians": 0.0,
             "canAct": True},
            # d2, d3 far
            {"id": "d2", "role": "outfield", "position": {"x": 22.0, "y": 13.0},
             "velocity": {"x": 0.0, "y": 0.0}, "facingRadians": 0.0,
             "canAct": True},
            {"id": "d3", "role": "outfield", "position": {"x": 22.0, "y": 27.0},
             "velocity": {"x": 0.0, "y": 0.0}, "facingRadians": 0.0,
             "canAct": True},
            # d4 is 5 m from carrier at (45,20) -> within CLOSE_DOWN_RANGE
            {"id": "d4", "role": "outfield", "position": {"x": 40.0, "y": 20.0},
             "velocity": {"x": 0.0, "y": 0.0}, "facingRadians": 0.0,
             "canAct": True},
        ]
        module = build_module({"family": "possession", "press_intensity": 0.0,
                               "press_trigger": 0.7})
        module.decide(observation(seq=0, bx=45.0, by=20.0, team="them",
                                  held="d1", us=us, them=them))
        brain = module._BRAINS["g"]
        self.assertIn("d4", brain._pressers,
                      "the man goal-side of the carrier was not told to close")

    def test_marking_actually_selects_a_defender(self):
        """Regression: `self._marks.get(q.pid) == q.pid` could never be true,
        because `_marks` is keyed by *opponent* ids. Marking therefore never
        ran for any team at any difficulty.

        The fix uses `_mark_for` keyed by *our* defender id, which cannot
        collide when both sides share id strings (offline sim mirrors the
        home ids). This test verifies the reverse map assigns each defender
        a distinct opponent threat."""
        module = build_module({"family": "possession", "compactness": 1.0})
        module.decide(observation(seq=0, bx=12.0, by=20.0, team="them",
                                  held="d2"))
        brain = module._BRAINS["g"]
        mark_for = getattr(brain, "_mark_for", None)
        self.assertIsInstance(mark_for, dict)
        self.assertGreater(len(mark_for), 0, "no marking assignments created")
        # Each defender appears at most once.
        self.assertEqual(len(mark_for), len(set(mark_for.keys())))
        # Each assigned threat is a real opponent player object.
        threat_ids = set()
        for defender_id, threat in mark_for.items():
            self.assertIn(defender_id, {"d1", "d2", "d3", "d4"})
            self.assertIn(threat.pid, {"d1", "d2", "d3", "d4"})
            threat_ids.add(threat.pid)
        # Threats are distinct (no two defenders marking the same man).
        self.assertEqual(len(threat_ids), len(mark_for))

    def test_goalkeeper_avoids_passing_into_an_opponent(self):
        """Regression: the old `_pressure_at` returned *space* (0 = crowded,
        1 = free) while the caller treated it as pressure, so distribution
        skipped every open option and chose the most crowded one -- which is
        how the keeper kept "hitting the blue player"."""
        module = build_module({"family": "possession"})
        module.decide(observation(seq=0))
        brain = module._BRAINS["g"]
        # Their d1 stands at (40, 18), so that point is crowded and a point
        # out by the near corner is not.
        self.assertLess(brain._space_at(40.0, 18.0), 0.2)
        self.assertGreater(brain._space_at(5.0, 1.0), 0.8)
        self.assertGreater(brain._space_at(5.0, 1.0),
                           brain._space_at(40.0, 18.0),
                           "space() must reward distance from the nearest foe")
        # The goalkeeper may not be counted as an interceptor, but outfield
        # pressure still registers.
        self.assertLess(brain._space_outfield_at(40.0, 18.0), 0.2)

    def test_space_helpers_are_not_inverted(self):
        """`space` up must mean *more* room. The feature vector reads these
        directly, so an inversion would train every learned team backwards."""
        module = build_module({"family": "possession"})
        module.decide(observation(seq=0))
        brain = module._BRAINS["g"]
        features = brain._features()
        # Index 4 = free = space, index 5 = press = 1 - space
        space = features[4]
        press = features[5]
        self.assertGreaterEqual(space, 0.0)
        self.assertLessEqual(space, 1.0)
        self.assertAlmostEqual(press, 1.0 - space, places=6)
        self.assertAlmostEqual(space, space, places=6)  # free == space

    def test_state_is_keyed_by_game_id(self):
        """Regression risk: one global brain would leak match state."""
        module = build_module({"family": "possession"})
        module.decide(observation(seq=0, game="a", bx=5.0))
        module.decide(observation(seq=1, game="b", bx=50.0))
        self.assertIn("a", module._BRAINS)
        self.assertIn("b", module._BRAINS)

    def test_deterministic_for_a_fixed_state(self):
        first = build_module({"family": "chaos", "noise": 0.4})
        second = build_module({"family": "chaos", "noise": 0.4})
        for seq in range(20):
            a = first.decide(observation(seq=seq, bx=20.0 + seq))
            b = second.decide(observation(seq=seq, bx=20.0 + seq))
            self.assertEqual(a, b, f"diverged at sequence {seq}")

    def test_decides_within_the_time_budget(self):
        import time
        module = build_module({"family": "overload"})
        start = time.monotonic()
        for seq in range(50):
            module.decide(observation(seq=seq, bx=15.0 + seq * 0.5))
        per_decision_ms = (time.monotonic() - start) / 50 * 1000.0
        self.assertLess(per_decision_ms, 50.0,
                        f"{per_decision_ms:.1f} ms per decision exceeds the 50 ms budget")


class SweepReportTests(unittest.TestCase):
    """`run_authoritative.py` exists to rank opponents, so its metric extraction
    is load-bearing. These are verbatim lines from a real `simulate` run."""

    SAMPLE = """[candidate 1/6] 1-0 win seed=simulation-0001 side=home
[candidate 2/6] 1-0 win seed=simulation-0002 side=home
[candidate 3/6] 1-0 win seed=simulation-0003 side=home
[candidate 4/6] 0-1 loss seed=simulation-0004 side=home
[candidate 5/6] 0-1 loss seed=simulation-0005 side=home
[candidate 6/6] 0-1 loss seed=simulation-0006 side=home
candidate (football-team-deleri-fc:dev)
Matches: 6/6 completed  W/D/L: 3/0/3  Win rate: 50.0 %
Goals: 3-3  Average: 0.5-0.5  Difference: 0
Clean sheets: 50.0 %  Possession: 58.9%  Shots: 18  Missed decisions: 0
"""

    def test_parses_the_discriminating_metric(self):
        from training.scripts.run_authoritative import RESULT_RE

        found = RESULT_RE.search(self.SAMPLE)
        self.assertIsNotNone(found)
        self.assertEqual(found.group(1), "3-3")
        scored, conceded = (float(v) for v in found.group(2).split("-"))
        self.assertEqual((scored, conceded), (0.5, 0.5))

    def test_detects_the_degenerate_scoreline_distribution(self):
        """Every 60 s match against a solid side is 1-0 or 0-1. A 6-game run
        therefore measures nothing, which is why the runner defaults to 24."""
        from training.scripts.run_authoritative import GAME_RE

        scorelines = [f"{a}-{b}" for a, b, _ in GAME_RE.findall(self.SAMPLE)]
        self.assertEqual(len(scorelines), 6)
        self.assertEqual(set(scorelines), {"1-0", "0-1"})
        self.assertNotIn("0-0", scorelines, "a draw would be a real signal")

    def test_a_short_run_cannot_rank_and_six_is_not_enough(self):
        from training.scripts.run_authoritative import GAME_RE

        scorelines = [f"{a}-{b}" for a, b, _ in GAME_RE.findall(self.SAMPLE)]
        self.assertEqual(len(scorelines), 6)
        # Documented reason the default is 24, not 2: this is the whole spread a
        # 6-game run can produce regardless of which opponent it faces.
        self.assertEqual(len(set(scorelines)), 2)


class MirrorTests(unittest.TestCase):
    def test_mirror_flips_the_pitch_and_the_sides(self):
        obs = observation(team="us", held="d2")
        flipped = mirror(obs)
        self.assertEqual(flipped["ball"]["position"]["x"], 60.0 - obs["ball"]["position"]["x"])
        self.assertEqual(flipped["ball"]["possessingTeam"], "them")
        self.assertEqual([q["id"] for q in flipped["us"]], [q["id"] for q in obs["them"]])


class OfflinePlayabilityTests(unittest.TestCase):
    """Every generated parameter set must actually play a full match."""

    def test_generated_params_play(self):
        from training.scripts.build_opponent_teams import build_targets

        for team_id, _name, params in build_targets()[:6]:
            with self.subTest(opponent=team_id):
                name = f"test-{team_id}"
                sim.OPPONENTS[name] = make_controller(build_module(params))
                result = sim.play_match(None, name, rng=random.Random(1), decisions=120)
                self.assertEqual(result.goals, result.score_us)


class GeneratedTeamTests(unittest.TestCase):
    REPO = ROOT.parent.parent

    def test_generated_teams_are_up_to_date(self):
        """A stale generated file is a mystery; make it a test failure."""
        proc = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "build_opponent_teams.py"), "--check"],
            capture_output=True, text=True, cwd=str(ROOT),
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_every_generated_team_is_self_contained(self):
        opponents = self.REPO / "opponents"
        if not opponents.is_dir():
            self.skipTest("opponents/ not generated")
        for directory in sorted(opponents.iterdir()):
            if not directory.is_dir():
                continue
            with self.subTest(team=directory.name):
                for required in ("football-team.json", "Dockerfile", "models.py",
                                 "server.py", "strategy.py"):
                    self.assertTrue((directory / required).exists(), required)
                manifest = json.loads((directory / "football-team.json").read_text())
                self.assertEqual(manifest["id"], directory.name)
                self.assertEqual(manifest["language"], "python")
                # Only these four files may be copied: the container is bare.
                dockerfile = (directory / "Dockerfile").read_text()
                self.assertIn("COPY models.py strategy.py server.py ./", dockerfile)
                self.assertNotIn("COPY src", dockerfile)

    def test_every_generated_strategy_imports_and_answers(self):
        """Regression: the six learned teams were shipped with JSON `true`
        literals inlined into the parameter block. That is byte-stable, so the
        drift check blessed them, but they raised NameError on import, never
        bound port 8080, and the engine reported "did not become healthy within
        15 seconds" for every seed and both sides.

        Byte-comparison cannot catch it; only executing the module can.
        """
        opponents = self.REPO / "opponents"
        if not opponents.is_dir():
            self.skipTest("opponents/ not generated")
        probe = ROOT / "scripts" / "_import_probe.py"
        proc = subprocess.run(
            [sys.executable, str(probe), str(opponents)],
            capture_output=True, text=True, cwd=str(ROOT),
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        checked = json.loads(proc.stdout)
        self.assertGreaterEqual(checked["ok"], 50, checked)
        self.assertEqual(checked["failures"], [])


if __name__ == "__main__":
    unittest.main()
