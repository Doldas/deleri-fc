"""Permanent regression tests for team coordination: distinct movement targets.

`docs/TEAM_COORDINATION_AUDIT.md` measured that off-ball players were routinely
told to occupy the *same* point: 86.7% of their-possession snapshots had two or
more off-ball players sharing a target. The audit attributed the defensive
majority of that to ``_cover_point`` ignoring its own ``p`` argument, but the
same shape lived in three other branches:

- ``_cover_point`` sent every cover to one ball-to-goal point;
- ``_block_wall_lanes`` / ``_intercept_wall_pass`` sent every outfielder on a
  flank to one wall-lane point;
- the deep danger mark sent both of the two deepest defenders to the single
  most-advanced attacker.

These tests pin each branch, plus the seeded sweep that catches the next one.
The pass-veto boundary is pinned in ``test_receiver_identity`` (a long wall pass
must survive the ``MIN_PASS_TRAVEL`` veto) and the mirror invariant in
``test_mirror_symmetry``.
"""

from __future__ import annotations

import math
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import RuntimeConfig, default_genome
from src.geom import PITCH_WIDTH
from src.policy import PolicyController, PolicyInput
from src.state import Ball, GameState, Player, WorldModel
from src.tactics import PressPlan, assign_roles, detect, plan_press
from tests.symmetry import build_random_state, mirror_state
from tests.test_policy import SLOTS

CFG = RuntimeConfig(genome=default_genome())
SWEEP_STATES = 200
SWEEP_SEED = 20260930
DUP_TOL = 0.5


def _player(pid, team, role, x, y, *, vx=0.0, vy=0.0, facing=0.0):
    return Player(
        id=pid, team=team, role=role, x=x, y=y, vx=vx, vy=vy, facing=facing,
        can_act=True,
    )


def _keeper(team, pid, x, y):
    return _player(pid, team, "goalkeeper", x, y)


def _state(ball, us, them):
    return GameState(
        protocol_version="1.0", game_id="coordination", sequence=1,
        simulation_tick=0, apply_at_tick=0, time_remaining=90.0,
        phase="openPlay", score_us=0, score_them=0, ball=ball, us=us, them=them,
    )


def _input(st: GameState, plan: PressPlan | None = None) -> PolicyInput:
    world = WorldModel.build(st)
    roles = assign_roles(st, SLOTS)
    return PolicyInput(
        state=st,
        world=world,
        config=CFG,
        tactical_state=detect(st, world, CFG.genome, False),
        press_plan=plan if plan is not None else plan_press(st, world, CFG.genome, roles),
        roles=roles,
        time_remaining=st.time_remaining,
        match_duration=120.0,
    )


def _pairwise_min_separation(points):
    return min(
        math.hypot(a[0] - b[0], a[1] - b[1])
        for i, a in enumerate(points)
        for b in points[i + 1:]
    )


class CoverGeometryTests(unittest.TestCase):
    """``_cover_point`` must give each cover player its own station.

    The audit measured that every covering player received one identical
    ball-to-our-goal interpolation point. A lone cover must keep that original
    point exactly; each additional cover slides back down the ball-to-goal line
    and out to its own side.
    """

    def _defending_state(self):
        # Our four outfielders on distinct lines; the ball with their carrier.
        return _state(
            Ball(40.0, 24.0, 0.0, 0.0, "them", "t1"),
            (
                _keeper("us", "us0", 2.0, 20.0),
                _player("us1", "us", "outfield", 24.0, 4.0),
                _player("us2", "us", "outfield", 30.0, 36.0),
                _player("us3", "us", "outfield", 36.0, 14.0),
                _player("us4", "us", "outfield", 42.0, 30.0),
            ),
            (_keeper("them", "tgk", 58.0, 20.0), _player("t1", "them", "outfield", 40.0, 24.0)),
        )

    @staticmethod
    def _legacy_point(ball):
        # geom.lerp(a, b, 0.45) == a + 0.45 * (b - a) == 0.55 * a + 0.45 * b
        tx = max(4.0, 0.55 * ball.x)
        cx = min(max(tx, 4.0), ball.x)
        cy = min(max(0.55 * ball.y + 0.45 * 20.0, 3.0), PITCH_WIDTH - 3.0)
        return (cx, cy)

    def test_a_lone_cover_keeps_the_single_cover_point(self):
        st = self._defending_state()
        plan = PressPlan(cover=["us1"])
        inp = _input(st, plan)
        controller = PolicyController()
        self.assertEqual(controller._cover_slot(inp, st.us[1]), 0)
        x, y = controller._cover_point(inp, st.us[1])
        lx, ly = self._legacy_point(st.ball)
        self.assertAlmostEqual(x, lx, places=9)
        self.assertAlmostEqual(y, ly, places=9)
    def test_two_covers_get_two_distinct_stations(self):
        st = self._defending_state()
        plan = PressPlan(cover=["us1", "us2"])
        inp = _input(st, plan)
        controller = PolicyController()
        pts = [
            controller._cover_point(inp, p)
            for p in st.outfield_us()
            if p.id in plan.cover
        ]
        self.assertEqual(len(pts), 2)
        self.assertGreater(_pairwise_min_separation(pts), 1.0)

    def test_three_covers_are_pairwise_distinct(self):
        st = self._defending_state()
        plan = PressPlan(cover=["us1", "us2", "us3"])
        inp = _input(st, plan)
        controller = PolicyController()
        pts = [
            controller._cover_point(inp, p)
            for p in st.outfield_us()
            if p.id in plan.cover
        ]
        self.assertEqual(len(pts), 3)
        self.assertGreater(_pairwise_min_separation(pts), 1.0)

    def test_cover_stations_drop_progressively_back_towards_our_goal(self):
        st = self._defending_state()
        plan = PressPlan(cover=["us1", "us2", "us3"])
        inp = _input(st, plan)
        controller = PolicyController()
        stations = sorted(
            (controller._cover_slot(inp, p), controller._cover_point(inp, p)[0])
            for p in st.outfield_us()
            if p.id in plan.cover
        )
        xs = [x for _, x in stations]
        self.assertEqual(
            xs, sorted(xs, reverse=True),
            "cover depth is not monotone back towards our own goal",
        )

    def test_cover_points_mirror(self):
        st = self._defending_state()
        plan = PressPlan(cover=["us1", "us2", "us3"])
        a_inp = _input(st, plan)
        b_inp = _input(mirror_state(st), plan)
        a = {p.id: PolicyController()._cover_point(a_inp, p) for p in st.outfield_us()}
        stm = mirror_state(st)
        b = {p.id: PolicyController()._cover_point(b_inp, p) for p in stm.outfield_us()}
        for pid in ("us1", "us2", "us3"):
            self.assertAlmostEqual(a[pid][0], b[pid][0], places=9, msg=pid)
            self.assertAlmostEqual(a[pid][1] + b[pid][1], PITCH_WIDTH, places=9, msg=pid)


class WallLaneCoordinationTests(unittest.TestCase):
    """A wall lane is a single-defender job, held by the nearest flank player.

    Every outfielder on the flank used to run the same geometry, so three
    players stacked on one lane point. The responsible defender is the one
    closest to the carrier, chosen by an order that survives the mirror.
    """

    def _fixture(self):
        return _state(
            Ball(30.0, 2.0, 1.0, 0.0, "them", "t1"),
            (
                _keeper("us", "us0", 2.0, 20.0),
                _player("us1", "us", "outfield", 28.0, 4.0),
                _player("us2", "us", "outfield", 32.0, 10.0),
                _player("us3", "us", "outfield", 20.0, 30.0),
                _player("us4", "us", "outfield", 26.0, 34.0),
            ),
            (
                _keeper("them", "tgk", 58.0, 20.0),
                _player("t1", "them", "outfield", 30.0, 2.0),
                _player("t2", "them", "outfield", 35.0, 1.0),
                _player("t3", "them", "outfield", 40.0, 25.0),
                _player("t4", "them", "outfield", 45.0, 35.0),
            ),
        )

    def test_only_the_nearest_flank_defender_blocks_the_lane(self):
        st = self._fixture()
        inp = _input(st)
        controller = PolicyController()
        us1 = next(p for p in st.outfield_us() if p.id == "us1")
        us2 = next(p for p in st.outfield_us() if p.id == "us2")
        self.assertIsNotNone(controller._block_wall_lanes(inp, us1, st.ball))
        self.assertIsNone(controller._block_wall_lanes(inp, us2, st.ball))

    def test_only_the_nearest_flank_defender_intercepts(self):
        st = self._fixture()
        inp = _input(st)
        controller = PolicyController()
        us1 = next(p for p in st.outfield_us() if p.id == "us1")
        us2 = next(p for p in st.outfield_us() if p.id == "us2")
        self.assertIsNotNone(controller._intercept_wall_pass(inp, us1, st.ball))
        self.assertIsNone(controller._intercept_wall_pass(inp, us2, st.ball))

    def test_the_responsible_defender_mirrors_with_the_flank(self):
        st = self._fixture()
        stm = mirror_state(st)
        a_inp = _input(st)
        b_inp = _input(stm)
        controller = PolicyController()
        a = controller._block_wall_lanes(
            a_inp, next(p for p in st.outfield_us() if p.id == "us1"), st.ball
        )
        b = controller._block_wall_lanes(
            b_inp, next(p for p in stm.outfield_us() if p.id == "us1"), stm.ball
        )
        self.assertIsNotNone(a)
        self.assertIsNotNone(b)
        assert a is not None and b is not None
        self.assertAlmostEqual(a.tx, b.tx, places=9)
        self.assertAlmostEqual(a.ty + b.ty, PITCH_WIDTH, places=9)


class DeepDangerMarkTests(unittest.TestCase):
    """The two deepest defenders must split the two most-advanced attackers.

    The comment above the branch already said "two deepest teammates split the
    deepest two attackers goal-side", but both were ordered to the single most
    advanced attacker, so a deep transition produced one doubled-up marker and
    one unmarked forward.
    """

    def _fixture(self):
        return _state(
            Ball(52.0, 10.0, 0.0, 0.0, "them", "t4"),
            (
                _keeper("us", "us0", 2.0, 20.0),
                _player("us1", "us", "outfield", 15.0, 20.0),
                _player("us2", "us", "outfield", 18.0, 20.0),
                _player("us3", "us", "outfield", 30.0, 10.0),
                _player("us4", "us", "outfield", 25.0, 30.0),
            ),
            (
                _keeper("them", "tgk", 58.0, 20.0),
                _player("t1", "them", "outfield", 40.0, 24.0),
                _player("t2", "them", "outfield", 38.0, 16.0),
                _player("t3", "them", "outfield", 50.0, 30.0),
                _player("t4", "them", "outfield", 52.0, 10.0),
            ),
        )

    def _marks(self):
        st = self._fixture()
        inp = _input(st)
        controller = PolicyController()
        marks = {}
        for pid in ("us1", "us2"):
            p = next(q for q in st.outfield_us() if q.id == pid)
            intent = controller._off_ball_defend(inp, p, inp.press_plan.role_for(pid))
            marks[pid] = (intent.tx, intent.ty)
        return st, marks

    def test_the_two_deepest_defenders_mark_two_different_attackers(self):
        _, marks = self._marks()
        self.assertNotAlmostEqual(marks["us1"][1], marks["us2"][1], places=3)
        self.assertGreater(math.hypot(
            marks["us1"][0] - marks["us2"][0], marks["us1"][1] - marks["us2"][1]
        ), 1.0)

    def test_each_mark_sits_between_its_attacker_and_our_goal(self):
        st, marks = self._marks()
        # The deepest defender takes the most advanced attacker, the other the
        # next one, and both stay goal-side.
        self.assertLess(marks["us1"][0], 52.0)
        self.assertLess(marks["us2"][0], 50.0)

    def test_the_split_mirrors(self):
        st, marks = self._marks()
        stm = mirror_state(st)
        inp = _input(stm)
        controller = PolicyController()
        for pid, (tx, ty) in marks.items():
            p = next(q for q in stm.outfield_us() if q.id == pid)
            intent = controller._off_ball_defend(inp, p, inp.press_plan.role_for(pid))
            self.assertAlmostEqual(intent.tx, tx, places=9, msg=pid)
            self.assertAlmostEqual(intent.ty + ty, PITCH_WIDTH, places=9, msg=pid)


class SeededCoordinationSweepTests(unittest.TestCase):
    """No two COVER players share a target, over seeded random pictures.

    This is the guard that catches the next coordination branch the way the
    audit's sweep caught these ones. The loose-ball meeting point is the one
    intentional convergence in the policy, and its runners are excluded by
    checking only press-plan COVER assignments.
    """

    def _duplicate_covers(self, situation):
        rng = random.Random(SWEEP_SEED)
        bad = []
        for index in range(SWEEP_STATES):
            st = build_random_state(rng, situation)
            inp = _input(st)
            intents = PolicyController().decide(inp)
            covers = [
                p for p in st.outfield_us()
                if inp.press_plan.role_for(p.id) == "COVER" and p.id in intents
            ]
            for i, a in enumerate(covers):
                for b in covers[i + 1:]:
                    ia, ib = intents[a.id], intents[b.id]
                    if math.hypot(ia.tx - ib.tx, ia.ty - ib.ty) <= DUP_TOL:
                        bad.append(
                            f"{situation}#{index}: {a.id} and {b.id} share "
                            f"({ia.tx:.3f}, {ia.ty:.3f})"
                        )
        return bad

    def test_defending_cover_targets_do_not_collapse(self):
        bad = self._duplicate_covers("defending")
        self.assertEqual(bad, [], "\n".join(bad[:6]))

    def test_possession_cover_targets_do_not_collapse(self):
        bad = self._duplicate_covers("possession")
        self.assertEqual(bad, [], "\n".join(bad[:6]))

    def test_loose_cover_targets_do_not_collapse(self):
        bad = self._duplicate_covers("loose")
        self.assertEqual(bad, [], "\n".join(bad[:6]))


if __name__ == "__main__":
    unittest.main()
