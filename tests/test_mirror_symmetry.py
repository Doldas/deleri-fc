"""Permanent regression tests for lateral (centre-line) mirror symmetry.

Every bug these guard was the same shape: a decision that read the pitch's
lateral axis through a *one-sided* test, so the team behaved differently in a
picture and in that picture reflected across ``y = 20``. The pitch is symmetric
about its centre line, so any deterministic decision must be equivariant under
that reflection: the same action, the same speed, and every lateral coordinate
and velocity with ``y`` replaced by ``40 - y``.

The tests are split in two. The first group pins one named defect each, so a
regression names itself. The second group sweeps seeded random situations and
compares the whole decision, which is the only way to catch the next one: every
individual fix here was found by that kind of sweep, not by reading the code.

Reflection is subtle in both directions, so the module also pins the cases that
are *already* correct and look wrong -- notably a one-sided comparison whose two
branch values happen to be mirror images of each other, which is legitimate and
must not be "fixed" into a bug.
"""

from __future__ import annotations

import dataclasses
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import RuntimeConfig, default_genome
from src.geom import (
    GOAL_CENTER_Y,
    GOAL_HIGH_Y,
    GOAL_LOW_Y,
    PITCH_LENGTH,
    PITCH_WIDTH,
    faces_toward_own_goal,
    wall_bounce,
)
from src.physics import (
    THRESHOLD_REL_EPS,
    at_or_above,
    at_or_below,
    pick_shot_target,
    wall_shot_target,
)
from src.policy import (
    PolicyController,
    PolicyInput,
    _detect_low_block,
    _pick_lateral_candidate,
    _touchline_side,
)
from src.state import Ball, GameState, Player, WorldModel
from src.tactics import assign_roles, detect, plan_press
from tests.symmetry import (
    assert_y_mirror,
    describe_mismatch,
    facing_is_mirrored,
    intent_is_mirrored,
    mirror_side,
    mirror_y,
    random_state_mismatches,
)
from tests.test_policy import SLOTS

# The sweep count is a compromise: enough pictures that the exact-tie cases are
# hit often, few enough that the whole module stays a fast unit test. The ignored
# `.lab/mirror_audit.py` harness runs the same idea at thousands of states.
SWEEP_STATES = 120
SWEEP_SEED = 20260930

CFG = RuntimeConfig(genome=default_genome())


def _player(pid, team, role, x, y, *, vx=0.0, vy=0.0, facing=0.0):
    return Player(
        id=pid,
        team=team,
        role=role,
        x=x,
        y=y,
        vx=vx,
        vy=vy,
        facing=facing,
        can_act=True,
    )


def _keeper(team: str, pid: str, x: float, y: float) -> Player:
    """A keeper facing the middle of the pitch."""
    return _player(pid, team, "goalkeeper", x, y, facing=0.0 if team == "us" else math.pi)


def _state(ball: Ball, us: tuple, them: tuple) -> GameState:
    return GameState(
        protocol_version="1.0",
        game_id="mirror-unit",
        sequence=7,
        simulation_tick=0,
        apply_at_tick=0,
        time_remaining=90.0,
        phase="openPlay",
        score_us=0,
        score_them=0,
        ball=ball,
        us=us,
        them=them,
    )


def make_input(st: GameState) -> PolicyInput:
    """Build a PolicyInput the way the runtime does, including the press plan.

    `detect` and `plan_press` are part of the decision, so a sweep that stubs
    them out would miss every trigger and pressing asymmetry.
    """
    world = WorldModel.build(st)
    roles = assign_roles(st, SLOTS)
    tstate = detect(st, world, CFG.genome, False)
    plan = plan_press(st, world, CFG.genome, roles)
    return PolicyInput(
        state=st,
        world=world,
        config=CFG,
        tactical_state=tstate,
        press_plan=plan,
        roles=roles,
        time_remaining=st.time_remaining,
        match_duration=120.0,
    )


# --------------------------------------------------------------------------- #
# geometry primitives
# --------------------------------------------------------------------------- #


class FacingTests(unittest.TestCase):
    """`faces_toward_own_goal` decides whether a carrier can be dispossessed from
    the front, and it is evaluated in triggers and in pressing. It has to wrap
    safely: a heading in [-pi, pi] names the same physical direction twice.
    """

    def test_facing_toward_own_goal_over_the_whole_circle(self):
        # The goal is at x = 0. Facing -pi and +pi both look straight at it, and
        # facing 0 looks away from it. A raw `facing < pi/2` test gets one of the
        # two "toward" cases wrong at the wrap-around.
        self.assertTrue(faces_toward_own_goal(30.0, 20.0, math.pi))
        self.assertTrue(faces_toward_own_goal(30.0, 20.0, -math.pi))
        self.assertFalse(faces_toward_own_goal(30.0, 20.0, 0.0))

    def test_facing_test_ignores_lateral_offset(self):
        # The reflection maps a heading f to -f, so the mirror has to negate it.
        # Holding the heading fixed while moving y would be a different physical
        # picture, not a reflection: heading down-left from y=32 faces the goal,
        # and the same heading from y=8 does not.
        for facing in (-3.0, -1.5, -0.2, 0.0, 0.2, 1.5, 3.0):
            low = faces_toward_own_goal(30.0, 8.0, facing)
            high = faces_toward_own_goal(30.0, 32.0, -facing)
            self.assertEqual(
                low,
                high,
                f"the reflection changed the facing test at heading {facing}",
            )

    def test_a_heading_mirrors_by_negation(self):
        # A player running towards the y=0 touchline has a heading that negates
        # under the reflection; the mirrored player runs towards y=40. Using
        # pi - heading instead would reflect a *vertical* line and point the
        # mirrored player back the way they came.
        for heading in (0.3, 1.1, 2.5, -0.7):
            self.assertTrue(facing_is_mirrored(heading, -heading))

    def test_wall_bounce_only_reverses_the_perpendicular_component(self):
        # Engine rule: tangential speed is retained and the perpendicular
        # component comes back at 75%.
        vx, vy = 3.0, 2.0
        self.assertEqual(wall_bounce(vx, vy, "top"), (vx, -0.75 * vy))
        self.assertEqual(wall_bounce(vx, vy, "bottom"), (vx, -0.75 * vy))
        self.assertEqual(wall_bounce(vx, vy, "left"), (-0.75 * vx, vy))
        self.assertEqual(wall_bounce(vx, vy, "right"), (-0.75 * vx, vy))


class WallAxisTests(unittest.TestCase):
    """Engine wall semantics: ``left``/``right`` are the two *goal lines* and
    ``top``/``bottom`` are the two *touchlines*.

    Conflating the two axes is how a defender sent to cut off a pass off the
    ``y = 0`` touchline ends up standing three metres from our own goal line.
    """

    def test_touchline_side_is_a_lateral_question_and_exchanges_under_mirror(self):
        for y, side in ((0.0, "bottom"), (1.0, "bottom"), (39.0, "top"), (40.0, "top")):
            self.assertEqual(_touchline_side(y), side)
            self.assertEqual(_touchline_side(mirror_y(y)), mirror_side(side))

    def test_no_touchline_is_named_in_the_middle(self):
        for y in (10.0, 20.0, 30.0):
            self.assertIsNone(_touchline_side(y))

    def test_a_wall_pass_uses_a_touchline_coordinate(self):
        # The wall a pass is played off sits at y = 0 or y = PITCH_WIDTH, never
        # at a goal-line x.
        self.assertEqual(0.0 if _touchline_side(1.0) == "bottom" else PITCH_WIDTH, 0.0)
        self.assertEqual(0.0 if _touchline_side(39.0) == "bottom" else PITCH_WIDTH, float(PITCH_WIDTH))


# --------------------------------------------------------------------------- #
# the named defects
# --------------------------------------------------------------------------- #


class ShotTargetTests(unittest.TestCase):
    """The far post has to come from the *sign* of the shooter's lateral offset.

    ``by >= GOAL_CENTER_Y`` looks like a correct "am I in the high half?" test and
    is not: the reflection turns it into ``by <= GOAL_CENTER_Y``, so "far" came
    out as the same physical post in a picture and in its reflection.
    """

    def test_far_post_flips_with_the_shooters_half(self):
        high = pick_shot_target(52.0, 24.0, 58.0, GOAL_CENTER_Y)
        low = pick_shot_target(52.0, 16.0, 58.0, GOAL_CENTER_Y)
        self.assertLess(high.y, GOAL_CENTER_Y, "a high shooter must aim low")
        self.assertGreater(low.y, GOAL_CENTER_Y, "a low shooter must aim high")
        assert_y_mirror(high.y, low.y)

    def test_a_mirrored_shooter_and_keeper_get_mirrored_targets(self):
        for by, gky in ((24.0, 22.0), (16.0, 18.0), (26.5, 19.0), (13.5, 21.0)):
            a = pick_shot_target(52.0, by, 58.0, gky)
            b = pick_shot_target(52.0, mirror_y(by), 58.0, mirror_y(gky))
            self.assertAlmostEqual(a.x, b.x)
            self.assertAlmostEqual(a.quality, b.quality)
            assert_y_mirror(a.y, b.y, msg=f"shooter y={by}, keeper y={gky}: ")

    def test_centre_line_shooter_is_its_own_mirror(self):
        # y = 20 is a fixed point of the reflection, so this state *equals* its
        # own mirror and any deterministic answer is already equivariant. Pinning
        # it stops a future tie-break from being invented here.
        for gky in (18.0, 20.0, 22.0):
            a = pick_shot_target(52.0, GOAL_CENTER_Y, 58.0, gky)
            self.assertEqual(a, pick_shot_target(52.0, GOAL_CENTER_Y, 58.0, gky))

    def test_an_exact_reach_tie_prefers_the_far_post(self):
        # A keeper on the centre line is equidistant from both posts, so reach
        # cannot choose between them. The far post is the football answer, and it
        # is the choice that mirrors with the shooter's half.
        self.assertLess(pick_shot_target(52.0, 24.0, 58.0, GOAL_CENTER_Y).y, GOAL_CENTER_Y)

    def test_targets_sit_inside_the_posts(self):
        t = pick_shot_target(52.0, 24.0, 58.0, 20.0)
        self.assertGreater(t.y, GOAL_LOW_Y)
        self.assertLess(t.y, GOAL_HIGH_Y)
        self.assertEqual(t.x, float(PITCH_LENGTH))


class WallShotTargetTests(unittest.TestCase):
    """The same far-post defect in the wall-bounce shot, plus a scan-order tie.

    The candidate contact heights are mirror-symmetric about the centre line and
    the keeper's distance to the chosen far post does not change under the
    reflection, so the common case scores an exact tie. A fixed scan order then
    sent the shot off the *same* contact height in a picture and its reflection.
    """

    def _target(self, by, gky):
        return wall_shot_target(50.0, by, 58.0, gky)

    def test_far_post_flips_with_the_shooters_half(self):
        assert_y_mirror(self._target(24.0, 20.0)[1], self._target(16.0, 20.0)[1],
                        msg="wall shot far post: ")

    def test_mirrored_shooters_get_mirrored_contacts(self):
        for by, gky in ((24.0, 21.0), (16.0, 19.0), (30.0, 20.0), (10.0, 20.0)):
            a = self._target(by, gky)
            b = self._target(mirror_y(by), mirror_y(gky))
            assert_y_mirror(a[1], b[1], msg=f"wall shot contact height, shooter y={by}: ")


class LateralCandidateTests(unittest.TestCase):
    """``_pick_lateral_candidate`` scores both flanks and averages exact ties.

    An exactly tied pair ``{low, high}`` mirrors to itself while each element
    mirrors to the other, so there is no equivariant way to *name* one of them.
    The mean is the one answer equal to its own reflection, and for a max-min
    score it is a genuine maximiser rather than a compromise.
    """

    def test_a_tie_resolves_to_the_midpoint(self):
        self.assertEqual(_pick_lateral_candidate([(4.0, 8.0), (4.0, 32.0)], lambda c: 0.0), (4.0, 20.0))

    def test_a_clear_winner_is_returned_unchanged(self):
        self.assertEqual(
            _pick_lateral_candidate([(4.0, 8.0), (4.0, 32.0)], lambda c: -c[1]),
            (4.0, 8.0),
        )

    def test_the_result_mirrors_for_a_mirrored_score(self):
        # Only mirror-invariant probes can pin equivariance: a score that reads
        # the raw y (`lambda c: c[1]`) is not equivariant in the first place, so
        # the mirrored list legitimately produces the mirrored answer. These are
        # the scores the callers actually pass.
        probes = (lambda c: abs(c[1] - 20.0), lambda c: 0.0, lambda c: c[0])
        for probe in probes:
            a = _pick_lateral_candidate([(4.0, 8.0), (4.0, 32.0)], probe)
            b = _pick_lateral_candidate(
                [(4.0, mirror_y(8.0)), (4.0, mirror_y(32.0))],
                lambda c, p=probe: p((c[0], mirror_y(c[1]))),
            )
            self.assertEqual(a, b, "the choice did not mirror")


class KeeperHoofTests(unittest.TestCase):
    """The goalkeeper's last-resort clearance.

    It was ``8.0 if ball.y >= 20.0 else 32.0``: a one-sided test of the ball's
    half, so the mirrored keeper saw the mirrored ball and cleared to the *same*
    flank. The comment above it already promised to try both flanks and keep the
    one the opponents are furthest from, which is what it does now.
    """

    def _hoof(self, ball_y, them_positions):
        # decide() short-circuits when we have no outfielders, so a keeper-only
        # side never reaches the hoof at all. Give us a full four.
        st = _state(
            Ball(3.0, ball_y, 0.0, 0.0, "us", "us0"),
            (
                _keeper("us", "us0", 2.0, GOAL_CENTER_Y),
                _player("us1", "us", "outfield", 12.0, 6.0),
                _player("us2", "us", "outfield", 10.0, 20.0),
                _player("us3", "us", "outfield", 12.0, 34.0),
                _player("us4", "us", "outfield", 20.0, 20.0),
            ),
            (_keeper("them", "tgk", 58.0, 20.0),) + tuple(
                _player(f"t{i}", "them", "outfield", x, y)
                for i, (x, y) in enumerate(them_positions, 1)
            ),
        )
        intent = PolicyController().decide(make_input(st))
        out = intent["us0"]
        self.assertEqual(out.action_type, "pass")
        if out.action_target is None:  # pragma: no cover - asserted above
            self.fail("the goalkeeper was told to pass without a target")
        tx, ty = out.action_target
        return tx, ty

    def test_the_hoof_goes_to_the_far_flank(self):
        crowd = [(30.0, 10.0), (34.0, 12.0), (28.0, 8.0)]
        ax, ay = self._hoof(20.0, crowd)
        bx, by = self._hoof(20.0, [(x, mirror_y(y)) for x, y in crowd])
        self.assertGreater(ay, GOAL_CENTER_Y)
        self.assertAlmostEqual(ax, bx)
        assert_y_mirror(ay, by, msg="goalkeeper hoof: ")

    def test_the_hoof_measures_the_block_not_the_balls_half(self):
        # A ball in the high half is not by itself a reason to clear high: with
        # the whole block in the high half, the low flank is the safe one.
        crowd = [(30.0, 30.0), (34.0, 32.0), (28.0, 28.0)]
        self.assertLess(self._hoof(30.0, crowd)[1], GOAL_CENTER_Y)

    def test_an_exactly_symmetric_block_hoofs_up_the_middle(self):
        crowd = [(30.0, 30.0), (34.0, 32.0), (28.0, 28.0), (32.0, 10.0), (36.0, 8.0), (26.0, 12.0)]
        self.assertAlmostEqual(self._hoof(20.0, crowd)[1], GOAL_CENTER_Y)


class TrackAssignmentSpeedTests(unittest.TestCase):
    """Shadowing an attacker who is *running* has to be a sprint either way.

    ``attacker.vy > 2.0`` only saw runs towards one touchline, and the reflection
    flips vy, so the defender jogging at a carrier sprinting at ``y = 0`` sprinted
    against the mirror image. The lateral term has to be a magnitude.
    """

    def _speed(self, vy):
        attacker = _player("t1", "them", "outfield", 30.0, 20.0, vy=vy)
        defender = _player("us1", "us", "outfield", 26.0, 20.0)
        st = _state(
            Ball(30.0, 20.0, 0.0, 0.0, "them", "t1"),
            (_keeper("us", "us0", 2.0, 20.0), defender),
            (_keeper("them", "tgk", 58.0, 20.0), attacker),
        )
        inp = make_input(st)
        return PolicyController()._track_assignment(inp, defender, attacker, st.ball, st).speed

    def test_a_run_towards_either_touchline_is_matched(self):
        self.assertEqual(self._speed(3.0), 1.0)
        self.assertEqual(self._speed(-3.0), 1.0)

    def test_a_stationary_attacker_is_walked(self):
        self.assertEqual(self._speed(0.0), 0.85)
        self.assertEqual(self._speed(1.0), 0.85)


class ThresholdTests(unittest.TestCase):
    """Exact-threshold comparisons are where mirrored floating point bites.

    ``MIN_PASS_TRAVEL`` is reached from both directions, and a mirrored
    computation can land a last bit either side of it, so ``>`` versus ``>=``
    silently accepted a different set of passes in a picture and its reflection.
    """

    def test_the_helpers_bracket_the_threshold(self):
        self.assertTrue(at_or_below(10.0, 10.0))
        self.assertTrue(at_or_below(9.999, 10.0))
        self.assertFalse(at_or_below(10.001, 10.0))
        self.assertTrue(at_or_above(10.0, 10.0))
        self.assertTrue(at_or_above(10.001, 10.0))
        self.assertFalse(at_or_above(9.999, 10.0))

    def test_the_tolerance_absorbs_only_last_bit_noise(self):
        self.assertTrue(at_or_below(10.0 - 1e-13, 10.0))
        self.assertFalse(at_or_above(10.0 - 1e-6, 10.0))

    def test_the_tolerance_band_is_symmetric_around_the_threshold(self):
        # An exact tie satisfies both inclusive comparisons by definition --
        # `10 <= 10` and `10 >= 10` -- and that is what makes the pair usable as
        # a band. Outside the band they must bracket the threshold strictly, and
        # the band must be the same width on both sides: a one-sided tolerance
        # would accept a value below the threshold but reject its mirror-image
        # partner above it.
        eps = abs(10.0) * THRESHOLD_REL_EPS
        for value in (10.0 - eps, 10.0 + eps):
            self.assertTrue(at_or_below(value, 10.0))
            self.assertTrue(at_or_above(value, 10.0))
        for value in (9.9, 9.999999, 10.000001, 10.1):
            below = at_or_below(value, 10.0)
            above = at_or_above(value, 10.0)
            self.assertNotEqual(
                below,
                above,
                f"{value} is outside the tolerance band but did not bracket the threshold",
            )

    def test_a_mirrored_pair_is_accepted_together(self):
        # The actual failure mode: two distances that are equal in exact
        # arithmetic straddle the threshold by a bit in floating point.
        for a, b in ((15.600000000000001, 15.599999999999998), (10.0, 10.0)):
            self.assertEqual(at_or_above(a, 10.0), at_or_above(b, 10.0))


class LowBlockSwitchTests(unittest.TestCase):
    """Equal flank counts have no weak flank.

    ``_detect_low_block`` used to answer "left" for a tie and
    ``_switch_play_choice`` acted on that answer, so a symmetric block was
    reported as weak on whichever side the code happened to test first and the
    attack switched play for no reason.
    """

    @staticmethod
    def _block(opponent_ys):
        them = (_keeper("them", "tgk", 58.0, 20.0),) + tuple(
            _player(f"t{i}", "them", "outfield", 40.0, y) for i, y in enumerate(opponent_ys, 1)
        )
        st = _state(
            Ball(30.0, 20.0, 0.0, 0.0, "us", "us1"),
            (_keeper("us", "us0", 2.0, 20.0), _player("us1", "us", "outfield", 30.0, 20.0)),
            them,
        )
        return st

    def test_equal_flank_counts_have_no_weak_flank(self):
        block = _detect_low_block(self._block((12.0, 16.0, 24.0, 28.0)))
        self.assertIsNone(block.get("weak_flank"))

    def test_a_genuine_weak_flank_is_still_reported(self):
        block = _detect_low_block(self._block((26.0, 28.0, 30.0, 32.0)))
        self.assertTrue(block.get("is_low_block"))
        self.assertIsNotNone(block.get("weak_flank"))

    def test_no_switch_is_chosen_when_the_weak_flank_is_unknown(self):
        # `_switch_play_choice` only consults the weak flank for a central
        # player, so drive it with one and let the flank test be the deciding
        # factor: with two defenders on each flank there is no weaker side to
        # switch to, and choosing one would mean naming a side the picture does
        # not distinguish.
        st = self._block((12.0, 16.0, 24.0, 28.0))
        st = dataclasses.replace(
            st,
            ball=Ball(44.0, 20.0, 0.0, 0.0, "us", "us1"),
            us=(
                _keeper("us", "us0", 2.0, 20.0),
                _player("us1", "us", "outfield", 44.0, 20.0),
                _player("us2", "us", "outfield", 30.0, 8.0),
                _player("us3", "us", "outfield", 30.0, 32.0),
            ),
        )
        inp = make_input(st)
        central = next(p for p in st.outfield_us() if inp.roles.get(p.id) not in ("WIDE_LEFT", "WIDE_RIGHT"))
        self.assertIsNone(PolicyController()._switch_play_choice(inp, central))


# --------------------------------------------------------------------------- #
# cases that are already right and look wrong
# --------------------------------------------------------------------------- #


class AlreadyCorrectTests(unittest.TestCase):
    """Pin the one-sided-looking comparisons that are genuinely equivariant.

    ``15.0 if p.y <= 20.0 else 25.0`` reads like the defect above and is not: the
    two branch values are mirror images of each other and the branch condition
    flips under the reflection in the matching way, so the two cancel. Rewriting
    it to ``abs(p.y - 20)``, or "fixing" it into a ``sign()`` call, would
    introduce the very bug the other tests exist to prevent.
    """

    def test_a_centre_line_split_is_equivariant(self):
        pick = lambda y: 15.0 if y <= 20.0 else 25.0
        # y = 20 is excluded on purpose: it is the fixed point of the
        # reflection, so it mirrors to itself and the sum is 2*pick(20) = 30 for
        # any deterministic pick. Equivariance is a statement about pairs of
        # distinct pictures; at the fixed point every answer is already its own
        # mirror.
        for y in (0.5, 5.0, 15.0, 19.999, 20.001, 25.0, 35.0, 39.5):
            self.assertAlmostEqual(
                pick(y) + pick(mirror_y(y)),
                PITCH_WIDTH,
                msg=f"the centre-line split broke at y={y}",
            )

    def test_far_post_branch_values_are_mirror_images(self):
        pick = lambda y: 17.0 if y >= 20.0 else 23.0
        for y in (1.0, 19.9, 20.1, 39.0):
            self.assertAlmostEqual(pick(y) + pick(mirror_y(y)), PITCH_WIDTH)


# --------------------------------------------------------------------------- #
# whole-decision sweep
# --------------------------------------------------------------------------- #


def _probe_all_intents(st: GameState, stm: GameState) -> bool:
    """Every player's full intent must mirror, action and speed included."""
    a = PolicyController().decide(make_input(st))
    b = PolicyController().decide(make_input(stm))
    if a.keys() != b.keys():
        return False
    for pid, ia in a.items():
        if not intent_is_mirrored(ia, b[pid]):
            raise AssertionError(f"{pid}: {describe_mismatch(ia, b[pid])}")
    return True


class WholeDecisionSweepTests(unittest.TestCase):
    """The invariant that catches the *next* bug, over seeded random pictures."""

    def _sweep(self, situation):
        bad = random_state_mismatches(situation, SWEEP_STATES, SWEEP_SEED, _probe_all_intents)
        self.assertEqual(
            bad,
            [],
            f"{situation}: {len(bad)} of {SWEEP_STATES} pictures did not mirror\n"
            + "\n".join(bad[:6]),
        )

    def test_possession_mirrors(self):
        self._sweep("possession")

    def test_defending_mirrors(self):
        self._sweep("defending")

    def test_loose_ball_mirrors(self):
        self._sweep("loose")

    def test_a_reported_failure_can_be_replayed(self):
        # The sweep is only actionable if a reported failure is reproducible, so
        # pin the generator: the same seed must rebuild the same state.
        import random

        from tests.symmetry import build_random_state

        a = build_random_state(random.Random(SWEEP_SEED), "defending")
        b = build_random_state(random.Random(SWEEP_SEED), "defending")
        self.assertEqual([(p.id, p.x, p.y, p.vy) for p in a.us],
                         [(p.id, p.x, p.y, p.vy) for p in b.us])
        self.assertEqual((a.ball.x, a.ball.y, a.ball.vy), (b.ball.x, b.ball.y, b.ball.vy))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
