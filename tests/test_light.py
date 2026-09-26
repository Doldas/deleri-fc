import math
import sys
import unittest
from pathlib import Path
from typing import Tuple, List

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.light import LightEngine, make_state, LIntent
from src.config import CONTROLLED_BALL_AHEAD, TACKLE_MAX, TACKLE_CLOSE
from src.geom import PITCH_LENGTH


Row = Tuple[str, str, float, float]


def team() -> List[Row]:
    return [
        ("gk", "goalkeeper", 5.0, 20.0),
        ("cd", "outfield", 20.0, 20.0),
        ("am", "outfield", 26.0, 16.0),
        ("w", "outfield", 24.0, 8.0),
        ("st", "outfield", 40.0, 24.0),
    ]


def mirror(rows: List[Row]) -> List[Row]:
    return [(pid, role, PITCH_LENGTH - x, y) for pid, role, x, y in rows]


class LightEngineTests(unittest.TestCase):
    def test_make_state_possessor_offset(self):
        st = make_state(team(), team(), ball=(30.0, 20.0), possess=("us", "am"))
        holder = st.player("us", "am")
        self.assertIsNotNone(holder)
        self.assertAlmostEqual(st.ball.x, holder.x + CONTROLLED_BALL_AHEAD)
        self.assertEqual(st.ball.possessing_team, "us")

    def test_ball_follows_holder(self):
        eng = LightEngine(seed=1)
        st = make_state(team(), team(), ball=(30.0, 20.0), possess=("us", "am"))
        moved = eng.step(st, {("us", "am"): LIntent(tx=40.0, ty=20.0, speed=1.0, act="none")})
        self.assertGreater(moved.ball.x, st.ball.x)

    def test_kick_sets_ball_free_and_fast(self):
        eng = LightEngine(seed=2)
        st = make_state(team(), team(), ball=(30.0, 20.0), possess=("us", "am"))
        target = (50.0, 20.0)
        moved = eng.step(
            st,
            {("us", "am"): LIntent(tx=35.0, ty=20.0, speed=0.5, act="pass", action_target=target, power=0.9)},
        )
        self.assertIsNone(moved.ball.possessing_team)
        self.assertGreater(moved.ball.vx, 0.0)

    def test_shot_toward_goal_eventually_ends_ball(self):
        eng = LightEngine(seed=3)
        st = make_state(team(), team(), ball=(58.0, 20.0), possess=("us", "st"))
        fired = False
        for _ in range(40):
            st = eng.step(
                st,
                {("us", "st"): LIntent(tx=59.0, ty=20.0, speed=0.5, act="shoot", action_target=(60.0, 20.0), power=1.0)},
            )
            if "goal" in st.events or st.score_us == 1:
                fired = True
                break
        self.assertTrue(fired)
        # After the goal the conceding team receives the kickoff at the centre
        # spot (RULES.md §127-131) — play continues, the ball is not frozen.
        self.assertIn("kickoff", st.events)
        self.assertAlmostEqual(st.ball.x, 30.0)

    def test_goal_restart_gives_conceding_team_kickoff(self):
        eng = LightEngine(seed=8)
        st = make_state(team(), mirror(team()), ball=(58.0, 20.0), possess=("us", "st"))
        for _ in range(40):
            st = eng.step(
                st,
                {("us", "st"): LIntent(tx=59.0, ty=20.0, speed=0.5, act="shoot", action_target=(60.0, 20.0), power=1.0)},
            )
            if "goal" in st.events:
                break
        self.assertEqual(st.ball.possessing_team, "them")
        self.assertEqual(st.ball.possessing_player, "st")
        # Both squads are back on their formation spots.
        st_us = next(p for p in st.players if p.team == "us" and p.pid == "st")
        self.assertAlmostEqual(st_us.x, 40.0)

    def test_gk_auto_distribution_after_hold(self):
        eng = LightEngine(seed=9)
        st = make_state(team(), mirror(team()), ball=(30.0, 20.0), possess=("us", "gk"))
        distributed = False
        for _ in range(30):
            st = eng.step(
                st,
                {("us", "gk"): LIntent(tx=8.0, ty=20.0, speed=0.0, act="none")},
            )
            if "gk:distribution" in st.events:
                distributed = True
                break
        self.assertTrue(distributed)
        self.assertIsNone(st.ball.possessing_team)

    def test_tackle_close_strips_ball(self):
        eng = LightEngine(seed=4)
        st = make_state(team(), team(), ball=(30.0, 20.0), possess=("us", "am"))
        st.ball.protection = 0.0
        attacker = next(p for p in st.players if p.team == "them" and p.role == "outfield")
        holder = st.player("us", "am")
        self.assertIsNotNone(holder)
        attacker.x = holder.x - TACKLE_CLOSE * 0.5
        attacker.y = st.ball.y
        moved = eng.step(
            st,
            {("them", attacker.pid): LIntent(tx=attacker.x, ty=attacker.y, speed=0.0, act="tackle")},
        )
        attacker2 = moved.player("them", attacker.pid)
        self.assertIsNotNone(attacker2)
        self.assertEqual(moved.ball.possessing_team, "them")
        self.assertIn("tackle:win", moved.events)

    def test_tackle_slide_grounds_attacker(self):
        eng = LightEngine(seed=5)
        st = make_state(team(), team(), ball=(30.0, 20.0), possess=("us", "am"))
        st.ball.protection = 0.0
        attacker = next(p for p in st.players if p.team == "them" and p.role == "outfield")
        holder = st.player("us", "am")
        self.assertIsNotNone(holder)
        attacker.x = holder.x + (TACKLE_MAX - TACKLE_CLOSE) / 2.0 + TACKLE_CLOSE
        attacker.y = holder.y
        moved = eng.step(
            st,
            {("them", attacker.pid): LIntent(tx=attacker.x, ty=attacker.y, speed=0.0, act="tackle")},
        )
        self.assertIsNone(moved.ball.possessing_team)
        self.assertIn("tackle:slide", moved.events)
        attacker2 = moved.player("them", attacker.pid)
        self.assertIsNotNone(attacker2)
        self.assertGreater(attacker2.grounded, 0.0)

    def test_tackle_out_of_range_no_effect(self):
        eng = LightEngine(seed=7)
        st = make_state(team(), team(), ball=(30.0, 20.0), possess=("us", "am"))
        st.ball.protection = 0.0
        attacker = next(p for p in st.players if p.team == "them" and p.role == "outfield")
        holder = st.player("us", "am")
        self.assertIsNotNone(holder)
        attacker.x = holder.x + TACKLE_MAX + 0.5
        attacker.y = holder.y
        moved = eng.step(
            st,
            {("them", attacker.pid): LIntent(tx=attacker.x, ty=attacker.y, speed=0.0, act="tackle")},
        )
        self.assertEqual(moved.ball.possessing_team, "us")
        self.assertNotIn("tackle:win", moved.events)

    def test_slap_knockdown_when_facing(self):
        eng = LightEngine(seed=6)
        st = make_state(team(), team(), ball=(30.0, 20.0), possess=("us", "am"))
        slapper = next(p for p in st.players if p.team == "them" and p.role == "outfield")
        holder = st.player("us", "am")
        self.assertIsNotNone(holder)
        # The move target sets facing: aim straight at the victim so the slap
        # counts as clean (within 60 deg of facing, within SLAP_CLEAN 0.75).
        slapper.x = holder.x - 0.6
        slapper.y = holder.y
        slapper.facing = math.pi
        moved = eng.step(
            st,
            {("them", slapper.pid): LIntent(tx=holder.x, ty=holder.y, speed=0.0, act="slap")},
        )
        victim = moved.player("us", holder.pid)
        self.assertIsNotNone(victim)
        self.assertGreater(victim.grounded, 0.0)
        self.assertIn("slap:down", moved.events)


if __name__ == "__main__":
    unittest.main()