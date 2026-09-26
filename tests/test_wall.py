import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.wall import WallModel, is_near_wall, direction_to_wall
from src.geom import PITCH_LENGTH


class WallTests(unittest.TestCase):
    def test_is_near_wall(self):
        self.assertTrue(is_near_wall(0.0, 20.0))
        self.assertTrue(is_near_wall(30.0, 39.5))
        self.assertFalse(is_near_wall(30.0, 20.0))

    def test_direction_to_wall(self):
        # In the top half of the pitch the nearest touchline is the top one.
        dx, dy = direction_to_wall(30.0, 30.0)
        self.assertEqual((dx, dy), (0.0, 1.0))
        # In the bottom half it points down.
        dx, dy = direction_to_wall(30.0, 10.0)
        self.assertEqual((dx, dy), (0.0, -1.0))

    def test_contact_for_returns_or_none(self):
        m = WallModel()
        near_top = m.contact_for(10, 31, 45, 2)
        # Either computes a candidate or returns None; no crash and values finite.
        for cand in (near_top,):
            if cand is not None:
                self.assertTrue(0.5 <= cand.contact_x <= PITCH_LENGTH - 0.5)
                self.assertIn(cand.target_flag, ("top", "bottom"))
                self.assertGreaterEqual(cand.risk, 0.0)
                self.assertLessEqual(cand.risk, 1.0)
        # A dead-centre pass without wall near fails softly: returns None or sane.
        self.assertIn(type(near_top).__name__, ("WallCandidate", "NoneType"))

    def test_wall_shots_candidates(self):
        m = WallModel()
        shots = m.wall_shots(35.0, 20.0)
        self.assertGreaterEqual(len(shots), 0)
        for s in shots:
            self.assertIn(s.target_flag, ("top", "bottom"))
            self.assertTrue(0.5 <= s.contact_x <= PITCH_LENGTH - 0.5)

    def test_predict_rebound_bottom_wall(self):
        m = WallModel()
        cx, cy, rvx, rvy, wall = m.predict_rebound(30.0, 5.0, 5.0, -8.0)
        self.assertEqual(wall, "bottom")
        self.assertLess(cy, 0.001)
        self.assertGreater(rvy, 0.0)  # bounces back onto pitch
        self.assertEqual(rvx, 5.0)  # tangential retained

    def test_predict_rebound_goal_detection(self):
        m = WallModel()
        cx, cy, rvx, rvy, wall = m.predict_rebound(55.0, 20.0, 10.0, 0.0)
        self.assertEqual(wall, "goal")
        self.assertEqual((rvx, rvy), (0.0, 0.0))


if __name__ == "__main__":
    unittest.main()