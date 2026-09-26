import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src import geom


class GeomTests(unittest.TestCase):
    def test_clamp_bounds(self):
        self.assertEqual(geom.clamp(-1.0, 0.0, 10.0), 0.0)
        self.assertEqual(geom.clamp(5.0, 0.0, 10.0), 5.0)
        self.assertEqual(geom.clamp(99.0, 0.0, 10.0), 10.0)

    def test_clamp_point_inside_pitch(self):
        x, y = geom.clamp_point(61.0, 41.0)
        self.assertLessEqual(x, geom.PITCH_LENGTH)
        self.assertLessEqual(y, geom.PITCH_WIDTH)

    def test_distance(self):
        self.assertAlmostEqual(geom.distance(0, 0, 3, 4), 5.0)
        self.assertAlmostEqual(geom.distance_sq(0, 0, 3, 4), 25.0)

    def test_angle_to_and_facing_diff(self):
        self.assertAlmostEqual(geom.angle_to(0, 0, 1, 0), 0.0)
        diff = geom.facing_diff(0.0, 1.0, 0.0, 0.0, 0.0)
        self.assertAlmostEqual(diff, 0.0)

    def test_wall_bounce_tangential_kept_normal_75(self):
        # Bottom wall: vy<0 normal, vx tangential stays.
        rvx, rvy = geom.wall_bounce(2.0, -4.0, "bottom")
        self.assertAlmostEqual(rvx, 2.0)
        self.assertAlmostEqual(rvy, 0.75 * 4.0)
        # Top wall.
        rvx, rvy = geom.wall_bounce(-1.0, 5.0, "top")
        self.assertAlmostEqual(rvx, -1.0)
        self.assertAlmostEqual(rvy, -0.75 * 5.0)

    def test_wall_bounce_goal_lines(self):
        rvx, rvy = geom.wall_bounce(3.0, 2.0, "right")
        self.assertAlmostEqual(rvx, -0.75 * 3.0)
        self.assertAlmostEqual(rvy, 2.0)

    def test_lerp(self):
        self.assertAlmostEqual(geom.lerp(0.0, 10.0, 0.5), 5.0)

    def test_iter_points_in(self):
        pts = list(geom.iter_points_in(0.0, 4.0))
        self.assertTrue(all(0.0 <= p <= 4.0 for p in pts))
        self.assertEqual(len(pts), len(set(round(p, 6) for p in pts)))


if __name__ == "__main__":
    unittest.main()