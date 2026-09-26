import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.space import SpaceModel, influence, control_at


class SpaceTests(unittest.TestCase):
    def test_influence_symmetric_and_bounded(self):
        a = influence(0.0, 0.0, 1.0, 0.0)
        b = influence(1.0, 0.0, 0.0, 0.0)
        self.assertAlmostEqual(a, b)
        self.assertGreater(a, 0.0)

    def test_control_at_single_side(self):
        c = control_at(10.0, 10.0, [(12.0, 10.0)], [(30.0, 30.0)])
        self.assertGreater(c, 0.0)

    def test_best_forward_point_ahead(self):
        m = SpaceModel()
        m.update(ours=[(12.0, 10.0), (20.0, 20.0)], theirs=[(40.0, 40.0)])
        bx, by, ctrl = m.best_forward_point(12.0, 10.0)
        self.assertGreaterEqual(bx, 15.0)
        self.assertTrue(-1e-9 <= by <= 40.0 + 1e-9)
        self.assertGreaterEqual(ctrl, -1.0)

    def test_nearest_our_control_no_crash(self):
        m = SpaceModel()
        m.update(ours=[], theirs=[(5.0, 5.0)])
        x, y = m.nearest_our_control(30.0, 20.0)
        self.assertTrue(0.0 <= x <= 60.0)
        self.assertTrue(0.0 <= y <= 40.0)

    def test_opposition_weakness_zone(self):
        m = SpaceModel()
        m.update(ours=[(40.0, 20.0), (45.0, 10.0)], theirs=[(30.0, 10.0)])
        zone = m.opposition_weakness_zone()
        if zone is not None:
            self.assertGreaterEqual(zone[0], 21.0)
            self.assertLessEqual(zone[0], 60.0)


if __name__ == "__main__":
    unittest.main()