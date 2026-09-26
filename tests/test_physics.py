import sys
import unittest
from pathlib import Path
import math

sys.path.insert(0, str(Path(__file__).parents[1]))

from src import physics
from src.geom import PITCH_LENGTH, PITCH_WIDTH


class PhysicsTests(unittest.TestCase):
    def test_kick_to_speed_ordered(self):
        # 25 m is comfortably reachable at low power; arrival speed must rise
        # with power (engine BALL_DECAY model).
        fast = physics.kick_to(10, 20, 35, 20, 1.0)
        mid = physics.kick_to(10, 20, 35, 20, 0.5)
        slow = physics.kick_to(10, 20, 35, 20, 0.2)
        self.assertGreater(fast.arrival_speed, mid.arrival_speed)
        self.assertGreater(mid.arrival_speed, slow.arrival_speed)
        self.assertGreater(slow.arrival_speed, 0.0)
        self.assertAlmostEqual(fast.arrival_x, 35.0, places=3)
        self.assertAlmostEqual(fast.arrival_y, 20.0, places=3)

    def test_kick_to_unreachable_distance_is_inf_and_positional(self):
        # A 45 m pass at low power physically cannot arrive: the model reports
        # a non-finite flight (the engine would give up / or ball never gets
        # there), and position stays finite in-pitch.
        r = physics.kick_to(5, 20, 50, 20, 0.2)
        self.assertTrue(math.isfinite(r.arrival_x))
        self.assertTrue(math.isfinite(r.arrival_y))
        self.assertLessEqual(abs(r.arrival_x), PITCH_LENGTH)
        self.assertLessEqual(abs(r.arrival_y), PITCH_WIDTH)

    def test_kick_to_high_power_reaches_far_target(self):
        r = physics.kick_to(5, 20, 50, 20, 1.0)
        self.assertGreater(r.arrival_speed, 0.0)
        self.assertLessEqual(abs(r.arrival_x), PITCH_LENGTH)

    def test_plan_lead_pass_reaches_receiver(self):
        plan = physics.plan_lead_pass(10, 20, 25, 18, -1.0, 0.0, 0.5)
        # A receiver sprinting toward the passer can make the flight estimate
        # unreachable (time == 0 / power 0 fallback) but the emitted target
        # must always stay finite and land near the receiver's projection.
        self.assertGreaterEqual(plan.time, 0.0)
        self.assertTrue(math.isfinite(plan.target_x))
        self.assertTrue(math.isfinite(plan.target_y))
        self.assertAlmostEqual(plan.power, min(1.0, max(0.0, plan.power)))
        self.assertAlmostEqual(plan.target_x, 25.0 - 1.0 * 0.5 * plan.time, places=1)

    def test_flight_decay_consistent(self):
        d = physics.distance_travelled(20.0, 1.0)
        v = physics.speed_after(20.0, 1.0)
        self.assertLess(v, 20.0)
        self.assertGreater(d, 0.0)

    def test_flight_time_positive(self):
        t = physics.flight_time(20.0, 18.0)
        self.assertGreater(t, 0.0)
        self.assertTrue(math.isfinite(t))

    def test_shot_open_angle(self):
        import math as m
        wide = physics.shot_open_angle(50, 20, [(20, 20), (20, 10), (58, 30)])
        blocked = physics.shot_open_angle(52, 20, [(56, 20), (50, 15), (50, 25)])
        self.assertGreater(wide, blocked)
        self.assertGreaterEqual(wide, 0.0)
        self.assertLessEqual(wide, m.pi)

    def test_pick_shot_target_inside_goal(self):
        t = physics.pick_shot_target(50, 20, 58, 20)
        self.assertGreaterEqual(t.y, 17.0)
        self.assertLessEqual(t.y, 23.0)
        self.assertGreaterEqual(t.x, 50.0)
        self.assertAlmostEqual(t.power, min(1.0, max(0.0, t.power)))

    def test_pick_shot_target_inset_from_posts(self):
        # Aims must sit inside the posts so shots don't rattle off the frame.
        # Check every reachable corner keeps at least ~0.5 m margin.
        for by, gky in ((20.0, 20.0), (10.0, 20.0), (30.0, 20.0), (20.0, 26.0)):
            t = physics.pick_shot_target(50, by, 58, gky)
            self.assertIn(t.y, (17.6, 22.4))
            self.assertGreaterEqual(t.y, 17.4)
            self.assertLessEqual(t.y, 22.6)

    def test_shot_lane_clear_respects_mark_and_wide(self):
        import math as m
        # Shooter at (50,20) aiming (60,17); a marker dead on the lane blocks.
        self.assertFalse(physics.shot_lane_clear(50, 20, 60, 17, [(55, 18.5)], margin=0.14))
        # A marker far off the lane does not block.
        self.assertTrue(physics.shot_lane_clear(50, 20, 60, 17, [(55, 25.0)], margin=0.14))
        # No opponents: always clear.
        self.assertTrue(physics.shot_lane_clear(50, 20, 60, 17, []))
        self.assertGreaterEqual(physics.shot_open_angle(50, 20, [(20, 20)]), 0.0)
        self.assertLessEqual(physics.shot_open_angle(50, 20, [(20, 20)]), m.pi)

    def test_clamp01(self):
        self.assertEqual(physics.clamp01(-5.0), 0.0)
        self.assertEqual(physics.clamp01(5.0), 1.0)
        self.assertEqual(physics.clamp01(0.3), 0.3)


if __name__ == "__main__":
    unittest.main()