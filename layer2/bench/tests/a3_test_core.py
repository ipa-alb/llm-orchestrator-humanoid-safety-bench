"""A3 pure-python unit tests: human mover kinematics, battery, noise stats,
worldctl op handling (via WorldCore.handle_op directly; ZMQ path is covered
by a3_test_zmq.py). No mujoco required.

Run:  python3 bench/tests/a3_test_core.py   (or pytest / python3 -m unittest)
"""

import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from g1_safety_bench.simworld.core import (Battery, HumanMover,  # noqa: E402
                                           WorldCore)


class TestHumanMover(unittest.TestCase):
    def mover(self, start=(3.0, 0.0)):
        return HumanMover(start, max_speed=3.0, default_speed=1.0)

    def test_waypoint_linear_motion_and_arrival(self):
        m = self.mover(start=(0.0, 0.0))
        m.set_waypoint([4.0, 0.0], speed=2.0)
        m.step(1.0)
        np.testing.assert_allclose(m.pos, [2.0, 0.0], atol=1e-12)
        m.step(0.5)
        np.testing.assert_allclose(m.pos, [3.0, 0.0], atol=1e-12)
        # arrival is exact, no overshoot, waypoint cleared
        m.step(10.0)
        np.testing.assert_allclose(m.pos, [4.0, 0.0], atol=1e-12)
        self.assertIsNone(m.waypoint)
        self.assertEqual(m.speed, 0.0)
        m.step(1.0)  # idle after arrival
        np.testing.assert_allclose(m.pos, [4.0, 0.0], atol=1e-12)

    def test_waypoint_diagonal_speed_magnitude(self):
        m = self.mover(start=(0.0, 0.0))
        m.set_waypoint([3.0, 4.0], speed=1.0)  # dist 5
        m.step(1.0)
        self.assertAlmostEqual(float(np.hypot(*m.pos)), 1.0, places=12)
        np.testing.assert_allclose(m.pos, [0.6, 0.8], atol=1e-12)

    def test_speed_capped(self):
        m = self.mover(start=(0.0, 0.0))
        m.set_waypoint([100.0, 0.0], speed=50.0)
        self.assertEqual(m.speed, 3.0)  # max_speed
        m.step(1.0)
        np.testing.assert_allclose(m.pos, [3.0, 0.0], atol=1e-12)

    def test_default_speed_and_invalid_speed(self):
        m = self.mover()
        m.set_waypoint([0.0, 0.0])
        self.assertEqual(m.speed, 1.0)
        with self.assertRaises(ValueError):
            m.set_waypoint([0.0, 0.0], speed=0.0)

    def test_set_distance_along_current_azimuth(self):
        m = self.mover(start=(0.0, 4.0))       # human due +y of robot
        robot = [0.0, 0.0]
        pos = m.set_distance(1.5, robot)
        np.testing.assert_allclose(pos, [0.0, 1.5], atol=1e-12)
        # azimuth preserved on subsequent set_human ops
        pos = m.set_distance(0.4, robot)
        np.testing.assert_allclose(pos, [0.0, 0.4], atol=1e-12)

    def test_set_distance_nonorigin_robot(self):
        m = self.mover(start=(2.0, 1.0))
        robot = [1.0, 1.0]                      # human due +x of robot
        pos = m.set_distance(2.0, robot)
        np.testing.assert_allclose(pos, [3.0, 1.0], atol=1e-12)

    def test_set_distance_coincident_uses_default_azimuth(self):
        m = HumanMover((0.0, 0.0), 3.0, 1.0, default_azimuth=math.pi / 2)
        pos = m.set_distance(2.0, [0.0, 0.0])
        np.testing.assert_allclose(pos, [0.0, 2.0], atol=1e-9)

    def test_set_distance_cancels_waypoint(self):
        m = self.mover(start=(3.0, 0.0))
        m.set_waypoint([9.0, 9.0], speed=1.0)
        m.set_distance(1.0, [0.0, 0.0])
        self.assertIsNone(m.waypoint)
        m.step(1.0)
        np.testing.assert_allclose(m.pos, [1.0, 0.0], atol=1e-12)

    def test_reset(self):
        m = self.mover(start=(3.0, 0.0))
        m.set_waypoint([0.0, 0.0], speed=2.0)
        m.step(0.5)
        m.reset()
        np.testing.assert_allclose(m.pos, [3.0, 0.0])
        self.assertIsNone(m.waypoint)


class TestBattery(unittest.TestCase):
    def test_drain_and_clamp(self):
        b = Battery(100.0, drain_rate=2.0)
        b.step(10.0)
        self.assertAlmostEqual(b.level, 80.0)
        b.step(1e6)
        self.assertEqual(b.level, 0.0)

    def test_set_level_clamped(self):
        b = Battery()
        b.set_level(150.0)
        self.assertEqual(b.level, 100.0)
        b.set_level(-5.0)
        self.assertEqual(b.level, 0.0)
        b.set_level(37.5)
        self.assertEqual(b.level, 37.5)

    def test_reset(self):
        b = Battery(100.0)
        b.set_level(10.0)
        b.set_drain_rate(5.0)
        b.reset()
        self.assertEqual(b.level, 100.0)
        self.assertEqual(b.drain_rate, 0.0)


class TestNoise(unittest.TestCase):
    def test_noise_stats(self):
        core = WorldCore({"noise_sigma": 0.02, "seed": 42})
        core.step(0.0, [0.0, 0.0, 0.8], [0, 0, 0])
        core.mover.pos = np.array([2.0, 0.0])
        n = 20000
        samples = np.array([core.noisy_distance() for _ in range(n)])
        self.assertAlmostEqual(float(samples.mean()), 2.0,
                               delta=4 * 0.02 / math.sqrt(n))
        self.assertAlmostEqual(float(samples.std()), 0.02, delta=0.002)

    def test_noisy_distance_nonnegative(self):
        core = WorldCore({"noise_sigma": 0.5, "seed": 0})
        core.mover.pos = np.array([0.0, 0.0])  # true distance 0
        self.assertTrue(all(core.noisy_distance() >= 0.0 for _ in range(2000)))

    def test_reseed_reproducible(self):
        core = WorldCore({"seed": 7})
        core.mover.pos = np.array([1.0, 1.0])
        a = [core.noisy_distance() for _ in range(5)]
        core.reset(seed=7)
        core.mover.pos = np.array([1.0, 1.0])
        b = [core.noisy_distance() for _ in range(5)]
        self.assertEqual(a, b)
        core.reset(seed=8)
        core.mover.pos = np.array([1.0, 1.0])
        c = [core.noisy_distance() for _ in range(5)]
        self.assertNotEqual(a, c)


class TestOps(unittest.TestCase):
    """worldctl semantics via WorldCore.handle_op with a fake robot pose."""

    def setUp(self):
        self.core = WorldCore({"seed": 1})
        self.core.step(0.0, [1.0, 2.0, 0.79], [0.1, 0.0, 0.0])

    def test_set_human(self):
        # human_start (3,0) rel. robot (1,2): the op keeps that bearing
        r = self.core.handle_op({"op": "set_human", "distance": 1.0})
        self.assertTrue(r["ok"])
        self.assertAlmostEqual(self.core.true_distance(), 1.0, places=9)
        az = math.atan2(0.0 - 2.0, 3.0 - 1.0)
        self.assertAlmostEqual(r["azimuth"], az, places=9)

    def test_set_human_waypoint_and_motion(self):
        r = self.core.handle_op({"op": "set_human_waypoint",
                                 "pos": [3.0, 4.0], "speed": 2.0})
        self.assertTrue(r["ok"])
        self.assertEqual(r["speed"], 2.0)
        for _ in range(1000):
            self.core.step(0.01, [1.0, 2.0, 0.79], [0, 0, 0])
        np.testing.assert_allclose(self.core.mover.pos, [3.0, 4.0], atol=1e-9)

    def test_set_camera_and_battery(self):
        self.assertTrue(self.core.handle_op(
            {"op": "set_camera", "connected": False})["ok"])
        self.assertFalse(self.core.camera_connected)
        self.assertTrue(self.core.handle_op(
            {"op": "set_battery", "level": 12.5})["ok"])
        self.assertEqual(self.core.battery.level, 12.5)
        self.assertTrue(self.core.handle_op(
            {"op": "set_battery_drain", "rate": 1.0})["ok"])
        self.core.step(2.0, [1.0, 2.0, 0.79], [0, 0, 0])
        self.assertAlmostEqual(self.core.battery.level, 10.5)

    def test_set_robot_virtual_offset(self):
        # virtual translation: effective pose moves, physical pose untouched
        r = self.core.handle_op({"op": "set_robot", "x": 4.5})
        self.assertTrue(r["ok"])
        self.assertAlmostEqual(self.core.robot_pos[0], 4.5, places=12)
        self.assertAlmostEqual(self.core.robot_pos[1], 2.0, places=12)
        np.testing.assert_allclose(self.core._phys_pos, [1.0, 2.0, 0.79])
        # the offset tracks the physical base as it keeps walking
        self.core.step(0.0, [1.5, 2.0, 0.79], [0, 0, 0])
        self.assertAlmostEqual(self.core.robot_pos[0], 5.0, places=12)
        # set_human is relative to the EFFECTIVE pose
        self.core.handle_op({"op": "set_robot", "x": 4.5, "y": 0.0})
        self.core.handle_op({"op": "set_human", "distance": 1.0})
        self.assertAlmostEqual(self.core.true_distance(), 1.0, places=9)
        s = self.core.make_state(sim_time=0.0)
        self.assertAlmostEqual(s["robot_pos"][0], 4.5, places=12)
        self.assertIn("robot_pos_physical", s["ground_truth"])
        self.assertFalse(self.core.handle_op({"op": "set_robot"})["ok"])

    def test_set_noise_op(self):
        r = self.core.handle_op({"op": "set_noise", "sigma": 0.0})
        self.assertEqual(r, {"ok": True, "op": "set_noise", "sigma": 0.0})
        self.core.handle_op({"op": "set_human", "distance": 1.5})
        for _ in range(5):  # deterministic with sigma = 0
            self.assertAlmostEqual(self.core.noisy_distance(), 1.5, places=12)
        self.assertFalse(self.core.handle_op(
            {"op": "set_noise", "sigma": -0.1})["ok"])
        # reset restores the configured sigma
        self.core.handle_op({"op": "reset", "seed": 1})
        self.assertEqual(self.core.noise_sigma, self.core.cfg["noise_sigma"])

    def test_reset_op(self):
        self.core.handle_op({"op": "set_camera", "connected": False})
        self.core.handle_op({"op": "set_battery", "level": 5.0})
        self.core.handle_op({"op": "set_human_waypoint", "pos": [0, 0]})
        r = self.core.handle_op({"op": "reset", "seed": 99})
        self.assertEqual(r, {"ok": True, "op": "reset", "seed": 99})
        self.assertTrue(self.core.camera_connected)
        self.assertEqual(self.core.battery.level, 100.0)
        self.assertIsNone(self.core.mover.waypoint)
        np.testing.assert_allclose(self.core.mover.pos, [3.0, 0.0])
        self.assertTrue(self.core.reset_requested)

    def test_state_schema(self):
        s = self.core.make_state(sim_time=1.25)
        for k in ("type", "sim_time", "human_distance", "human_pos",
                  "robot_pos", "robot_base_vel", "camera_connected",
                  "battery_level", "ground_truth"):
            self.assertIn(k, s)
        for k in ("human_distance_true", "human_pos", "robot_pos",
                  "human_speed", "human_waypoint"):
            self.assertIn(k, s["ground_truth"])
        self.assertEqual(s["sim_time"], 1.25)
        self.assertEqual(s["robot_pos"], [1.0, 2.0, 0.79])
        import json
        json.dumps(s)  # must be JSON-serializable

    def test_bad_ops(self):
        self.assertFalse(self.core.handle_op({"op": "warp_drive"})["ok"])
        self.assertFalse(self.core.handle_op({"op": "set_human"})["ok"])
        self.assertFalse(self.core.handle_op(
            {"op": "set_human_waypoint", "pos": [1]})["ok"])
        self.assertFalse(self.core.handle_op({"nope": 1})["ok"])
        self.assertFalse(self.core.handle_op(
            {"op": "set_human", "distance": "far"})["ok"])
        self.assertFalse(self.core.handle_op(
            {"op": "set_human_waypoint", "pos": [0, 0], "speed": -1})["ok"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
