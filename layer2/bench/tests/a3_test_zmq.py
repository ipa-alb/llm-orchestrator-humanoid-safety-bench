"""A3 ZMQ integration tests: worldctl ops over a real REP socket and
worldstate over a real PUB socket, on ephemeral ports, with a fake robot
state provider (no mujoco). Single-threaded: REQ.send -> server.poll() ->
REQ.recv, exploiting REQ/REP async send.

Run:  python3 bench/tests/a3_test_zmq.py   (or pytest / python3 -m unittest)
"""

import json
import os
import sys
import time
import unittest

import zmq

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from g1_safety_bench.simworld.core import WorldCore  # noqa: E402
from g1_safety_bench.simworld.zmq_iface import (StatePublisher,  # noqa: E402
                                                WorldCtlServer)

RCV_TIMEOUT_MS = 5000


class FakeRobot:
    """Fake robot base state provider (what mjData supplies in production)."""

    def __init__(self):
        self.pos = [0.0, 0.0, 0.793]
        self.vel = [0.0, 0.0, 0.0]


class ZmqWorldFixture(unittest.TestCase):
    def setUp(self):
        self.ctx = zmq.Context()
        self.core = WorldCore({"seed": 3})
        self.robot = FakeRobot()
        self.server = WorldCtlServer(self.core.handle_op, port=0,
                                     host="127.0.0.1", ctx=self.ctx)
        self.pub = StatePublisher(port=0, host="127.0.0.1", ctx=self.ctx)

        self.req = self.ctx.socket(zmq.REQ)
        self.req.setsockopt(zmq.RCVTIMEO, RCV_TIMEOUT_MS)
        self.req.setsockopt(zmq.LINGER, 0)
        self.req.connect(f"tcp://127.0.0.1:{self.server.port}")

        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.setsockopt(zmq.SUBSCRIBE, b"")
        self.sub.setsockopt(zmq.RCVTIMEO, RCV_TIMEOUT_MS)
        self.sub.setsockopt(zmq.LINGER, 0)
        self.sub.connect(f"tcp://127.0.0.1:{self.pub.port}")

    def tearDown(self):
        for s in (self.req, self.sub):
            s.close(0)
        self.server.close()
        self.pub.close()
        self.ctx.term()

    def step_world(self, n=1, dt=0.002):
        """One fake physics step: poll ops, advance core with fake robot."""
        for _ in range(n):
            self.server.poll()
            self.core.step(dt, self.robot.pos, self.robot.vel)

    def raw_rpc(self, payload: bytes, max_wait_s=5.0):
        """Send raw bytes, keep 'stepping physics' until the reply arrives
        (the first send races the async TCP connect)."""
        self.req.send(payload)
        deadline = time.monotonic() + max_wait_s
        while time.monotonic() < deadline:
            self.step_world()
            if self.req.poll(10):
                return json.loads(self.req.recv().decode())
        self.fail(f"no worldctl reply within {max_wait_s}s")

    def rpc(self, op):
        return self.raw_rpc(json.dumps(op).encode())


class TestWorldCtlOverZmq(ZmqWorldFixture):
    def test_poll_nonblocking_when_idle(self):
        self.assertEqual(self.server.poll(), 0)  # returns immediately

    def test_ping_and_roundtrip(self):
        self.assertEqual(self.rpc({"op": "ping"}), {"ok": True, "op": "ping"})

    def test_ops_mutate_state(self):
        r = self.rpc({"op": "set_human", "distance": 0.5})
        self.assertTrue(r["ok"])
        self.assertAlmostEqual(self.core.true_distance(), 0.5, places=9)

        r = self.rpc({"op": "set_camera", "connected": False})
        self.assertTrue(r["ok"])
        self.assertFalse(self.core.camera_connected)

        r = self.rpc({"op": "set_battery", "level": 42.0})
        self.assertTrue(r["ok"])
        self.assertEqual(self.core.battery.level, 42.0)

        r = self.rpc({"op": "set_human_waypoint", "pos": [0.5, 0.0],
                      "speed": 1.0})
        self.assertTrue(r["ok"])
        self.step_world(n=500, dt=0.002)  # 1 s of walking
        self.assertAlmostEqual(float(self.core.mover.pos[0]), 0.5, places=6)

    def test_reset_over_zmq_reseeds(self):
        # two resets with the same seed must reproduce the same noise stream
        r = self.rpc({"op": "reset", "seed": 3})
        self.assertTrue(r["ok"])
        self.rpc({"op": "set_human", "distance": 2.0})
        b = [self.core.noisy_distance() for _ in range(4)]
        self.rpc({"op": "reset", "seed": 3})
        self.rpc({"op": "set_human", "distance": 2.0})
        c = [self.core.noisy_distance() for _ in range(4)]
        self.assertEqual(b, c)
        # a different seed must give a different stream
        self.rpc({"op": "reset", "seed": 4})
        self.rpc({"op": "set_human", "distance": 2.0})
        d = [self.core.noisy_distance() for _ in range(4)]
        self.assertNotEqual(b, d)

    def test_bad_json_and_bad_op_still_reply(self):
        r = self.raw_rpc(b"{not json")
        self.assertFalse(r["ok"])
        self.assertIn("bad JSON", r["error"])
        # socket still usable afterwards (REP replied exactly once)
        r = self.rpc({"op": "nonsense"})
        self.assertFalse(r["ok"])
        self.assertEqual(self.rpc({"op": "ping"})["ok"], True)

    def test_burst_of_requests_drained(self):
        # queue several REQ clients' worth via DEALER to simulate a burst
        dealer = self.ctx.socket(zmq.DEALER)
        try:
            dealer.setsockopt(zmq.RCVTIMEO, RCV_TIMEOUT_MS)
            dealer.setsockopt(zmq.LINGER, 0)
            dealer.connect(f"tcp://127.0.0.1:{self.server.port}")
            n = 5
            for i in range(n):
                dealer.send_multipart(
                    [b"", json.dumps({"op": "ping"}).encode()])
            handled = 0
            deadline = time.monotonic() + 5.0
            while handled < n and time.monotonic() < deadline:
                handled += self.server.poll()
                time.sleep(0.005)
            self.assertEqual(handled, n)
            for _ in range(n):
                frames = dealer.recv_multipart()
                self.assertTrue(json.loads(frames[-1].decode())["ok"])
        finally:
            dealer.close(0)  # before tearDown's ctx.term()


class TestWorldStateOverZmq(ZmqWorldFixture):
    def _recv_state(self):
        return json.loads(self.sub.recv().decode())

    def _wait_subscribed(self):
        """PUB/SUB joins asynchronously: publish until the SUB sees one."""
        for _ in range(100):
            self.pub.publish(self.core.make_state(0.0))
            if self.sub.poll(50):
                return self._recv_state()
        self.fail("subscriber never received a worldstate")

    def test_worldstate_schema_and_values(self):
        self._wait_subscribed()
        self.rpc({"op": "set_human", "distance": 1.0})
        self.rpc({"op": "set_battery", "level": 55.0})
        self.step_world()
        self.pub.publish(self.core.make_state(sim_time=0.123))
        s = self._recv_state()
        for k in ("type", "sim_time", "human_distance", "human_pos",
                  "robot_pos", "robot_base_vel", "camera_connected",
                  "battery_level", "ground_truth"):
            self.assertIn(k, s)
        self.assertEqual(s["type"], "worldstate")
        self.assertEqual(s["battery_level"], 55.0)
        self.assertEqual(s["robot_pos"], self.robot.pos)
        gt = s["ground_truth"]
        self.assertAlmostEqual(gt["human_distance_true"], 1.0, places=9)
        # noisy sensor value close to truth (sigma 0.02 -> 6 sigma bound)
        self.assertLess(abs(s["human_distance"] - 1.0), 0.12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
