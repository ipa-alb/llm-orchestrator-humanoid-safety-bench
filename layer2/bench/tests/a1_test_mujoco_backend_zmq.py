"""A1 tests: MuJoCoBackend ZMQ message emission against a fake sim.

Binds real ZMQ sockets on ephemeral localhost ports in-process:
  fake sim   : PUB (worldstate) + REP (worldctl) on random ports
  fake loco  : SUB connected to the backend's bound skill PUB
then checks sensor mapping, skill message shapes, worldctl ops, and
advance() sim-time synchronization.

Run standalone:  python bench/tests/a1_test_mujoco_backend_zmq.py
Requires pyzmq (a hard dependency of g1_safety_bench); skips gracefully if
it is missing on the host.
"""

import json
import socket
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import zmq
    HAVE_ZMQ = True
except ImportError:
    HAVE_ZMQ = False


def _skip_all(msg):
    print(f"SKIP: {msg}")
    if "pytest" in sys.modules:
        import pytest
        pytest.skip(msg, allow_module_level=True)
    sys.exit(0)


if not HAVE_ZMQ:
    _skip_all("pyzmq not installed — MuJoCoBackend ZMQ test needs it "
              "(mock-free test preferred; pyzmq is a package dependency)")


def _free_tcp_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class FakeSim:
    """Publishes worldstate and answers worldctl, like A3's sim process."""

    def __init__(self):
        self.ctx = zmq.Context()
        self.pub = self.ctx.socket(zmq.PUB)
        self.ws_port = self.pub.bind_to_random_port("tcp://127.0.0.1")
        self.rep = self.ctx.socket(zmq.REP)
        self.ctl_port = self.rep.bind_to_random_port("tcp://127.0.0.1")
        self.ctl_requests: list[dict] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.state = {
            "sim_time": 0.0,
            "human_distance": 2.34,          # noisy measurement
            "human_pos": [2.0, 1.0],
            "robot_pos": [1.0, 0.5, 0.0],
            "robot_base_vel": [0.0, 0.0, 0.0],
            "camera_connected": True,
            "battery_level": 77,
            "ground_truth": {"human_distance": 2.30, "robot_pos": [1.0, 0.5, 0.0]},
        }
        self._threads = [
            threading.Thread(target=self._pub_loop, daemon=True),
            threading.Thread(target=self._rep_loop, daemon=True),
        ]
        for t in self._threads:
            t.start()

    def set_state(self, **kw):
        with self._lock:
            self.state.update(kw)

    def _pub_loop(self):
        while not self._stop.is_set():
            with self._lock:
                self.state["sim_time"] = round(self.state["sim_time"] + 0.02, 4)
                msg = dict(self.state)
            self.pub.send_json(msg)
            time.sleep(0.02)

    def _rep_loop(self):
        poller = zmq.Poller()
        poller.register(self.rep, zmq.POLLIN)
        while not self._stop.is_set():
            if dict(poller.poll(timeout=50)):
                req = self.rep.recv_json()
                self.ctl_requests.append(req)
                self.rep.send_json({"status": "ok"})

    def close(self):
        self._stop.set()
        for t in self._threads:
            t.join(timeout=1.0)
        for s in (self.pub, self.rep):
            s.setsockopt(zmq.LINGER, 0)
            s.close()
        self.ctx.term()


def _drain(sub, timeout=1.0):
    """Collect all JSON messages arriving on `sub` within `timeout` seconds."""
    msgs = []
    deadline = time.monotonic() + timeout
    poller = zmq.Poller()
    poller.register(sub, zmq.POLLIN)
    while time.monotonic() < deadline:
        if dict(poller.poll(timeout=50)):
            msgs.append(sub.recv_json())
            # short grace period for trailing messages
            deadline = min(deadline, time.monotonic() + 0.2)
    return msgs


def _wait_for_worldstate(backend, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            backend.read_sensor("battery_level")
            return
        except RuntimeError:
            time.sleep(0.02)
    raise AssertionError("backend never received worldstate")


def test_mujoco_backend_end_to_end():
    from g1_safety_bench.backend.constants import V_MAX
    from g1_safety_bench.backend.mujoco_zmq import MuJoCoBackend

    sim = FakeSim()
    skill_port = _free_tcp_port()
    backend = MuJoCoBackend(
        worldstate_addr=f"tcp://127.0.0.1:{sim.ws_port}",
        skill_addr=f"tcp://127.0.0.1:{skill_port}",
        worldctl_addr=f"tcp://127.0.0.1:{sim.ctl_port}",
        worldctl_timeout_s=2.0,
    )
    loco_ctx = zmq.Context()
    loco_sub = loco_ctx.socket(zmq.SUB)
    loco_sub.setsockopt(zmq.SUBSCRIBE, b"")
    loco_sub.connect(f"tcp://127.0.0.1:{skill_port}")
    time.sleep(0.3)  # PUB/SUB slow-joiner grace

    try:
        # --- sensor mapping from worldstate ---
        _wait_for_worldstate(backend)
        assert backend.read_sensor("human_distance") == 2.34
        assert backend.read_sensor("camera_connected") is True
        assert backend.read_sensor("battery_level") == 77
        assert backend.read_sensor("robot_position") == (1.0, 0.5, 0.0)
        assert backend.read_sensor("robot_speed") == 0.0
        assert backend.read_sensor("robot_status") == "idle"

        # --- move_forward: cmd_vel with vx = (speed/100) * V_MAX ---
        out = backend.apply_skill("move_forward", speed=40)
        assert out == {"status": "ok", "speed": 40}
        msgs = _drain(loco_sub)
        assert len(msgs) == 1, msgs
        m = msgs[0]
        assert m["type"] == "cmd_vel" and m["vy"] == 0.0 and m["wz"] == 0.0
        assert abs(m["vx"] - 0.4 * V_MAX) < 1e-9
        assert backend.read_sensor("robot_speed") == 40
        assert backend.read_sensor("robot_status") == "moving"

        # --- move_backward: negative vx ---
        out = backend.apply_skill("move_backward", speed=100)
        assert out == {"status": "ok", "speed": 100}
        (m,) = _drain(loco_sub)
        assert m["type"] == "cmd_vel" and abs(m["vx"] - (-V_MAX)) < 1e-9

        # --- out-of-bounds refusal: no message published ---
        sim.set_state(robot_pos=[4.95, 0.0, 0.0])
        time.sleep(0.1)  # let the new worldstate reach the backend cache
        out = backend.apply_skill("move_forward", speed=100)  # 4.95 + 1.0 > 5.0
        assert out == {"status": "out_of_bounds", "speed": 100}
        assert _drain(loco_sub, timeout=0.3) == []
        sim.set_state(robot_pos=[1.0, 0.5, 0.0])

        # --- stop / sit_down ---
        assert backend.apply_skill("stop") == {"status": "ok"}
        (m,) = _drain(loco_sub)
        assert m == {"type": "stop"}
        assert backend.read_sensor("robot_status") == "stopped"

        assert backend.apply_skill("sit_down") == {"status": "ok"}
        (m,) = _drain(loco_sub)
        assert m == {"type": "sit_down"}
        assert backend.read_sensor("robot_status") == "seated"

        # --- move_hand ---
        out = backend.apply_skill("move_hand", x=0.3, y=-0.2, z=1.1)
        assert out == {"status": "ok", "position": {"x": 0.3, "y": -0.2, "z": 1.1}}
        (m,) = _drain(loco_sub)
        assert m == {"type": "arm_target", "target_xyz": [0.3, -0.2, 1.1]}
        # out-of-bounds hand target refused without publishing
        out = backend.apply_skill("move_hand", x=0.0, y=0.0, z=5.0)
        assert out["status"] == "out_of_bounds"
        assert _drain(loco_sub, timeout=0.3) == []

        # --- reset + scenario_cfg → worldctl ops ---
        backend.reset(seed=7, scenario_cfg={
            "human_distance": 0.4,
            "camera_connected": False,
            "battery_level": 12.0,
            "human_waypoint": {"pos": [3.0, 1.0], "speed": 0.8},
        })
        time.sleep(0.2)
        reqs = list(sim.ctl_requests)
        assert reqs[0] == {"op": "reset", "seed": 7}
        assert {"op": "set_human", "distance": 0.4} in reqs
        assert {"op": "set_camera", "connected": False} in reqs
        assert {"op": "set_battery", "level": 12.0} in reqs
        assert {"op": "set_human_waypoint", "pos": [3.0, 1.0], "speed": 0.8} in reqs
        assert backend.read_sensor("robot_status") == "idle"  # reset clears cmd state

        # --- ground_truth passthrough ---
        gt = backend.ground_truth()
        assert gt["human_distance"] == 2.30
        assert gt["commanded_status"] == "idle"
        assert gt["sim_time"] is not None

        # --- advance(): sim_time moves under continuous physics ---
        t0 = time.monotonic()
        backend.advance(0.2)  # fake sim advances 0.02 sim-s per 20 ms
        assert time.monotonic() - t0 < 2.0

        # --- execute_tool parity through the real backend ---
        from g1_safety_bench.backend import execute_tool
        assert execute_tool(backend, "get_camera_status") == {"status": "connected"}
        assert execute_tool(backend, "get_battery_level") == {"battery_percent": 77}
        assert execute_tool(backend, "get_human_distance") == {"distance_meters": 2.34}
        _drain(loco_sub, timeout=0.1)
    finally:
        backend.close()
        loco_sub.setsockopt(zmq.LINGER, 0)
        loco_sub.close()
        loco_ctx.term()
        sim.close()


def test_worldctl_timeout_raises():
    from g1_safety_bench.backend.mujoco_zmq import MuJoCoBackend
    # REP port with nobody answering
    dead_port = _free_tcp_port()
    ws_port = _free_tcp_port()
    backend = MuJoCoBackend(
        worldstate_addr=f"tcp://127.0.0.1:{ws_port}",
        skill_addr=f"tcp://127.0.0.1:{_free_tcp_port()}",
        worldctl_addr=f"tcp://127.0.0.1:{dead_port}",
        worldctl_timeout_s=0.3,
    )
    try:
        try:
            backend.reset(seed=0)
        except TimeoutError:
            pass
        else:
            raise AssertionError("expected TimeoutError from dead worldctl")
    finally:
        backend.close()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"OK ({len(fns)} tests)")
