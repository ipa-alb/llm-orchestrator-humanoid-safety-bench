#!/usr/bin/env python3
"""A4 live acceptance: full DDS + ZMQ stack against the headless sim bridge.

Prerequisite (separate process/container, W0's launcher -- note --dt 0.002):
    python3 scripts/run_sim_headless.py --dt 0.002

This test then:
  1. spawns the loco runner (python -m g1_safety_bench.loco.runner),
  2. binds a ZMQ PUB on tcp://127.0.0.1:5556 (skill-message injection),
  3. measures base motion from rt/sportmodestate (frame_pos/frame_vel of the
     pelvis imu site, published by the vendored bridge), and
  4. checks:  STAND 30 s (height > 0.55 m throughout),
              WALK cmd 0.3 m/s x 5 s (planar displacement in [0.7, 1.95] m),
              STOP (0.5 s-windowed base speed < 0.05 m/s within 1.5 s,
              still standing 3 s later).

Run inside a g1-base container:
    docker run --rm --network host \
        -v <repo>/layer2:/ws g1-base \
        bash -c "cd /ws && python3 bench/tests/a4_live_acceptance.py"

Exit codes: 0 pass, 1 fail, 2 environment not ready (no sim lowstate).
"""
import argparse
import json
import os
import subprocess
import sys
import time
from collections import deque

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
BENCH = os.path.join(REPO, "bench")
sys.path.insert(0, BENCH)

from g1_safety_bench.loco import config as C


class BaseMonitor:
    """Tracks pelvis world position/velocity from rt/sportmodestate."""

    def __init__(self):
        from unitree_sdk2py.core.channel import ChannelSubscriber
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
        self.pos = None
        self.vel = None
        self.n = 0
        self._hist = deque(maxlen=4000)   # (t, x, y, z)
        self.sub = ChannelSubscriber(C.TOPIC_HIGHSTATE, SportModeState_)
        self.sub.Init(self._cb, 10)

    def _cb(self, msg):
        self.pos = np.array(msg.position, dtype=np.float64)
        self.vel = np.array(msg.velocity, dtype=np.float64)
        self.n += 1
        self._hist.append((time.monotonic(), *self.pos))

    def windowed_speed(self, window_s=0.5):
        """Planar speed from position displacement over the last window."""
        if len(self._hist) < 2:
            return float("nan")
        now = self._hist[-1]
        for entry in self._hist:
            if now[0] - entry[0] <= window_s:
                first = entry
                break
        dt = now[0] - first[0]
        if dt <= 1e-6:
            return float("nan")
        d = np.hypot(now[1] - first[1], now[2] - first[2])
        return float(d / dt)


def monitor_span(mon, seconds):
    """Watch height for `seconds`; returns (min_height, fell)."""
    t0 = time.monotonic()
    min_h = np.inf
    while time.monotonic() - t0 < seconds:
        time.sleep(0.02)
        if mon.pos is not None:
            min_h = min(min_h, mon.pos[2])
    return min_h, (min_h <= 0.55)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", type=int, default=C.DDS_DOMAIN_ID)
    ap.add_argument("--iface", default=C.DDS_INTERFACE)
    ap.add_argument("--fallback", action="store_true",
                    help="test scripted_base instead of the ONNX runner")
    ap.add_argument("--settle", type=float, default=5.0,
                    help="seconds to wait after runner start before STAND")
    args = ap.parse_args()

    import zmq
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize

    ChannelFactoryInitialize(args.domain, args.iface)
    mon = BaseMonitor()

    # skill-message PUB (runner SUBs and connects to this endpoint)
    ctx = zmq.Context.instance()
    pub = ctx.socket(zmq.PUB)
    pub.bind(C.ZMQ_SKILL_ENDPOINT.replace("127.0.0.1", "*"))

    def send(msg):
        pub.send_string(json.dumps(msg))
        print(f"[a4_live] sent {msg}")

    # Spawn the runner BEFORE waiting for sim traffic: the robot spawns
    # standing but uncontrolled, and tips over within ~2 s of physics start,
    # so the controller must engage as soon as lowstate flows.  (The runner
    # itself blocks on the first lowstate.)
    module = ("g1_safety_bench.loco.scripted_base" if args.fallback
              else "g1_safety_bench.loco.runner")
    env = dict(os.environ)
    env["PYTHONPATH"] = BENCH + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.Popen(
        [sys.executable, "-m", module, "--domain", str(args.domain),
         "--iface", args.iface],
        env=env)
    print(f"[a4_live] spawned {module} (pid {proc.pid})")

    print("[a4_live] waiting for rt/sportmodestate from the sim bridge ...")
    # The bridge starts its publisher threads before the physics loop, so the
    # first samples can be all-zero: wait for a plausible standing height.
    t0 = time.monotonic()
    while mon.pos is None or mon.pos[2] < 0.3:
        time.sleep(0.05)
        if time.monotonic() - t0 > 60.0:
            if mon.n == 0:
                print("[a4_live] NOT READY: no sim traffic on domain "
                      f"{args.domain}/{args.iface}. Start "
                      "scripts/run_sim_headless.py --dt 0.002.")
            else:
                print("[a4_live] NOT READY: robot never at standing height "
                      f"(h={mon.pos[2]:.3f}); restart the sim and rerun "
                      "immediately (it must not have fallen already).")
            proc.terminate()
            return 2
    print(f"[a4_live] sim ok, base at {mon.pos.round(3)}")

    results = {}
    ok = True
    try:
        time.sleep(args.settle)   # ramp + policy engage
        if mon.pos is None or mon.pos[2] < 0.55:
            print("[a4_live] FAIL: robot not standing after settle "
                  f"(h={None if mon.pos is None else mon.pos[2]:.3f})")
            return 1

        # (i) STAND 30 s
        min_h, fell = monitor_span(mon, 30.0)
        stand_ok = not fell
        results["stand"] = f"min height {min_h:.3f} m over 30 s (>0.55 required)"
        print(f"[a4_live] STAND: {'PASS' if stand_ok else 'FAIL'} - {results['stand']}")

        # (ii) WALK
        p0 = mon.pos.copy()
        send({"type": "cmd_vel", "vx": 0.3, "vy": 0.0, "wz": 0.0})
        min_h, fell = monitor_span(mon, 5.0)
        disp = float(np.hypot(*(mon.pos - p0)[:2]))
        walk_ok = (0.7 <= disp <= 1.95) and not fell
        results["walk"] = (f"displacement {disp:.3f} m in 5 s (band [0.7, 1.95]), "
                           f"min height {min_h:.3f} m")
        print(f"[a4_live] WALK : {'PASS' if walk_ok else 'FAIL'} - {results['walk']}")

        # (iii) STOP
        send({"type": "stop"})
        t_stop = time.monotonic()
        t_below = None
        while time.monotonic() - t_stop < 1.5:
            time.sleep(0.05)
            v = mon.windowed_speed(0.5)
            if t_below is None and v == v and v < 0.05:
                t_below = time.monotonic() - t_stop
        v15 = mon.windowed_speed(0.5)
        min_h, fell = monitor_span(mon, 3.0)
        stop_ok = (t_below is not None) and not fell
        results["stop"] = (f"windowed speed<0.05 first at "
                           f"{t_below if t_below is not None else float('nan'):.2f} s "
                           f"(<=1.5 required), v(1.5s)={v15:.3f} m/s, "
                           f"standing after (min h {min_h:.3f} m)")
        print(f"[a4_live] STOP : {'PASS' if stop_ok else 'FAIL'} - {results['stop']}")

        ok = stand_ok and walk_ok and stop_ok
        print(f"[a4_live] overall: {'PASS' if ok else 'FAIL'}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
