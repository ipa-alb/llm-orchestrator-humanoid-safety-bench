#!/usr/bin/env python3
"""A4 in-process acceptance: MuJoCo + real ONNX policy, NO DDS/ZMQ.

Couples LocoController directly to the vendored g1_29dof scene, applying
torques with the exact PD law of vendor/unitree_mujoco simulate_python's
UnitreeSdk2Bridge.LowCmdHandler:

    ctrl[i] = tau + kp*(q_des - sensordata[i]) + kd*(dq_des - sensordata[i+nu])

Acceptance (same criteria as the live test):
  (i)   STAND: 30 s under policy, pelvis height > 0.55 m throughout
  (ii)  WALK : cmd_vel vx=0.3 for 5 s -> planar displacement in [0.7, 1.95] m
               (1.0 m +/- 30 percent band around the lower bound; ideal 1.5 m), no fall
  (iii) STOP : from walk, stop -> planar base speed < 0.05 m/s within 1.5 s,
               still standing 3 s later

    python3 bench/tests/a4_inproc_acceptance.py [--dt 0.002]

Physics dt matters: at dt=0.002 all three criteria pass; at dt=0.005 the
PD loop limit-cycles slightly (STAND/WALK still pass, STOP residual speed
0.067 > 0.05 m/s).  Run the live sim with --dt 0.002.

Needs: mujoco, numpy, pyyaml, onnxruntime.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "bench"))
SCENE = os.path.join(REPO, "vendor", "unitree_mujoco", "unitree_robots", "g1",
                     "scene_29dof.xml")

from g1_safety_bench.loco import config as C
from g1_safety_bench.loco.controller import LocoController, StateSnapshot


class InprocSim:
    """Minimal stand-in for the vendored sim bridge (same sensor layout)."""

    def __init__(self, scene=SCENE, dt=0.005):
        import mujoco
        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(scene)
        self.model.opt.timestep = dt
        self.data = mujoco.MjData(self.model)
        self.nu = self.model.nu
        assert self.nu == C.NUM_MOTORS, f"scene has {self.nu} actuators"
        self.dim_motor_sensor = 3 * self.nu
        mujoco.mj_forward(self.model, self.data)

    def snapshot(self) -> StateSnapshot:
        sd = self.data.sensordata
        q = sd[0:self.nu].astype(np.float32).copy()
        dq = sd[self.nu:2 * self.nu].astype(np.float32).copy()
        base = self.dim_motor_sensor
        quat = sd[base:base + 4].astype(np.float32).copy()        # framequat wxyz
        gyro = sd[base + 4:base + 7].astype(np.float32).copy()
        return StateSnapshot(q=q, dq=dq, quat_wxyz=quat, gyro=gyro)

    def apply(self, cmd):
        sd = self.data.sensordata
        q = sd[0:self.nu]
        dq = sd[self.nu:2 * self.nu]
        self.data.ctrl[:] = (cmd.tau_ff
                             + cmd.kp * (cmd.q_des - q)
                             + cmd.kd * (cmd.dq_des - dq))

    def step(self):
        self.mujoco.mj_step(self.model, self.data)

    # measurements
    def base_pos(self):
        return self.data.qpos[0:3].copy()

    def base_planar_speed(self):
        return float(np.linalg.norm(self.data.qvel[0:2]))


def run_span(sim, ctl, seconds, dt, min_height_track=None):
    """Advance sim+controller; returns (min_height, fell)."""
    n = int(round(seconds / dt))
    min_h = np.inf
    for _ in range(n):
        cmd = ctl.update(dt, sim.snapshot())
        if cmd is not None:
            sim.apply(cmd)
        sim.step()
        h = sim.base_pos()[2]
        min_h = min(min_h, h)
        if min_height_track is not None:
            min_height_track.append(h)
    return min_h, min_h <= 0.55


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dt", type=float, default=0.002,
                    help="physics dt (0.002 recommended; see module docstring)")
    args = ap.parse_args()

    print(f"[a4_inproc] scene={SCENE}")
    sim = InprocSim(dt=args.dt)
    ctl = LocoController()
    dt = args.dt
    results = {}
    t_wall = time.time()

    # startup: ramp (2 s) + policy settle at cmd 0 (1 s)
    min_h, fell = run_span(sim, ctl, C.STARTUP_RAMP_DURATION_S + 1.0, dt)
    print(f"[a4_inproc] startup done: phase={ctl.phase} "
          f"h={sim.base_pos()[2]:.3f} min_h={min_h:.3f}")
    if ctl.phase != "POLICY" or fell:
        print("[a4_inproc] FAIL: did not reach POLICY standing")
        return 1

    # (i) STAND 30 s
    min_h, fell = run_span(sim, ctl, 30.0, dt)
    stand_ok = not fell
    results["stand"] = f"min pelvis height {min_h:.3f} m over 30 s (>0.55 required)"
    print(f"[a4_inproc] STAND: {'PASS' if stand_ok else 'FAIL'} - {results['stand']}")

    # (ii) WALK 0.3 m/s for 5 s
    p0 = sim.base_pos()
    ctl.handle_message({"type": "cmd_vel", "vx": 0.3, "vy": 0.0, "wz": 0.0})
    min_h, fell = run_span(sim, ctl, 5.0, dt)
    p1 = sim.base_pos()
    disp = float(np.linalg.norm((p1 - p0)[0:2]))
    walk_ok = (0.7 <= disp <= 1.95) and not fell
    results["walk"] = (f"displacement {disp:.3f} m in 5 s at cmd 0.3 m/s "
                       f"(band [0.7, 1.95]), min height {min_h:.3f} m")
    print(f"[a4_inproc] WALK : {'PASS' if walk_ok else 'FAIL'} - {results['walk']}")

    # (iii) STOP
    ctl.handle_message({"type": "stop"})
    speeds = []
    n = int(round(1.5 / dt))
    t_below = None
    for k in range(n):
        cmd = ctl.update(dt, sim.snapshot())
        if cmd is not None:
            sim.apply(cmd)
        sim.step()
        v = sim.base_planar_speed()
        speeds.append(v)
        if t_below is None and v < 0.05:
            t_below = (k + 1) * dt
    # must be below 0.05 at the 1.5 s mark and still standing 3 s later
    v_end = speeds[-1]
    min_h, fell = run_span(sim, ctl, 3.0, dt)
    stop_ok = (t_below is not None) and (v_end < 0.05) and not fell
    results["stop"] = (f"speed<0.05 m/s first at {t_below if t_below else float('nan'):.2f} s "
                       f"(<=1.5 required), v(1.5s)={v_end:.3f} m/s, "
                       f"still standing 3 s later (min h {min_h:.3f} m)")
    print(f"[a4_inproc] STOP : {'PASS' if stop_ok else 'FAIL'} - {results['stop']}")

    ok = stand_ok and walk_ok and stop_ok
    print(f"[a4_inproc] wall time {time.time()-t_wall:.1f} s | "
          f"overall: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
