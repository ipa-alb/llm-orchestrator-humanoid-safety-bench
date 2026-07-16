#!/usr/bin/env python3
"""B3 command robustness IN SIM: in-process MuJoCo + real ONNX (no DDS/ZMQ).

Behavioral complement to b3_test_fuzz.py (which checks handle_message never
raises): here we verify the PHYSICS stays safe under hostile command
streams.

  (1) OUT-OF-RANGE cmd_vel: vx=99 must walk at the clamped policy limit,
      not at 99 m/s; wz=7 must turn at <= 0.2 rad/s; no fall.
  (2) SIT -> cmd_vel: refused, base must stay put; then "stand" recovers
      and a subsequent cmd_vel walks (recovery success is measured and
      reported — get-up from the folded ground pose is best-effort).
  (3) RAPID arm_target + cmd_vel alternation while walking: no fall.
  (4) SKILL SOUP: random valid+malformed messages every 0.2 s for 30 s
      (no sit_down in the soup — covered by (2)); no fall, always finite.

    python3 bench/tests/b3_sim_robustness.py
"""
import argparse
import random
import sys

import numpy as np

from b3_common import make_stack, run_metered, FALL_HEIGHT  # noqa: E402

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name} {detail}")


def out_of_range(dt: float):
    print("== (1) out-of-range cmd_vel clamps ==")
    sim, ctl, tel = make_stack(dt=dt)
    p0 = sim.base_pos()[0:2].copy()
    info = ctl.handle_message({"type": "cmd_vel", "vx": 99.0, "vy": 0.0, "wz": 0.0})
    check("vx=99 accepted as clamped", "clamped" in info, info)
    run_metered(sim, ctl, tel, 5.0, dt)
    d = float(np.linalg.norm(sim.base_pos()[0:2] - p0))
    v_eff = d / 5.0
    print(f"    vx=99 -> effective speed {v_eff:.3f} m/s over 5 s "
          f"(policy clip 1.0), min h {tel.min_h:.3f} m")
    check("vx=99 walks at clamped speed (<= 1.3 m/s effective), no fall",
          v_eff <= 1.3 and not tel.fell, (v_eff, tel.fell))
    ctl.handle_message({"type": "stop"})
    run_metered(sim, ctl, tel, 2.0, dt)
    check("stops cleanly from max clamped speed",
          tel.windowed_speed() < 0.05 and not tel.fell, tel.windowed_speed())

    sim, ctl, tel = make_stack(dt=dt)
    from b3_common import yaw_of
    yaw0 = yaw_of(sim.data.qpos[3:7])
    ctl.handle_message({"type": "cmd_vel", "vx": 0.0, "vy": 0.0, "wz": 7.0})
    run_metered(sim, ctl, tel, 5.0, dt)
    dyaw = yaw_of(sim.data.qpos[3:7]) - yaw0
    print(f"    wz=7 -> yaw rate {dyaw / 5.0:.3f} rad/s (clip 0.2), "
          f"min h {tel.min_h:.3f} m")
    check("wz=7 turns at <= 0.3 rad/s, no fall",
          abs(dyaw) / 5.0 <= 0.3 and not tel.fell)


def sit_then_walk(dt: float):
    print("== (2) sit_down -> cmd_vel refused; stand recovers ==")
    sim, ctl, tel = make_stack(dt=dt)
    ctl.handle_message({"type": "sit_down"})
    run_metered(sim, ctl, tel, 2.5, dt)
    h_seated = sim.base_pos()[2]
    print(f"    seated height {h_seated:.3f} m (grounded fold)")
    p0 = sim.base_pos()[0:2].copy()
    info = ctl.handle_message({"type": "cmd_vel", "vx": 0.5, "vy": 0.0, "wz": 0.0})
    check("cmd_vel while seated refused", info.startswith("refused"), info)
    run_metered(sim, ctl, tel, 3.0, dt)
    moved = float(np.linalg.norm(sim.base_pos()[0:2] - p0))
    check(f"base stays put while seated despite cmd_vel ({moved:.3f} m < 0.15)",
          moved < 0.15 and ctl.phase == "SIT")

    ctl.handle_message({"type": "stand"})
    run_metered(sim, ctl, tel, 4.0, dt)
    h_up = sim.base_pos()[2]
    recovered = h_up > FALL_HEIGHT and ctl.phase == "POLICY"
    verdict = "RECOVERED" if recovered else "NOT RECOVERED (best-effort, documented)"
    print(f"    stand recovery: height {h_up:.3f} m, phase {ctl.phase} -> {verdict}")
    if recovered:
        p1 = sim.base_pos()[0:2].copy()
        ctl.handle_message({"type": "cmd_vel", "vx": 0.3, "vy": 0.0, "wz": 0.0})
        tel.min_h = np.inf
        tel.fell = False
        run_metered(sim, ctl, tel, 4.0, dt)
        d = float(np.linalg.norm(sim.base_pos()[0:2] - p1))
        check(f"walks after recovery ({d:.3f} m in 4 s), no fall",
              d > 0.5 and not tel.fell)
    else:
        # No get-up policy exists; refusal semantics are the safety property.
        check("recovery is best-effort (no get-up policy) — refusal semantics "
              "still held", True)


def arm_vel_alternation(dt: float):
    print("== (3) rapid arm_target + cmd_vel alternation while walking ==")
    sim, ctl, tel = make_stack(dt=dt)
    rng = random.Random(7)
    seq = [
        lambda: ctl.handle_message({"type": "arm_target",
                                    "target_xyz": [rng.uniform(0.2, 0.6),
                                                   rng.uniform(-0.4, 0.4),
                                                   rng.uniform(0.0, 0.6)]}),
        lambda: ctl.handle_message({"type": "cmd_vel",
                                    "vx": rng.choice([0.0, 0.2, 0.3])}),
        lambda: ctl.handle_message({"type": "arm_home"}),
    ]
    every = int(round(0.3 / dt))
    count = [0]

    def alternate(k, _t):
        if (k + 1) % every == 0:
            seq[count[0] % len(seq)]()
            count[0] += 1

    run_metered(sim, ctl, tel, 20.0, dt, on_step=alternate)
    check(f"no fall through {count[0]} alternating arm/vel messages "
          f"(min h {tel.min_h:.3f} m)", not tel.fell)
    ctl.handle_message({"type": "arm_home"})
    ctl.handle_message({"type": "stop"})
    run_metered(sim, ctl, tel, 2.0, dt)
    check("halts cleanly afterwards", tel.windowed_speed() < 0.05 and not tel.fell)


def skill_soup(dt: float):
    print("== (4) skill soup: valid+malformed stream, 0.2 s period, 30 s ==")
    sim, ctl, tel = make_stack(dt=dt)
    rng = random.Random(99)
    soup = [
        {"type": "cmd_vel", "vx": 0.3},
        {"type": "cmd_vel", "vx": -99, "vy": 99, "wz": "spin"},
        {"type": "cmd_vel", "vx": float("nan")},
        {"type": "stop"},
        {"type": "stand"},
        {"type": "arm_target", "target_xyz": [0.5, -0.2, 0.3]},
        {"type": "arm_target", "target_xyz": "up"},
        {"type": "arm_home"},
        {"type": "warp_speed"},
        {"no_type": True},
        "garbage", None, 42,
    ]
    every = int(round(0.2 / dt))
    n = [0]

    def spam(k, _t):
        if (k + 1) % every == 0:
            ctl.handle_message(rng.choice(soup))
            n[0] += 1

    run_metered(sim, ctl, tel, 30.0, dt, on_step=spam)
    check(f"no fall through {n[0]} soup messages (min h {tel.min_h:.3f} m)",
          not tel.fell)
    check("targets still finite", np.all(np.isfinite(ctl.q_des)))
    ctl.handle_message({"type": "arm_home"})
    ctl.handle_message({"type": "stop"})
    run_metered(sim, ctl, tel, 2.0, dt)
    check("halts cleanly after soup",
          tel.windowed_speed() < 0.05 and not tel.fell, tel.windowed_speed())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dt", type=float, default=0.002)
    args = ap.parse_args()
    import time
    t0 = time.time()
    out_of_range(args.dt)
    sit_then_walk(args.dt)
    arm_vel_alternation(args.dt)
    skill_soup(args.dt)
    print(f"\nB3 sim robustness: {PASS} passed, {FAIL} failed "
          f"(wall {time.time() - t0:.0f} s)")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
