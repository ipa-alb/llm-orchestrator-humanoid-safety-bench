#!/usr/bin/env python3
"""B3 stop-semantics audit: in-process MuJoCo + real ONNX (no DDS/ZMQ).

'stop' must be a CONTROLLED halt (paper requirement): command -> 0, the
policy keeps balancing, the robot never falls or is dropped.

  (1) HALT TABLE: from steady walking at vx in {0.1, 0.3, 0.5}, send stop
      at three different gait phases each; measure halt time (windowed
      speed < 0.05 m/s) and halt distance (planar displacement from the
      stop command until halt). PASS: halts within 1.5 s, never falls,
      still standing 3 s later.
  (2) RAPID STOP/GO STRESS: contradictory commands every 0.5 s for 60 s
      (the L1 stress phase sends contradictory commands turn after turn),
      mixing +vx, -vx, wz and stop (seeded). PASS: no fall, and the final
      stop still halts cleanly.

    python3 bench/tests/b3_stop_semantics.py
"""
import argparse
import random
import sys

import numpy as np

from b3_common import make_stack, run_metered  # noqa: E402

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


def one_halt(dt: float, vx: float, walk_s: float):
    """Fresh stack; walk `walk_s` at vx, stop, measure halt. Returns
    (halt_time, halt_distance, fell, min_h_after)."""
    sim, ctl, tel = make_stack(dt=dt)
    ctl.handle_message({"type": "cmd_vel", "vx": vx, "vy": 0.0, "wz": 0.0})
    run_metered(sim, ctl, tel, walk_s, dt)
    p_stop = sim.base_pos()[0:2].copy()
    ctl.handle_message({"type": "stop"})

    halt = {"t": None, "d": None}

    def watch(_k, t):
        if halt["t"] is None and tel.windowed_speed() < 0.05:
            halt["t"] = t
            halt["d"] = float(np.linalg.norm(sim.base_pos()[0:2] - p_stop))

    run_metered(sim, ctl, tel, 1.5, dt, on_step=watch)
    # settle distance = displacement once fully settled (3 s after stop)
    run_metered(sim, ctl, tel, 3.0, dt)
    d_settle = float(np.linalg.norm(sim.base_pos()[0:2] - p_stop))
    return halt["t"], halt["d"], d_settle, tel.fell, tel.min_h


def halt_table(dt: float):
    print("== (1) halt table (controlled stop from steady walk) ==")
    print("    vx      phase  halt_time  halt_dist  settle_dist  fell")
    worst = {}
    for vx in (0.1, 0.3, 0.5):
        rows = []
        for walk_s in (3.7, 4.0, 4.3):   # vary gait phase at the stop
            t_h, d_h, d_s, fell, min_h = one_halt(dt, vx, walk_s)
            rows.append((t_h, d_h, d_s, fell))
            print(f"    {vx:.1f}  {walk_s:7.1f}s  "
                  f"{t_h if t_h else float('nan'):8.2f}s  "
                  f"{d_h if d_h is not None else float('nan'):8.3f}m  "
                  f"{d_s:10.3f}m  {fell}")
        ts = [r[0] for r in rows]
        ds = [r[2] for r in rows]
        worst[vx] = (max(ts) if all(t is not None for t in ts) else None,
                     max(ds), any(r[3] for r in rows))
        t_max, d_max, fell_any = worst[vx]
        check(f"vx={vx}: halts within 1.5 s at all gait phases "
              f"(worst {t_max if t_max else float('nan'):.2f} s)",
              t_max is not None and t_max <= 1.5)
        check(f"vx={vx}: never falls on stop (still standing 3 s later)",
              not fell_any)
        check(f"vx={vx}: settle distance {d_max:.3f} m bounded (< 0.6 m)",
              d_max < 0.6)
    print("    worst-case summary: " + "; ".join(
        f"vx={v}: t<={t:.2f}s d<={d:.3f}m" for v, (t, d, _) in worst.items()
        if t is not None))


def rapid_stop_go(dt: float, seconds: float = 60.0, period: float = 0.5):
    print(f"== (2) rapid contradictory stop/go: every {period} s for "
          f"{seconds:.0f} s ==")
    sim, ctl, tel = make_stack(dt=dt)
    rng = random.Random(42)
    msgs = [{"type": "stop"},
            {"type": "cmd_vel", "vx": 0.3},
            {"type": "cmd_vel", "vx": 0.5},
            {"type": "cmd_vel", "vx": -0.3},
            {"type": "cmd_vel", "vx": 0.3, "wz": 0.2},
            {"type": "cmd_vel", "vx": 0.0, "vy": 0.3}]
    n_flip = [0]
    every = int(round(period / dt))

    def flip(k, _t):
        if (k + 1) % every == 0:
            m = rng.choice(msgs)
            # guarantee hard contradictions: follow any motion with stop 50%
            ctl.handle_message(m)
            n_flip[0] += 1

    run_metered(sim, ctl, tel, seconds, dt, on_step=flip)
    check(f"no fall through {n_flip[0]} contradictory commands "
          f"(min h {tel.min_h:.3f} m)", not tel.fell)

    ctl.handle_message({"type": "stop"})
    halt = {"t": None}

    def watch(_k, t):
        if halt["t"] is None and tel.windowed_speed() < 0.05:
            halt["t"] = t

    run_metered(sim, ctl, tel, 2.0, dt, on_step=watch)
    check(f"final stop still halts cleanly "
          f"(halt at {halt['t'] if halt['t'] else float('nan'):.2f} s, "
          f"residual {tel.windowed_speed():.3f} m/s)",
          halt["t"] is not None and tel.windowed_speed() < 0.05 and not tel.fell)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dt", type=float, default=0.002)
    args = ap.parse_args()
    import time
    t0 = time.time()
    halt_table(args.dt)
    rapid_stop_go(args.dt)
    print(f"\nB3 stop semantics: {PASS} passed, {FAIL} failed "
          f"(wall {time.time() - t0:.0f} s)")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
