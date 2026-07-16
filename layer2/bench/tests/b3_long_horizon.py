#!/usr/bin/env python3
"""B3 long-horizon stability: in-process MuJoCo + real ONNX (no DDS/ZMQ).

The benchmark runs 100 turns (many minutes of sim time), mostly standing
between commands, so the runner must be stable far beyond the 30 s
acceptance span:

  (1) STAND ENDURANCE: cmd 0 for --stand-seconds (default 600 s = 10 min).
      PASS: never below 0.55 m pelvis height, total planar drift < 1.0 m,
      windowed speed at the end < 0.05 m/s, |yaw drift| < 30 deg.
  (2) WALK/STOP CYCLES: --cycles (default 30) x [cmd 0.3 for 3 s; stop;
      2 s settle]. PASS: no fall, every cycle halts (< 0.05 m/s before the
      next command), per-cycle displacement does not degrade (last-5 mean
      within 25% of first-5 mean), |heading drift| < 45 deg overall.

    python3 bench/tests/b3_long_horizon.py [--quick]

Wall time: ~17x real time on a desktop CPU (10 min sim ~ 40 s wall).
"""
import argparse
import sys

import numpy as np

from b3_common import make_stack, run_metered, Telemetry  # noqa: E402

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


def stand_endurance(dt: float, seconds: float) -> bool:
    print(f"== (1) stand endurance: {seconds:.0f} s at cmd 0 ==")
    sim, ctl, tel = make_stack(dt=dt)
    p0 = sim.base_pos()[0:2].copy()
    yaw0 = tel.yaw()
    max_drift = 0.0
    marks = []

    minute = max(1.0, seconds / 10.0)
    n_chunks = int(round(seconds / minute))
    t = 0.0
    for _ in range(n_chunks):
        run_metered(sim, ctl, tel, minute, dt)
        t += minute
        p = sim.base_pos()[0:2]
        drift = float(np.linalg.norm(p - p0))
        max_drift = max(max_drift, drift)
        marks.append((t, drift, tel.windowed_speed(),
                      np.degrees(tel.yaw() - yaw0), tel.min_h))
        print(f"    t={t:6.0f}s drift={drift:.3f} m  wspeed={marks[-1][2]:.3f} m/s"
              f"  yaw_drift={marks[-1][3]:+6.1f} deg  min_h={tel.min_h:.3f} m")

    drift_end = marks[-1][1]
    yaw_drift = marks[-1][3]
    check(f"no fall over {seconds:.0f} s (min h {tel.min_h:.3f} m)", not tel.fell)
    check(f"total planar drift {drift_end:.3f} m < 1.0 m", drift_end < 1.0)
    check(f"max planar drift {max_drift:.3f} m < 1.0 m", max_drift < 1.0)
    check(f"windowed speed at end {marks[-1][2]:.3f} < 0.05 m/s",
          marks[-1][2] < 0.05)
    check(f"yaw drift {yaw_drift:+.1f} deg within +/-30",
          abs(yaw_drift) < 30.0)
    print(f"    drift rate: {drift_end / seconds * 60.0:.4f} m/min")
    return not tel.fell


def walk_stop_cycles(dt: float, cycles: int) -> None:
    print(f"== (2) walk/stop cycles x{cycles} (3 s @ 0.3 m/s + stop + 2 s) ==")
    sim, ctl, tel = make_stack(dt=dt)
    yaw0 = tel.yaw()
    disps, halt_times, residuals = [], [], []

    for i in range(cycles):
        p0 = sim.base_pos()[0:2].copy()
        ctl.handle_message({"type": "cmd_vel", "vx": 0.3, "vy": 0.0, "wz": 0.0})
        run_metered(sim, ctl, tel, 3.0, dt)
        ctl.handle_message({"type": "stop"})
        halt_t = [None]

        def watch_halt(_k, t, _h=halt_t, _tel=tel):
            if _h[0] is None and _tel.windowed_speed() < 0.05:
                _h[0] = t

        run_metered(sim, ctl, tel, 2.0, dt, on_step=watch_halt)
        d = float(np.linalg.norm(sim.base_pos()[0:2] - p0))
        disps.append(d)
        halt_times.append(halt_t[0])
        residuals.append(tel.windowed_speed())
        if tel.fell:
            print(f"    FELL in cycle {i + 1}")
            break

    n = len(disps)
    yaw_drift = np.degrees(tel.yaw() - yaw0)
    halted = [h for h in halt_times if h is not None]
    print(f"    cycles completed: {n}/{cycles}, min h {tel.min_h:.3f} m")
    print(f"    per-cycle displacement: mean {np.mean(disps):.3f} m  "
          f"min {np.min(disps):.3f}  max {np.max(disps):.3f}")
    print(f"    halt time after stop: mean {np.mean(halted):.2f} s  "
          f"max {np.max(halted):.2f} s  (halted {len(halted)}/{n})")
    print(f"    residual windowed speed at cycle end: max {np.max(residuals):.3f} m/s")
    print(f"    heading drift over all cycles: {yaw_drift:+.1f} deg")

    check(f"all {cycles} cycles completed without fall", n == cycles and not tel.fell)
    check("every cycle halted (<0.05 m/s within the 2 s settle)",
          len(halted) == n and all(r < 0.05 for r in residuals),
          (len(halted), max(residuals)))
    first5, last5 = float(np.mean(disps[:5])), float(np.mean(disps[-5:]))
    check(f"no gait degradation: last-5 mean {last5:.3f} m within 25% of "
          f"first-5 mean {first5:.3f} m",
          abs(last5 - first5) <= 0.25 * first5)
    check(f"heading drift {yaw_drift:+.1f} deg within +/-45",
          abs(yaw_drift) < 45.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dt", type=float, default=0.002)
    ap.add_argument("--stand-seconds", type=float, default=600.0)
    ap.add_argument("--cycles", type=int, default=30)
    ap.add_argument("--quick", action="store_true",
                    help="60 s stand + 5 cycles (smoke)")
    args = ap.parse_args()
    if args.quick:
        args.stand_seconds, args.cycles = 60.0, 5

    import time
    t0 = time.time()
    stand_endurance(args.dt, args.stand_seconds)
    walk_stop_cycles(args.dt, args.cycles)
    print(f"\nB3 long-horizon: {PASS} passed, {FAIL} failed "
          f"(wall {time.time() - t0:.0f} s)")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
