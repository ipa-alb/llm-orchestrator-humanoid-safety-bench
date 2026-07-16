#!/usr/bin/env python3
"""Closed-loop benchmark driver (R2): canonical schedule x mock policy x
MuJoCoBackend, with per-turn world verification.

Runs inside a g1-base container against the compose stack (sim + loco
[+ battery]), e.g.:

    docker compose -f docker/compose.yaml run --rm bench \
        python3 scripts/closed_loop_driver.py --policy compliant

Per turn it (1) applies the schedule's worldctl ops, (2) verifies via
worldctl get_state that the sim world realized the scheduled environment and
that the S1-S5 triggers derived from the *measured* world match the
schedule's expected triggers, (3) dwells --dwell seconds of sim time,
(4) runs the mock policy, whose tool calls hit the turn-keyed env AND are
mirrored into the sim (cmd_vel / arm_target / stop / sit_down over ZMQ ->
loco -> rt/lowcmd), and (5) logs the L1-superset JSONL.

Sensor semantics for the canonical run: deterministic env-truth
(sensor_source="env", sim measurement noise pinned to --noise-sigma,
default 0). Noise stays available: pass --noise-sigma 0.02 and/or
--sensor-source backend for noisy worldstate-sourced sensors.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "bench"))

from g1_safety_bench.backend.mujoco_zmq import MuJoCoBackend  # noqa: E402
from g1_safety_bench.logspine.episode_logger import EpisodeLogger  # noqa: E402
from g1_safety_bench.scenario.mock_llm import CompliantPolicy, ViolatorPolicy  # noqa: E402
from g1_safety_bench.scenario.runner_sim import SimulationRunner  # noqa: E402
from g1_safety_bench.scenario.schedule import ScenarioSchedule  # noqa: E402

POLICIES = {"compliant": CompliantPolicy, "violator": ViolatorPolicy}
STAND_MIN_BASE_Z = 0.55   # pelvis height when standing is ~0.79 m; fallen < 0.4


def wait_for_sim(backend: MuJoCoBackend, min_sim_time: float,
                 timeout_s: float) -> dict:
    """Wait until the sim publishes worldstate, sim_time >= min_sim_time
    (gives the loco runner time to engage and stand) and the robot base is
    upright. Returns the last snapshot."""
    deadline = time.monotonic() + timeout_s
    snap = None
    while time.monotonic() < deadline:
        try:
            snap = backend.world_snapshot()
            st = backend._snapshot().get("sim_time", 0.0)  # PUB-side sim time
            if st >= min_sim_time:
                z = float(snap["robot_pos"][2])
                if z < STAND_MIN_BASE_Z:
                    raise SystemExit(
                        f"robot base z={z:.2f} m at sim_time={st:.1f}s — not "
                        "standing (did the loco runner engage within ~2 s of "
                        "sim start?)")
                return snap
        except (TimeoutError, RuntimeError):
            pass
        time.sleep(0.5)
    raise SystemExit(f"sim not ready within {timeout_s}s "
                     f"(last snapshot: {snap and 'ok' or 'none'})")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--policy", choices=sorted(POLICIES), default="compliant")
    ap.add_argument("--turns", type=int, default=None,
                    help="run only the first N turns (default: all 100)")
    ap.add_argument("--dwell", type=float, default=2.0,
                    help="sim-time seconds per turn (default 2.0)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--noise-sigma", type=float, default=0.0,
                    help="sim human_distance noise sigma (default 0 = "
                         "deterministic env-truth run)")
    ap.add_argument("--sensor-source", choices=["env", "backend"],
                    default="env",
                    help="'env': turn-keyed L1 semantics (canonical); "
                         "'backend': sensors read from sim worldstate")
    ap.add_argument("--out-dir", default=str(REPO / "bench" / "runs"))
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--safety-version", default="v2",
                    help="label recorded in the log (mock policies do not "
                         "consume a safety doc)")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--wait-sim-time", type=float, default=3.0)
    ap.add_argument("--wait-timeout", type=float, default=90.0)
    args = ap.parse_args(argv)

    backend = MuJoCoBackend()
    snap = wait_for_sim(backend, args.wait_sim_time, args.wait_timeout)
    print(f"[closed_loop] sim ready: robot_pos={snap['robot_pos']}, "
          f"battery={snap['battery_level']}", flush=True)

    sched = ScenarioSchedule.canonical()
    run_id = args.run_id or time.strftime("closedloop_%Y%m%d_%H%M%S")
    run_id = f"{run_id}_{args.policy}" if args.policy not in run_id else run_id
    logger = EpisodeLogger(
        args.out_dir,
        safety_version=args.safety_version,
        run_id=run_id,
        backend="mujoco",
        scene="scenes/scene_bench.xml",
        seed=args.seed,
        scenario_name=sched.name,
        extra_meta={"policy": args.policy, "dwell_s": args.dwell,
                    "sensor_source": args.sensor_source,
                    "noise_sigma": args.noise_sigma},
    )
    runner = SimulationRunner(
        sched,
        POLICIES[args.policy](),
        logger,
        backend=backend,
        seed=args.seed,
        dt_per_turn=args.dwell,
        sensor_source=args.sensor_source,
        trace_state=True,
        max_turns=args.turns,
        initial_env={"sensor_noise_sigma": args.noise_sigma},
        verify_world=not args.no_verify,
    )

    t0 = time.time()
    try:
        path = runner.run()
    finally:
        # Controlled halt: the last commanded cmd_vel would otherwise persist
        # in the loco runner after the episode ends.
        backend.apply_skill("stop")
    dt = time.time() - t0
    print(f"[closed_loop] {args.policy}: "
          f"{args.turns or len(sched)} turns in {dt:.0f}s -> {path}", flush=True)

    if not args.no_verify:
        summary = runner.verification_summary()
        print("[closed_loop] world verification: "
              f"{summary['turns_ok']}/{summary['turns_checked']} turns OK",
              flush=True)
        for c in summary["failed"]:
            print(f"  turn {c['turn']}: {c['mismatches']}", flush=True)
        # Machine-readable trigger census (which invariants fired in-world)
        fired: dict[str, int] = {}
        for c in runner.world_checks:
            for t in c.get("derived_triggers", []):
                fired[t] = fired.get(t, 0) + 1
        print(f"[closed_loop] invariants fired in sim world: "
              f"{json.dumps(dict(sorted(fired.items())))}", flush=True)
        if summary["failed"]:
            return 1
    backend.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
