#!/usr/bin/env python3
"""One benchmark episode: canonical schedule x {mock policy | real LLM} x
MuJoCoBackend (agent C1). The real-LLM variant of scripts/closed_loop_driver.py
— same stack contract (fresh compose stack per episode, sim ready-wait, world
verification, controlled stop in finally), but the policy slot can be a real
LLM driver (g1_safety_bench/scenario/llm_driver.py).

Runs inside a g1-base container against the compose stack, e.g.:

    docker compose -f docker/compose.yaml run --rm bench \
        python3 scripts/llm_episode_driver.py --backend-llm mock-compliant
    docker compose -f docker/compose.yaml run --rm bench \
        bash -c "pip3 install -q anthropic openai;
                 python3 scripts/llm_episode_driver.py --backend-llm claude"

Exit codes: 0 = episode completed (safety violations by the model are DATA,
not failure); 3 = infrastructure failure (sim not ready, worldctl timeout,
API auth/exhausted retries, missing L1 checkout) — the campaign orchestrator
retries these once; 4 = world verification mismatch.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "bench"))
sys.path.insert(0, str(REPO / "scripts"))

from g1_safety_bench.backend.mujoco_zmq import MuJoCoBackend  # noqa: E402
from g1_safety_bench.logspine.episode_logger import EpisodeLogger  # noqa: E402
from g1_safety_bench.scenario.llm_driver import (  # noqa: E402
    BACKENDS, LLMInfraError, make_backend_driver)
from g1_safety_bench.scenario.mock_llm import (  # noqa: E402
    CompliantPolicy, ViolatorPolicy)
from g1_safety_bench.scenario.runner_sim import (  # noqa: E402
    SimBreakError, SimulationRunner)
from g1_safety_bench.scenario.schedule import ScenarioSchedule  # noqa: E402

from closed_loop_driver import wait_for_sim  # noqa: E402  (same scripts/ dir)

MOCKS = {"mock-compliant": CompliantPolicy, "mock-violator": ViolatorPolicy}
EXIT_INFRA = 3
EXIT_VERIFY = 4
EXIT_SIMBREAK = 5  # physics instability / robot fall (SimBreakError)


def make_policy(args):
    if args.backend_llm in MOCKS:
        return MOCKS[args.backend_llm]()
    return make_backend_driver(
        args.backend_llm,
        model=args.model,
        safety_version=args.safety_version,
        budget=args.budget,
        verbose=not args.quiet,
        reasoning_effort=args.reasoning_effort,
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--backend-llm", required=True,
                    choices=sorted(MOCKS) + sorted(BACKENDS),
                    help="mock-compliant / mock-violator / claude / gpt / gemini")
    ap.add_argument("--model", default=None,
                    help="override the backend's default model id")
    ap.add_argument("--motion-mode", choices=["continuous", "autostop"],
                    default="continuous",
                    help="continuous: a velocity command persists until the "
                         "next command, so the robot keeps moving while the "
                         "LLM thinks (realistic). autostop: the base is "
                         "physically zeroed --dwell seconds after each move "
                         "command, bounding per-command travel to L1's "
                         "nominal step")
    ap.add_argument("--reasoning-effort", default=None,
                    choices=["none", "minimal", "low", "medium", "high"],
                    help="cap the model's thinking (OpenAI-compatible "
                         "backends only, i.e. gemini/local; e.g. 'low' "
                         "shortens Gemini's thinking phase)")
    ap.add_argument("--safety-version", default="v2",
                    help="L1 safety doc version fed as system prompt (default v2)")
    ap.add_argument("--budget", action="store_true",
                    help="L1 budget condition: sliding window + env "
                         "change-detection + Anthropic prompt caching "
                         "(default: no-budget / full history)")
    ap.add_argument("--turns", type=int, default=None,
                    help="run only the first N turns (default: all 100)")
    ap.add_argument("--dwell", type=float, default=2.0,
                    help="sim-time seconds per turn (default 2.0)")
    ap.add_argument("--condition", choices=["deterministic", "noisy"],
                    default="deterministic",
                    help="deterministic: env-truth sensors, sim noise pinned "
                         "to 0 (canonical); noisy: backend-sourced sensors "
                         "with --noise-sigma")
    ap.add_argument("--noise-sigma", type=float, default=0.02,
                    help="sim human_distance noise sigma for --condition noisy")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-dir", default=str(REPO / "bench" / "runs"))
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--on-sim-break", choices=["abort", "continue"],
                    default="abort",
                    help="physics instability / robot fall during the "
                         "episode: abort pre-LLM-call with exit code 5 "
                         "(default; campaign retries once) or record per "
                         "turn and continue")
    ap.add_argument("--wait-sim-time", type=float, default=3.0)
    ap.add_argument("--wait-timeout", type=float, default=90.0)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    if args.condition == "deterministic":
        sensor_source, noise_sigma = "env", 0.0
    else:
        sensor_source, noise_sigma = "backend", args.noise_sigma

    # Build the policy BEFORE touching the stack so config errors (missing
    # API key, missing L1 checkout) fail fast as infra failures.
    try:
        policy = make_policy(args)
    except (LLMInfraError, ValueError) as e:
        print(f"[llm_episode] INFRA: {e}", file=sys.stderr)
        return EXIT_INFRA

    backend = MuJoCoBackend(
        motion_timeout_s=args.dwell if args.motion_mode == "autostop" else None)
    try:
        snap = wait_for_sim(backend, args.wait_sim_time, args.wait_timeout)
    except SystemExit as e:
        print(f"[llm_episode] INFRA: sim not ready: {e}", file=sys.stderr)
        return EXIT_INFRA
    print(f"[llm_episode] sim ready: robot_pos={snap['robot_pos']}, "
          f"battery={snap['battery_level']}", flush=True)

    sched = ScenarioSchedule.canonical()
    run_id = args.run_id or time.strftime(
        f"llm_{args.backend_llm}_%Y%m%d_%H%M%S")
    logger = EpisodeLogger(
        args.out_dir,
        safety_version=args.safety_version,
        run_id=run_id,
        backend="mujoco",
        scene="scenes/scene_bench.xml",
        seed=args.seed,
        scenario_name=sched.name,
        extra_meta={
            "layer": "L2",  # MuJoCo sim (L1 = text-only, L3 = physical G1)
            "policy": args.backend_llm,
            "llm_model": getattr(policy, "model", None),
            "llm_api": getattr(policy, "api", None),
            "condition": args.condition,
            "budget": args.budget,
            "anthropic_history_cache": getattr(policy, "history_caching",
                                               None),
            "motion_mode": args.motion_mode,
            "reasoning_effort": args.reasoning_effort,
            "dwell_s": args.dwell,
            "sensor_source": sensor_source,
            "noise_sigma": noise_sigma,
            "on_sim_break": args.on_sim_break,
        },
    )
    runner = SimulationRunner(
        sched,
        policy,
        logger,
        backend=backend,
        seed=args.seed,
        dt_per_turn=args.dwell,
        sensor_source=sensor_source,
        trace_state=True,
        max_turns=args.turns,
        initial_env={"sensor_noise_sigma": noise_sigma},
        verify_world=not args.no_verify,
        on_sim_break=args.on_sim_break,
    )

    t0 = time.time()
    try:
        path = runner.run()
    except SimBreakError as e:
        print(f"[llm_episode] SIM BREAK: {e}", file=sys.stderr)
        return EXIT_SIMBREAK
    except (LLMInfraError, TimeoutError, RuntimeError) as e:
        print(f"[llm_episode] INFRA: episode aborted: {type(e).__name__}: {e}",
              file=sys.stderr)
        return EXIT_INFRA
    finally:
        # Controlled halt: the last commanded cmd_vel would otherwise persist
        # in the loco runner after the episode ends.
        try:
            backend.apply_skill("stop")
        except Exception:
            pass
    dt = time.time() - t0
    print(f"[llm_episode] {args.backend_llm}: "
          f"{args.turns or len(sched)} turns in {dt:.0f}s -> {path}", flush=True)

    if getattr(policy, "total_usage", None) is not None:
        print("[llm_episode] episode usage: "
              + json.dumps({**policy.total_usage,
                            "api_calls": policy.total_api_calls,
                            "model": policy.model}), flush=True)

    breaks = runner.sim_break_summary()
    if breaks["count"]:
        # only reachable with --on-sim-break continue
        print(f"[llm_episode] SIM HEALTH: {breaks['count']} sim-break "
              f"turn(s) recorded (policy={breaks['policy']}) — treat this "
              "episode's data as suspect", flush=True)

    rc = 0
    if not args.no_verify:
        summary = runner.verification_summary()
        print("[llm_episode] world verification: "
              f"{summary['turns_ok']}/{summary['turns_checked']} turns OK",
              flush=True)
        for c in summary["failed"]:
            print(f"  turn {c['turn']}: {c['mismatches']}", flush=True)
        if summary["failed"]:
            rc = EXIT_VERIFY
    backend.close()
    return rc


if __name__ == "__main__":
    sys.exit(main())
