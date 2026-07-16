#!/usr/bin/env python3
"""Real-LLM campaign orchestrator (agent C1) — runs on the HOST.

For each (backend, rep) job: bring up a FRESH compose stack in a parallel
slot (isolated ZMQ ports + DDS domain, compose project g1bench_s<slot>), run
one 100-turn episode via scripts/llm_episode_driver.py inside the bench
container, tear the stack down, then run the UNCHANGED L1 analyzer
(degradation_test/experiment/analyzer.py, imported verbatim) on the produced
JSONL. Infra failures (sim not ready, API auth, exhausted retries, physics
instability / robot fall — episode exit codes 3/4/5) are retried once with a
fresh stack; safety violations by the model are DATA and are never retried.

Output tree:

    bench/runs/l2_campaign_<ts>/
        <backend>_rep<k>/run_v2_<...>.jsonl (+ .meta.json + .state.csv)
        <backend>_rep<k>/analysis.json
        campaign_summary.json
        campaign_summary.md

Examples (host, from the repo root):

    python3 scripts/campaign.py --preflight
    python3 scripts/campaign.py --backends mock-compliant,mock-violator \
        --reps 1 --parallel 2                       # end-to-end dry run
    python3 scripts/campaign.py --backends gpt --reps 1 --turns 10  # smoke
    python3 scripts/campaign.py --backends claude,gpt,gemini --reps 5 \
        --parallel 2                                # the full campaign

Requires: docker compose + images g1-base/g1-sim (docker/up.sh), the
degradation_test checkout as a sibling of humanoid_testing (or
G1_BENCH_L1_DIR / G1_BENCH_L1_HOST), host python3 with matplotlib for the
analyzer, and API keys in the environment for real backends.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue

REPO = Path(__file__).resolve().parents[1]
COMPOSE_FILE = REPO / "docker" / "compose.yaml"
sys.path.insert(0, str(REPO / "bench"))

from g1_safety_bench.scenario import l1_source  # noqa: E402
from g1_safety_bench.scenario.llm_driver import BACKENDS  # noqa: E402

MOCK_BACKENDS = ("mock-compliant", "mock-violator")
ALL_BACKENDS = tuple(sorted(MOCK_BACKENDS)) + tuple(sorted(BACKENDS))

# Slot -> env parametrization (see bench/g1_safety_bench/ports.py). Slot 0 is
# exactly the historical single-stack defaults.
SLOT_PORT_STRIDE = 20
SLOT_PORT_BASE0 = 5555
SLOT_DDS_BASE0 = 1
SLOT_MCP_BASE0 = 8765

EXIT_INFRA = 3
EXIT_VERIFY = 4
EXIT_SIMBREAK = 5  # physics instability / robot fall (llm_episode_driver)

# ---------------------------------------------------------------------------
# Cost estimation.
# PRICES ARE ESTIMATES, as of 2026-07-11, from the providers' public price
# lists (USD per 1M tokens). Re-verify before quoting costs anywhere formal:
#   Anthropic: claude-haiku-4-5  $1.00 in / $5.00 out (cache read ~0.1x in)
#   OpenAI:    gpt-4o-mini       $0.15 in / $0.60 out (cached in $0.075)
#   Google:    gemini-2.5-flash  $0.30 in / $2.50 out (cached in ~0.25x)
# ---------------------------------------------------------------------------
PRICE_TABLE_AS_OF = "2026-07-11"
PRICES_PER_MTOK = {
    # "cache_write" (Anthropic only, 1.25x input for the 5-min TTL) doubles
    # as the semantics marker: Anthropic usage reports input / cache_read /
    # cache_creation as DISJOINT counters, while OpenAI-compat prompt_tokens
    # INCLUDE the cached tokens. cost_estimate() branches on its presence.
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00, "cache_read": 0.10,
                         "cache_write": 1.25},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60, "cache_read": 0.075},
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50, "cache_read": 0.075},
    # gemini-2.5-flash is not served to new API accounts (404); 3.5-flash is
    # its replacement tier. Prices assumed equal to 2.5-flash — UNVERIFIED.
    "gemini-3.5-flash": {"input": 0.30, "output": 2.50, "cache_read": 0.075},
}

_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


# ------------------------------------------------------------------ helpers
def slot_env(slot: int) -> dict:
    env = dict(os.environ)
    env["BENCH_PORT_BASE"] = str(SLOT_PORT_BASE0 + SLOT_PORT_STRIDE * slot)
    env["BENCH_DDS_DOMAIN"] = str(SLOT_DDS_BASE0 + slot)
    env["BENCH_MCP_PORT"] = str(SLOT_MCP_BASE0 + slot)
    return env


def compose(project: str, args: list[str], env: dict,
            capture: bool = False) -> subprocess.CompletedProcess:
    cmd = ["docker", "compose", "-p", project, "-f", str(COMPOSE_FILE)] + args
    return subprocess.run(
        cmd, env=env, cwd=str(REPO),
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
        text=True)


def price_for(model: str | None) -> dict | None:
    if not model:
        return None
    for prefix, prices in PRICES_PER_MTOK.items():
        if model.startswith(prefix):
            return prices
    return None


def cost_estimate(model: str | None, usage: dict) -> float | None:
    prices = price_for(model)
    if prices is None:
        return None
    inp = usage.get("input_tokens", 0)
    out = usage.get("output_tokens", 0)
    cread = usage.get("cache_read_input_tokens", 0)
    ccreate = usage.get("cache_creation_input_tokens", 0)
    if "cache_write" in prices:
        # Anthropic: input_tokens EXCLUDE cached tokens; cache writes are
        # billed at 1.25x input.
        billed_input = (inp * prices["input"]
                        + ccreate * prices["cache_write"])
    else:
        # OpenAI-compat: prompt_tokens INCLUDE the cached portion; cache
        # writes are free.
        billed_input = (inp - min(cread, inp)) * prices["input"]
    return (billed_input
            + cread * prices["cache_read"]
            + out * prices["output"]) / 1e6


def p95(values: list[float]) -> float | None:
    if not values:
        return None
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round(0.95 * (len(s) - 1)))))
    return s[idx]


# ------------------------------------------------------------- episode run
def episode_command(args, backend: str, run_id: str, ctr_out_dir: str) -> str:
    ep = ["python3", "scripts/llm_episode_driver.py",
          "--backend-llm", backend,
          "--safety-version", args.safety_version,
          "--dwell", str(args.dwell),
          "--condition", args.condition,
          "--run-id", run_id,
          "--out-dir", ctr_out_dir]
    if args.turns is not None:
        ep += ["--turns", str(args.turns)]
    if args.model:
        ep += ["--model", args.model]
    if args.motion_mode != "continuous":
        ep += ["--motion-mode", args.motion_mode]
    if args.reasoning_effort:
        ep += ["--reasoning-effort", args.reasoning_effort]
    if args.budget:
        ep += ["--budget"]
    if args.on_sim_break != "abort":
        ep += ["--on-sim-break", args.on_sim_break]
    ep_str = " ".join(shlex.quote(x) for x in ep)
    if backend in BACKENDS:
        # pip-install the bench "llm" extra deps at container start (same
        # pattern as the compose bench service installing `mcp`).
        return ("python3 -c 'import anthropic, openai' 2>/dev/null || "
                "pip3 install -q anthropic openai; exec " + ep_str)
    return "exec " + ep_str


def run_episode(args, backend: str, rep: int, slot: int,
                out_root: Path) -> dict:
    """One job: fresh stack -> episode -> teardown (retry once on infra)."""
    name = f"{backend}_rep{rep}"
    run_dir = out_root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    ctr_out_dir = f"/ws/{run_dir.relative_to(REPO)}"
    project = f"g1bench_s{slot}"
    env = slot_env(slot)
    run_id = name
    result = {"backend": backend, "rep": rep, "slot": slot, "run_dir": str(run_dir),
              "run_id": run_id, "attempts": 0, "ok": False, "infra_errors": []}

    for attempt in range(1, args.retries + 2):
        result["attempts"] = attempt
        # The episode logger APPENDS to an existing JSONL; stale files from
        # a previous invocation OR A PREVIOUS ATTEMPT of this rep would
        # contaminate counts (the exact failure mode of L1's crashed-Gemini
        # artifact, and of gpt_rep4/5 in l2_campaign_20260714_180626 whose
        # aborted 52-turn attempts were scored into the arm aggregate).
        # Fresh attempt => fresh dir — this MUST run per attempt, not once.
        for stale in run_dir.iterdir():
            if stale.is_file():
                stale.unlink()
        log(f"[campaign] {name} (slot {slot}, attempt {attempt}): "
            "starting fresh stack ...")
        compose(project, ["down", "--remove-orphans"], env, capture=True)
        up = compose(project, ["up", "-d", "sim", "loco", "battery"], env,
                     capture=True)
        if up.returncode != 0:
            result["infra_errors"].append(f"compose up failed: {up.stdout[-2000:]}")
            compose(project, ["down", "--remove-orphans"], env, capture=True)
            continue
        try:
            cmd = episode_command(args, backend, run_id, ctr_out_dir)
            t0 = time.time()
            proc = compose(project, ["run", "--rm", "bench",
                                     "bash", "-c", cmd], env)
            wall = time.time() - t0
        finally:
            compose(project, ["down", "--remove-orphans"], env, capture=True)

        if proc.returncode == 0:
            result["ok"] = True
            result["wall_s"] = round(wall, 1)
            log(f"[campaign] {name}: episode OK in {wall:.0f}s")
            return result
        kind = {EXIT_INFRA: "infra", EXIT_VERIFY: "world-verify",
                EXIT_SIMBREAK: "sim-break"}.get(
            proc.returncode, f"exit {proc.returncode}")
        result["infra_errors"].append(
            f"attempt {attempt}: episode failed ({kind})")
        log(f"[campaign] {name}: episode FAILED ({kind}); "
            + ("retrying once with a fresh stack"
               if attempt <= args.retries else "giving up"))
        # NOTE: retries happen ONLY for infra failures (nonzero exit).
        # Safety violations by the model leave exit code 0 and are analyzed,
        # never re-rolled.
    return result


# ---------------------------------------------------------------- analysis
def analyze_run(result: dict, safety_version: str) -> dict | None:
    """UNCHANGED L1 analyzer on the run's JSONL + per-turn stats extraction."""
    run_dir = Path(result["run_dir"])
    jsonl = run_dir / f"run_{safety_version}_{result['run_id']}.jsonl"
    if not jsonl.is_file():
        result["infra_errors"].append(f"missing JSONL: {jsonl}")
        result["ok"] = False
        return None

    analyzer = l1_source.import_l1("analyzer")
    if analyzer is None:
        raise RuntimeError("L1 analyzer not importable on the host "
                           "(set G1_BENCH_L1_DIR / install matplotlib)")
    res = analyzer.analyze(str(jsonl), save_plots=False)

    # Per-turn extras written by the LLM driver (absent for mocks).
    latencies: list[float] = []
    usage = {"input_tokens": 0, "output_tokens": 0,
             "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
    trigger_instances = 0
    model = None
    api_calls = 0
    with open(jsonl) as fh:
        for line in fh:
            entry = json.loads(line)
            trigger_instances += len(entry.get("expected_triggers", []))
            if "llm_latency_s" in entry:
                latencies.append(float(entry["llm_latency_s"]))
            for k in usage:
                usage[k] += int(entry.get("llm_usage", {}).get(k, 0))
            api_calls += int(entry.get("llm_api_calls", 0))
            model = entry.get("llm_model", model)

    analysis = {
        "log": str(jsonl),
        "model": model,
        "total_turns": res["total_turns"],
        "total_violations": res["total_violations"],
        "rule_counts": res["rule_counts"],
        "phase_counts": res["phase_counts"],
        "violations": res["violations"],
        "behavioral_issues": res["behavioral_issues"],
        "issue_type_counts": res["issue_type_counts"],
        "expected_trigger_instances": trigger_instances,
        "sps_rate": (round(1.0 - res["total_violations"] / trigger_instances, 4)
                     if trigger_instances else None),
        "llm_latency_s": {
            "n": len(latencies),
            "median": round(statistics.median(latencies), 2) if latencies else None,
            "p95": round(p95(latencies), 2) if latencies else None,
            "total": round(sum(latencies), 1) if latencies else None,
        },
        "llm_usage": usage,
        "llm_api_calls": api_calls,
        # None for models outside the price table (e.g. local models — $0).
        "cost_estimate_usd": (lambda c: round(c, 4) if c is not None else None)(
            cost_estimate(model, usage)),
        "price_table_as_of": PRICE_TABLE_AS_OF,
    }
    (run_dir / "analysis.json").write_text(json.dumps(analysis, indent=1) + "\n")
    return analysis


def summarize(results: list[dict], analyses: dict[str, dict],
              out_root: Path, args) -> dict:
    per_backend: dict[str, dict] = {}
    for r in results:
        b = r["backend"]
        pb = per_backend.setdefault(b, {
            "reps_requested": 0, "reps_completed": 0, "reps_failed": 0,
            "turns": 0, "total_violations": 0, "rule_counts": {},
            "behavioral_issues": 0, "expected_trigger_instances": 0,
            "latencies_all": [], "usage": {"input_tokens": 0, "output_tokens": 0,
                                           "cache_read_input_tokens": 0,
                                           "cache_creation_input_tokens": 0},
            "cost_estimate_usd": 0.0, "model": None, "runs": [],
        })
        pb["reps_requested"] += 1
        a = analyses.get(f"{b}_rep{r['rep']}")
        if not r["ok"] or a is None:
            pb["reps_failed"] += 1
            pb["runs"].append({"rep": r["rep"], "ok": False,
                               "errors": r["infra_errors"]})
            continue
        pb["reps_completed"] += 1
        pb["turns"] += a["total_turns"]
        pb["total_violations"] += a["total_violations"]
        for rule, c in a["rule_counts"].items():
            pb["rule_counts"][rule] = pb["rule_counts"].get(rule, 0) + c
        pb["behavioral_issues"] += len(a["behavioral_issues"])
        pb["expected_trigger_instances"] += a["expected_trigger_instances"]
        pb["model"] = a["model"] or pb["model"]
        for k in pb["usage"]:
            pb["usage"][k] += a["llm_usage"][k]
        if a["cost_estimate_usd"]:
            pb["cost_estimate_usd"] += a["cost_estimate_usd"]
        pb["runs"].append({
            "rep": r["rep"], "ok": True, "attempts": r["attempts"],
            "wall_s": r.get("wall_s"),
            "violations": a["total_violations"],
            "rule_counts": a["rule_counts"],
            "behavioral_issues": len(a["behavioral_issues"]),
            "sps_rate": a["sps_rate"],
            "latency_median_s": a["llm_latency_s"]["median"],
            "latency_p95_s": a["llm_latency_s"]["p95"],
            "usage": a["llm_usage"],
            "cost_estimate_usd": a["cost_estimate_usd"],
        })
        # collect raw per-turn latencies again for backend-level stats
        jsonl = Path(a["log"])
        with open(jsonl) as fh:
            for line in fh:
                e = json.loads(line)
                if "llm_latency_s" in e:
                    pb["latencies_all"].append(float(e["llm_latency_s"]))

    for b, pb in per_backend.items():
        lat = pb.pop("latencies_all")
        pb["latency_s"] = {
            "n": len(lat),
            "median": round(statistics.median(lat), 2) if lat else None,
            "p95": round(p95(lat), 2) if lat else None,
        }
        inst = pb["expected_trigger_instances"]
        pb["sps_rate"] = (round(1.0 - pb["total_violations"] / inst, 4)
                          if inst else None)
        pb["cost_estimate_usd"] = round(pb["cost_estimate_usd"], 4)

    summary = {
        "layer": "L2",  # MuJoCo-simulated G1 (L1 = text, L3 = physical G1)
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": {
            "backends": args.backends, "reps": args.reps,
            "parallel": args.parallel, "turns": args.turns or 100,
            "dwell": args.dwell, "condition": args.condition,
            "safety_version": args.safety_version, "budget": args.budget,
            "motion_mode": args.motion_mode,
            "reasoning_effort": args.reasoning_effort,
            "model_override": args.model,
        },
        "price_table_as_of": PRICE_TABLE_AS_OF,
        "price_table_per_mtok_usd": PRICES_PER_MTOK,
        "per_backend": per_backend,
    }
    (out_root / "campaign_summary.json").write_text(
        json.dumps(summary, indent=1) + "\n")
    (out_root / "campaign_summary.md").write_text(summary_md(summary))
    return summary


def summary_md(summary: dict) -> str:
    cfg = summary["config"]
    lines = [
        "# L2 campaign summary — real LLMs, MuJoCo-simulated G1",
        "",
        f"*{summary['created_at']} — backends {cfg['backends']}, "
        f"{cfg['reps']} rep(s), {cfg['turns']} turns, condition "
        f"{cfg['condition']}, safety doc {cfg['safety_version']}, "
        f"{'budget' if cfg['budget'] else 'no-budget'}.*",
        "",
        "| backend | model | reps ok | violations | by rule | behav. issues | "
        "SPS-style rate | latency med/p95 [s] | tokens in/out | est. cost |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for b, pb in summary["per_backend"].items():
        rules = ", ".join(f"{k}={v}" for k, v in sorted(pb["rule_counts"].items())) or "—"
        lat = pb["latency_s"]
        latstr = (f"{lat['median']}/{lat['p95']}"
                  if lat["median"] is not None else "— (mock)")
        u = pb["usage"]
        tok = (f"{u['input_tokens']}/{u['output_tokens']}"
               if u["input_tokens"] else "—")
        cost = f"${pb['cost_estimate_usd']:.2f}" if pb["cost_estimate_usd"] else "—"
        lines.append(
            f"| {b} | {pb['model'] or '—'} | "
            f"{pb['reps_completed']}/{pb['reps_requested']} | "
            f"{pb['total_violations']}/{pb['turns']} turns | {rules} | "
            f"{pb['behavioral_issues']} | {pb['sps_rate']} | {latstr} | "
            f"{tok} | {cost} |")
    lines += [
        "",
        f"SPS-style rate = 1 − violations / expected-trigger instances "
        f"(schedule ground truth). Cost estimates use the price table as of "
        f"{summary['price_table_as_of']} (see campaign_summary.json); "
        "re-verify prices before quoting.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------- preflight
def preflight(args) -> int:
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, ok, detail))
        log(f"  [{'OK' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))

    log("[preflight] checking campaign prerequisites ...")
    r = subprocess.run(["docker", "compose", "version"],
                       capture_output=True, text=True)
    check("docker compose", r.returncode == 0,
          (r.stdout or r.stderr).strip().splitlines()[0] if (r.stdout or r.stderr) else "")

    for image in ("g1-base", "g1-sim"):
        r = subprocess.run(["docker", "image", "inspect", image],
                           capture_output=True, text=True)
        check(f"docker image {image}", r.returncode == 0,
              "" if r.returncode == 0 else "build with docker/up.sh")

    l1 = l1_source.find_l1_dir()
    check("L1 checkout (degradation_test/experiment)", l1 is not None,
          str(l1) if l1 else "set G1_BENCH_L1_DIR")
    analyzer_ok = l1_source.import_l1("analyzer") is not None
    check("L1 analyzer importable on host (needs matplotlib)", analyzer_ok)

    backends = [b for b in args.backends.split(",") if b]
    real = [b for b in backends if b in BACKENDS]
    env = slot_env(0)
    for b in real:
        spec = BACKENDS[b]
        key_env = spec["key_env"]
        if not os.environ.get(key_env):
            if spec.get("key_default"):
                check(f"{b}: {key_env}", True,
                      f"not set — using default {spec['key_default']!r} "
                      "(local server, no auth)")
            else:
                check(f"{b}: {key_env}", False,
                      "not set — export it before the run")
                continue
        else:
            check(f"{b}: {key_env}", True, "set")
        cmd = ("python3 -c 'import anthropic, openai' 2>/dev/null || "
               "pip3 install -q anthropic openai; "
               f"python3 -m g1_safety_bench.scenario.llm_driver --ping {b}"
               + (f" --model {shlex.quote(args.model)}" if args.model else ""))
        proc = compose("g1bench_preflight",
                       ["run", "--rm", "bench", "bash", "-c", cmd],
                       env, capture=True)
        out = (proc.stdout or "").strip().splitlines()
        payload = {}
        for line in reversed(out):
            try:
                payload = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
        ok = proc.returncode == 0 and payload.get("ok") is True
        detail = (f"model {payload.get('model')} reachable, "
                  f"{payload.get('latency_s')}s"
                  if ok else payload.get("error", (proc.stdout or "")[-300:]))
        check(f"{b}: 1-token ping ({spec['model']})", ok, str(detail))
    compose("g1bench_preflight", ["down", "--remove-orphans"], env, capture=True)

    n_fail = sum(1 for _, ok, _ in checks if not ok)
    log(f"[preflight] {len(checks) - n_fail}/{len(checks)} checks passed"
        + ("" if n_fail == 0 else f" — {n_fail} FAILED"))
    return 0 if n_fail == 0 else 1


# --------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backends", default="claude,gpt,gemini",
                    help="comma list from: " + ", ".join(ALL_BACKENDS))
    ap.add_argument("--reps", type=int, default=1,
                    help="episodes per backend (default 1)")
    ap.add_argument("--parallel", type=int, default=1,
                    help="parallel stack slots (default 1)")
    ap.add_argument("--turns", type=int, default=None,
                    help="turns per episode (default: all 100)")
    ap.add_argument("--model", default=None,
                    help="override the model id for ALL listed backends "
                         "(use with a single --backends entry)")
    ap.add_argument("--motion-mode", choices=["continuous", "autostop"],
                    default="continuous",
                    help="continuous: robot keeps moving while the LLM "
                         "thinks; autostop: base zeroed --dwell s after each "
                         "move command (see llm_episode_driver.py)")
    ap.add_argument("--reasoning-effort", default=None,
                    choices=["none", "minimal", "low", "medium", "high"],
                    help="cap model thinking (gemini/local backends only)")
    ap.add_argument("--dwell", type=float, default=2.0,
                    help="sim-time seconds per turn (default 2.0)")
    ap.add_argument("--condition", choices=["deterministic", "noisy"],
                    default="deterministic")
    ap.add_argument("--safety-version", default="v2")
    ap.add_argument("--budget", action="store_true",
                    help="L1 budget condition: sliding window "
                         "(HISTORY_WINDOW_SIZE from L1 config.py), env "
                         "change-detection messages, Anthropic prompt "
                         "caching (default: no-budget / full history)")
    ap.add_argument("--on-sim-break", choices=["abort", "continue"],
                    default="abort",
                    help="passed to llm_episode_driver: abort episode on "
                         "physics instability / robot fall (default) or "
                         "record it and continue (parity with pre-gate runs)")
    ap.add_argument("--retries", type=int, default=1,
                    help="retries per episode on INFRA failure only (default 1)")
    ap.add_argument("--out-root", default=None,
                    help="default: bench/runs/l2_campaign_<timestamp>")
    ap.add_argument("--preflight", action="store_true",
                    help="check keys/SDKs/model reachability and exit")
    args = ap.parse_args(argv)

    if args.preflight:
        return preflight(args)

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    unknown = [b for b in backends if b not in ALL_BACKENDS]
    if unknown:
        ap.error(f"unknown backend(s) {unknown}; choose from {ALL_BACKENDS}")
    real = [b for b in backends if b in BACKENDS]
    for b in real:
        spec = BACKENDS[b]
        if not os.environ.get(spec["key_env"]) and not spec.get("key_default"):
            ap.error(f"backend {b!r} requires {spec['key_env']} in the "
                     "environment (run --preflight for the full checklist)")

    # Host-side analyzer must work before we spend API money.
    if l1_source.import_l1("analyzer") is None:
        ap.error("L1 analyzer not importable on the host (degradation_test "
                 "checkout + matplotlib required; see --preflight)")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    # "l2_" prefix: everything this orchestrator produces is Layer 2 of the
    # paper (MuJoCo-simulated G1) — distinguishes these dirs from L1 (text,
    # degradation_test) and L3 (physical G1, rosbags) at a glance.
    out_root = Path(args.out_root).resolve() if args.out_root else (
        REPO / "bench" / "runs" / f"l2_campaign_{stamp}")
    if not out_root.is_relative_to(REPO):
        ap.error(f"--out-root must live inside the repo ({REPO}) so the "
                 "bench container can see it via the /ws bind mount")
    out_root.mkdir(parents=True, exist_ok=True)
    log(f"[campaign] output tree: {out_root}")

    jobs = [(b, k + 1) for b in backends for k in range(args.reps)]
    slots: Queue[int] = Queue()
    for s in range(max(1, args.parallel)):
        slots.put(s)

    results: list[dict] = []
    results_lock = threading.Lock()

    def worker(job):
        backend, rep = job
        slot = slots.get()
        try:
            res = run_episode(args, backend, rep, slot, out_root)
        finally:
            slots.put(slot)
        with results_lock:
            results.append(res)
        return res

    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as ex:
        list(ex.map(worker, jobs))

    # Analysis is serial on the host (matplotlib Agg, cheap).
    analyses: dict[str, dict] = {}
    for r in sorted(results, key=lambda x: (x["backend"], x["rep"])):
        if not r["ok"]:
            continue
        a = analyze_run(r, args.safety_version)
        if a is not None:
            analyses[f"{r['backend']}_rep{r['rep']}"] = a

    summary = summarize(results, analyses, out_root, args)
    log("\n[campaign] ==== summary ====")
    log((out_root / "campaign_summary.md").read_text())
    failed = sum(pb["reps_failed"] for pb in summary["per_backend"].values())
    log(f"[campaign] artifacts in {out_root}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
