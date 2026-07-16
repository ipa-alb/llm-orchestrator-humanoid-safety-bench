"""A2 test (iii): log schema is a strict SUPERSET of the L1 JSONL schema.

Compares a line from a generated episode against a real L1 log line (found
under degradation_test/experiment/results_*); if no real log is readable,
validates against the field list hard-coded from logger.py's write code.
Also checks the run-metadata sidecar and the added sim-bench fields.

Run:  python3 bench/tests/a2_test_logger_schema.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a2_common import L1_DIR, StubBackend  # noqa: E402

from g1_safety_bench.logspine.episode_logger import (  # noqa: E402
    CONTRACT_VERSION,
    EpisodeLogger,
)
from g1_safety_bench.scenario.mock_llm import CompliantPolicy  # noqa: E402
from g1_safety_bench.scenario.runner_sim import SimulationRunner  # noqa: E402
from g1_safety_bench.scenario.schedule import ScenarioSchedule  # noqa: E402

# Field -> type, as written by degradation_test/experiment/logger.py
L1_FALLBACK_SCHEMA = {
    "turn": int,
    "timestamp": str,
    "safety_version": str,
    "command": str,
    "env_state": dict,
    "expected_triggers": list,
    "llm_response": str,
    "tool_calls": list,
    "tool_results": list,
}

ADDED_FIELDS = {
    "sim_time": float,
    "wall_time_start": float,
    "wall_time_end": float,
    "backend": str,
    "run_id": str,
    "contract_version": str,
}

META_REQUIRED = {"record_type", "contract_version", "run_id",
                 "safety_version", "backend", "scene", "seed",
                 "image_digest", "scenario_name"}


def find_real_l1_line() -> dict | None:
    if L1_DIR is None:
        return None
    for p in sorted(L1_DIR.glob("results_*/run_*.jsonl")) + \
            sorted(L1_DIR.glob("InitialResults/**/run_*.jsonl")):
        try:
            with open(p) as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        entry = json.loads(line)
                        if "turn" in entry and "env_state" in entry:
                            print(f"reference L1 log line: {p}")
                            return entry
        except OSError:
            continue
    return None


def main() -> None:
    out_dir = Path(tempfile.mkdtemp(prefix="a2_schema_"))
    sched = ScenarioSchedule.canonical()
    logger = EpisodeLogger(out_dir, safety_version="v2", run_id="a2schema",
                           backend="mujoco", scene="stub", seed=1,
                           scenario_name=sched.name)
    runner = SimulationRunner(sched, CompliantPolicy(), logger,
                              backend=StubBackend(), max_turns=3)
    log_path = runner.run()

    ours = [json.loads(l) for l in
            Path(log_path).read_text().strip().splitlines()]
    assert len(ours) == 3
    mine = ours[0]

    real = find_real_l1_line()
    if real is not None:
        ref_schema = {k: type(v) for k, v in real.items()}
    else:
        print("no readable real L1 log found; using logger.py field list")
        ref_schema = dict(L1_FALLBACK_SCHEMA)

    # 1. every L1 field present, same name, compatible type
    for key, typ in ref_schema.items():
        assert key in mine, f"missing L1 field: {key}"
        ok = isinstance(mine[key], typ) or \
            (typ in (int, float) and isinstance(mine[key], (int, float)))
        assert ok, f"field {key}: type {type(mine[key]).__name__} != {typ.__name__}"
    print(f"PASS [superset]: all {len(ref_schema)} L1 fields present with "
          f"matching types")

    # 2. env_state keys are a superset of the L1 env_state keys
    if real is not None:
        missing = set(real["env_state"]) - set(mine["env_state"])
        assert not missing, f"env_state missing keys: {missing}"
        print(f"PASS [env_state]: superset of {sorted(real['env_state'])}")

    # 3. tool_calls / tool_results element shape matches L1
    assert mine["tool_calls"], "expected tool calls on turn 1"
    tc = mine["tool_calls"][0]
    assert set(tc) == {"tool_name", "tool_input", "tool_use_id"}, set(tc)
    tr = mine["tool_results"][0]
    assert set(tr) == {"tool_use_id", "result"}, set(tr)
    print("PASS [tool shapes]: tool_calls/tool_results element keys match L1")

    # 4. sim-bench additions present with declared types
    for key, typ in ADDED_FIELDS.items():
        assert key in mine, f"missing added field: {key}"
        assert isinstance(mine[key], typ) or \
            (typ is float and isinstance(mine[key], (int, float))), key
    assert mine["backend"] == "mujoco"
    assert mine["contract_version"] == CONTRACT_VERSION
    assert ours[1]["sim_time"] > ours[0]["sim_time"], "sim_time must advance"
    print(f"PASS [additions]: {sorted(ADDED_FIELDS)} present")

    # 5. run-metadata header line in the sidecar
    meta_path = Path(logger.meta_filename)
    assert meta_path.is_file(), meta_path
    meta_lines = meta_path.read_text().strip().splitlines()
    assert len(meta_lines) == 1, "meta sidecar must be a single JSON line"
    meta = json.loads(meta_lines[0])
    missing = META_REQUIRED - set(meta)
    assert not missing, f"meta header missing: {missing}"
    assert meta["record_type"] == "run_header"
    print(f"PASS [meta header]: {meta_path.name} has "
          f"{sorted(META_REQUIRED)}")

    print("PASS: a2_test_logger_schema")


if __name__ == "__main__":
    main()
