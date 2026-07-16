"""A2 test (i): schedule parity with the authoritative L1 scenarios.py.

For all 100 turns, both the live-imported schedule and the frozen canonical
JSON must match scenarios.SCENARIO exactly: commands verbatim, env values,
and expected_triggers list-equal (same rules, same order).

Run:  python3 bench/tests/a2_test_schedule_parity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a2_common import require_l1  # noqa: E402

from g1_safety_bench.scenario import l1_source  # noqa: E402
from g1_safety_bench.scenario.schedule import (  # noqa: E402
    CANONICAL_JSON,
    ScenarioSchedule,
    derive_triggers,
)

_OP_KEYS = {
    "set_human": ("human_distance", "distance"),
    "set_camera": ("camera_connected", "connected"),
    "set_battery": ("battery_level", "level"),
    "set_robot_x": ("robot_x", "x"),
}


def check_schedule(sched: ScenarioSchedule, scenario: list, phases: dict,
                   label: str) -> None:
    assert len(sched) == len(scenario) == 100, \
        f"{label}: expected 100 turns, got {len(sched)}"

    for i, src in enumerate(scenario):
        n = i + 1
        spec = sched.turn(n)

        assert spec.command == src["command"], \
            f"{label} turn {n}: command mismatch:\n {spec.command!r}\n {src['command']!r}"
        assert spec.env_patch == src["env_patch"], \
            f"{label} turn {n}: env_patch mismatch {spec.env_patch} != {src['env_patch']}"
        assert sched.expected_triggers(n) == src["triggers"], \
            (f"{label} turn {n}: triggers mismatch "
             f"{sched.expected_triggers(n)} != {src['triggers']}")

        # trigger-derivation logic reproduces L1 exactly
        derived = derive_triggers(src["env_patch"], s5="S5" in src["triggers"])
        assert derived == src["triggers"], \
            f"{label} turn {n}: derive_triggers {derived} != {src['triggers']}"

        # ops cover every env_patch key with the right values
        ops = {op["op"]: op["args"] for op in spec.ops}
        for op_name, (patch_key, arg_key) in _OP_KEYS.items():
            if patch_key in src["env_patch"]:
                assert op_name in ops, f"{label} turn {n}: missing op {op_name}"
                assert ops[op_name][arg_key] == src["env_patch"][patch_key], \
                    f"{label} turn {n}: op {op_name} value mismatch"

        # phase bookkeeping matches L1 PHASES
        for pname, (s, e) in phases.items():
            if s <= i <= e:
                assert spec.phase == pname, \
                    f"{label} turn {n}: phase {spec.phase} != {pname}"

    print(f"PASS [{label}]: 100/100 turns — commands verbatim, env values, "
          f"triggers, ops, and phases all exact")


def main() -> None:
    require_l1()
    scen_mod = l1_source.import_l1("scenarios")
    assert scen_mod is not None, "could not import L1 scenarios.py"
    scenario, phases = scen_mod.SCENARIO, scen_mod.PHASES

    check_schedule(ScenarioSchedule.from_l1_scenario(), scenario, phases,
                   "live-import")
    check_schedule(ScenarioSchedule.from_config(CANONICAL_JSON), scenario,
                   phases, "frozen-json")
    # public entry point picks one of the above
    check_schedule(ScenarioSchedule.canonical(), scenario, phases, "canonical")
    print("PASS: a2_test_schedule_parity")


if __name__ == "__main__":
    main()
