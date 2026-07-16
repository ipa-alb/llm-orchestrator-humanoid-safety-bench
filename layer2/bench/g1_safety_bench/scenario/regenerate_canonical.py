"""Regenerate canonical_degradation_100.json 1:1 from the live L1 scenario.

Usage:
    python3 bench/g1_safety_bench/scenario/regenerate_canonical.py

Requires the degradation_test checkout (sibling of humanoid_testing, or
G1_BENCH_L1_DIR). The frozen JSON is what ScenarioSchedule.canonical() falls
back to when the L1 checkout is not importable.
"""

from __future__ import annotations

import sys
from pathlib import Path

_BENCH = str(Path(__file__).resolve().parents[2])
if _BENCH not in sys.path:
    sys.path.insert(0, _BENCH)

from g1_safety_bench.scenario.schedule import (  # noqa: E402
    CANONICAL_JSON,
    ScenarioSchedule,
)


def main() -> None:
    sched = ScenarioSchedule.from_l1_scenario()  # raises if L1 missing
    sched.save(CANONICAL_JSON)
    print(f"Wrote {len(sched)} turns -> {CANONICAL_JSON}")


if __name__ == "__main__":
    main()
