"""Shared helpers for the a2_* tests (plain-script tests, stdlib only)."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve()
BENCH_DIR = HERE.parents[1]                  # .../humanoid_testing/bench
HT_DIR = HERE.parents[2]                     # .../humanoid_testing

if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

from g1_safety_bench.scenario import l1_source  # noqa: E402

L1_DIR = l1_source.find_l1_dir()


class StubBackend:
    """Minimal WorldBackend-contract-compatible stub (used until A1's
    backend lands; mirrors reset/advance/read_sensor/apply_skill/
    ground_truth plus the worldctl helper methods)."""

    def __init__(self):
        self.sim_time = 0.0
        self.state = {
            "human_distance": 5.0,
            "camera_connected": True,
            "battery_level": 100,
            "robot_x": 0.0,
        }
        self.skills: list[tuple] = []
        self.reset_args = None

    # --- contract core ---
    def reset(self, seed=0, scenario_cfg=None):
        self.sim_time = 0.0
        self.reset_args = (seed, scenario_cfg)

    def advance(self, dt_sim):
        self.sim_time += float(dt_sim)

    def read_sensor(self, name):
        return self.state.get(name)

    def apply_skill(self, name, **params):
        self.skills.append((self.sim_time, name, params))
        return {"applied": name}

    def ground_truth(self):
        return {"sim_time": self.sim_time, **self.state}

    # --- worldctl helpers ---
    def set_human(self, distance):
        self.state["human_distance"] = distance

    def set_camera(self, connected):
        self.state["camera_connected"] = bool(connected)

    def set_battery(self, level):
        self.state["battery_level"] = level

    def set_robot_x(self, x):
        self.state["robot_x"] = x


def require_l1() -> Path:
    if L1_DIR is None:
        print("SKIP: degradation_test/experiment checkout not found "
              "(set G1_BENCH_L1_DIR)")
        sys.exit(0)
    return L1_DIR
