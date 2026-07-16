"""StubBackend — pure in-memory world replicating L1's EnvironmentState.

Semantics are copied from degradation_test/experiment/mock_tools.py so that
unit tests (and dry runs of the MCP server) behave exactly like the L1
benchmark environment, without a simulator:

  * move_forward/backward(speed) displace x by +/- speed*0.01 m instantly,
    refused with {"status": "out_of_bounds", ...} if the destination leaves
    WORKSPACE_BOUNDS.
  * move_hand(x, y, z) checks bounds and echoes the position; no state change
    (same as L1).
  * human_distance readings carry uniform(-0.1, 0.1) jitter, clamped >= 0
    (L1 applied this in the tool handler; here the *measurement* is noisy and
    ground_truth() is exact, matching the L2 worldstate contract).

Deterministic: reset(seed) seeds the jitter RNG.
"""

import random
from typing import Any

from .base import WorldBackend
from .constants import L1_DEFAULT_SPEED, L1_STEP_PER_SPEED_PCT, in_bounds

_DEFAULT_STATE = {
    "human_distance": 5.0,
    "camera_connected": True,
    "battery_level": 100,
    "robot_x": 0.0,
    "robot_y": 0.0,
    "robot_z": 0.0,
    "robot_speed": 0.0,
    "robot_status": "idle",
}


class StubBackend(WorldBackend):
    """In-memory dict world for unit tests and stub MCP runs."""

    def __init__(self, seed: int = 0) -> None:
        self._state: dict[str, Any] = dict(_DEFAULT_STATE)
        self._sim_time = 0.0
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------ ABC

    def reset(self, seed: int, scenario_cfg: dict | None = None) -> None:
        self._state = dict(_DEFAULT_STATE)
        self._sim_time = 0.0
        self._rng = random.Random(seed)
        if scenario_cfg:
            self.apply_env(scenario_cfg)

    def advance(self, dt_sim: float) -> None:
        # No continuous physics in the stub: time simply elapses.
        self._sim_time += dt_sim

    def read_sensor(self, name: str) -> Any:
        s = self._state
        if name == "human_distance":
            # Noisy measurement (jitter identical to L1's tool handler).
            return max(0.0, s["human_distance"] + self._rng.uniform(-0.1, 0.1))
        if name == "camera_connected":
            return s["camera_connected"]
        if name == "battery_level":
            return s["battery_level"]
        if name == "robot_position":
            return (s["robot_x"], s["robot_y"], s["robot_z"])
        if name == "robot_speed":
            return s["robot_speed"]
        if name == "robot_status":
            return s["robot_status"]
        raise KeyError(f"Unknown sensor: {name}")

    def apply_skill(self, name: str, **params: Any) -> dict:
        s = self._state

        if name in ("move_forward", "move_backward"):
            speed = params.get("speed", L1_DEFAULT_SPEED)
            step = speed * L1_STEP_PER_SPEED_PCT
            if name == "move_backward":
                step = -step
            new_x = s["robot_x"] + step
            if not in_bounds(new_x, s["robot_y"], s["robot_z"]):
                return {"status": "out_of_bounds", "speed": speed}
            s["robot_x"] = round(new_x, 3)
            s["robot_speed"] = speed
            s["robot_status"] = "moving"
            return {"status": "ok", "speed": speed}

        if name == "stop":
            s["robot_speed"] = 0
            s["robot_status"] = "stopped"
            return {"status": "ok"}

        if name == "sit_down":
            s["robot_speed"] = 0
            s["robot_status"] = "seated"
            return {"status": "ok"}

        if name == "move_hand":
            x = params.get("x", 0)
            y = params.get("y", 0)
            z = params.get("z", 0)
            if not in_bounds(x, y, z):
                return {"status": "out_of_bounds", "position": {"x": x, "y": y, "z": z}}
            return {"status": "ok", "position": {"x": x, "y": y, "z": z}}

        return {"error": f"Unknown skill: {name}"}

    def ground_truth(self) -> dict:
        return {
            "sim_time": self._sim_time,
            "human_distance": self._state["human_distance"],
            "camera_connected": self._state["camera_connected"],
            "battery_level": self._state["battery_level"],
            "robot_pos": [self._state["robot_x"], self._state["robot_y"], self._state["robot_z"]],
            "robot_speed": self._state["robot_speed"],
            "robot_status": self._state["robot_status"],
        }

    # ------------------------------------------------------------ extension

    def apply_env(self, patch: dict) -> None:
        """Accepts L1 env_patch keys (human_distance, camera_connected,
        battery_level, robot_x/y/z, robot_speed, robot_status)."""
        for key, val in patch.items():
            if key == "human_waypoint":
                # Stub has no moving human: teleport to the waypoint distance
                # if a distance can be derived, otherwise ignore.
                continue
            if key not in self._state:
                raise KeyError(f"Unknown env key: {key}")
            self._state[key] = val
