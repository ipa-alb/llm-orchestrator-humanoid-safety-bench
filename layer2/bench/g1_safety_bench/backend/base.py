"""WorldBackend — the abstract world interface the benchmark loop talks to.

A backend hides *where* the world lives (in-memory stub, MuJoCo sim over ZMQ,
eventually real hardware) behind five methods. Sensor names and skill names /
payload shapes replicate the L1 benchmark exactly (see backend/README.md and
degradation_test/experiment/mock_tools.py).
"""

from abc import ABC, abstractmethod
from typing import Any


class WorldBackend(ABC):
    """Abstract world backend.

    Canonical sensor names (read_sensor):
      human_distance    float, meters (noisy — measurement, not ground truth)
      camera_connected  bool
      battery_level     number, percent 0-100
      robot_position    (x, y, z) tuple, meters
      robot_speed       number, last commanded speed percent 0-100
      robot_status      str: "idle" | "moving" | "stopped" | "seated"

    Canonical skill names (apply_skill) — the L1 movement tools:
      move_forward(speed)   move_backward(speed)   stop()
      sit_down()            move_hand(x, y, z)

    apply_skill returns the exact L1 tool-result dicts, e.g.
      {"status": "ok", "speed": 40} or {"status": "out_of_bounds", ...}.
    """

    @abstractmethod
    def reset(self, seed: int, scenario_cfg: dict | None = None) -> None:
        """Reset the world to initial conditions.

        scenario_cfg (optional) carries initial environment overrides using L1
        env_patch keys: human_distance, camera_connected, battery_level, plus
        the L2 extension human_waypoint = {"pos": [x, y], "speed": v}.
        """

    @abstractmethod
    def advance(self, dt_sim: float) -> None:
        """Let at least dt_sim seconds of simulated time elapse.

        NOTE: physics is NOT frozen between calls in sim-backed backends —
        see backend/README.md ("Time semantics"). This is a synchronization
        point for turn-keyed scenario logic, not a lockstep step().
        """

    @abstractmethod
    def read_sensor(self, name: str) -> Any:
        """Return the current value of a canonical sensor (see class doc)."""

    @abstractmethod
    def apply_skill(self, name: str, **params: Any) -> dict:
        """Execute a skill; return the L1-shaped result dict."""

    @abstractmethod
    def ground_truth(self) -> dict:
        """Noise-free world state for the analyzer/logger (never for the LLM)."""

    # --- Non-abstract extension used by the scenario driver (A2) ------------

    def apply_env(self, patch: dict) -> None:
        """Apply a scenario env patch at a turn boundary.

        Keys follow L1 env_patch: human_distance, camera_connected,
        battery_level (+ human_waypoint for L2). Default raises so backends
        must opt in explicitly.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support apply_env")

    def close(self) -> None:
        """Release resources (sockets, threads). Default: no-op."""
