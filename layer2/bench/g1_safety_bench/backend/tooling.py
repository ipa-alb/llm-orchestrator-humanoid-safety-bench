"""execute_tool — the L1 tool surface on top of any WorldBackend.

Replicates handle_tool_call() from degradation_test/experiment/mock_tools.py:
same tool names, same argument names, same result payload shapes. The MCP
server is a thin wrapper around this function, so payload parity with the L1
runner lives in exactly one place.
"""

from typing import Any

from .base import WorldBackend

# Tool names whose invocation counts as motion for the safety analyzer
# (mirrors mock_tools.MOVEMENT_TOOLS).
MOVEMENT_TOOLS = {"move_forward", "move_backward", "move_hand"}

SENSOR_TOOLS = {"get_human_distance", "get_camera_status", "get_battery_level",
                "get_robot_state"}
SKILL_TOOLS = {"move_forward", "move_backward", "stop", "sit_down", "move_hand"}
ALL_TOOLS = SENSOR_TOOLS | SKILL_TOOLS


def execute_tool(backend: WorldBackend, name: str, args: dict[str, Any] | None = None) -> dict:
    """Execute one L1 tool against a backend; return the L1-shaped dict."""
    args = args or {}

    if name == "get_human_distance":
        # Noise is owned by the measurement (worldstate / stub RNG); this
        # layer only applies L1's clamp + rounding.
        d = backend.read_sensor("human_distance")
        return {"distance_meters": round(max(0.0, float(d)), 2)}

    if name == "get_camera_status":
        connected = backend.read_sensor("camera_connected")
        return {"status": "connected" if connected else "disconnected"}

    if name == "get_battery_level":
        level = backend.read_sensor("battery_level")
        return {"battery_percent": int(round(float(level)))}

    if name == "get_robot_state":
        x, y, z = backend.read_sensor("robot_position")
        return {
            "position": {"x": x, "y": y, "z": z},
            "speed": backend.read_sensor("robot_speed"),
            "status": backend.read_sensor("robot_status"),
        }

    if name in SKILL_TOOLS:
        return backend.apply_skill(name, **args)

    return {"error": f"Unknown tool: {name}"}
