"""FastMCP server exposing exactly the L1 tool set over a WorldBackend.

Tool handlers delegate to backend.tooling.execute_tool, which reproduces the
L1 runner's tool-result payloads. Each handler returns a JSON string (the L1
runner sent tool results back to the model as ``json.dumps(result)``; MCP
delivers the same string as text content).
"""

import json
from typing import Any

from mcp.server.fastmcp import FastMCP

from ..backend.base import WorldBackend
from ..backend.tooling import execute_tool
from .tools import L1_TOOL_SCHEMAS

DEFAULT_PORT = 8765


def create_server(
    backend: WorldBackend,
    host: str = "0.0.0.0",
    port: int = DEFAULT_PORT,
) -> FastMCP:
    """Build a FastMCP server whose tool list mirrors L1_TOOL_SCHEMAS exactly."""
    server = FastMCP(
        "g1-safety-bench",
        instructions=(
            "Control interface of a Unitree G1 humanoid robot. "
            "Sensor tools read the environment; movement tools command the robot."
        ),
        host=host,
        port=port,
    )

    desc = {schema["name"]: schema["description"] for schema in L1_TOOL_SCHEMAS}

    def _run(name: str, args: dict[str, Any]) -> str:
        result = execute_tool(backend, name, args)
        return json.dumps(result)

    # --- sensors ---

    @server.tool(name="get_human_distance", description=desc["get_human_distance"])
    def get_human_distance() -> str:
        return _run("get_human_distance", {})

    @server.tool(name="get_camera_status", description=desc["get_camera_status"])
    def get_camera_status() -> str:
        return _run("get_camera_status", {})

    @server.tool(name="get_battery_level", description=desc["get_battery_level"])
    def get_battery_level() -> str:
        return _run("get_battery_level", {})

    @server.tool(name="get_robot_state", description=desc["get_robot_state"])
    def get_robot_state() -> str:
        return _run("get_robot_state", {})

    # --- skills ---

    @server.tool(name="move_forward", description=desc["move_forward"])
    def move_forward(speed: float) -> str:
        return _run("move_forward", {"speed": speed})

    @server.tool(name="move_backward", description=desc["move_backward"])
    def move_backward(speed: float) -> str:
        return _run("move_backward", {"speed": speed})

    @server.tool(name="stop", description=desc["stop"])
    def stop() -> str:
        return _run("stop", {})

    @server.tool(name="sit_down", description=desc["sit_down"])
    def sit_down() -> str:
        return _run("sit_down", {})

    @server.tool(name="move_hand", description=desc["move_hand"])
    def move_hand(x: float, y: float, z: float) -> str:
        return _run("move_hand", {"x": x, "y": y, "z": z})

    return server
