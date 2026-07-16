"""A1 tests: MCP server tool registration matches the L1 tool set exactly.

Checks, for every tool: name, parameter names, and required parameters equal
those of degradation_test/experiment/mock_tools.py::TOOL_SCHEMAS. Also
exercises one full tool call through the MCP layer and verifies the payload
equals the L1 handler output shape.

Run standalone:  python bench/tests/a1_test_mcp_tools.py
Requires the `mcp` package (a hard dependency of g1_safety_bench); skips
gracefully if it is missing on the host.
"""

import asyncio
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import mcp  # noqa: F401
    HAVE_MCP = True
except ImportError:
    HAVE_MCP = False

# Path to the authoritative L1 file (present in this workspace; absent in the
# docker image — the cross-check is skipped there and the vendored copy rules).
L1_MOCK_TOOLS = (
    Path(__file__).resolve().parents[3]
    / "layer1" / "experiment" / "mock_tools.py"
)


def _skip(msg):
    print(f"SKIP: {msg}")
    if "pytest" in sys.modules:
        import pytest
        pytest.skip(msg, allow_module_level=False)


def _load_upstream_schemas():
    """Import the real L1 mock_tools.py (needs its sibling config.py)."""
    exp_dir = str(L1_MOCK_TOOLS.parent)
    sys.path.insert(0, exp_dir)
    try:
        spec = importlib.util.spec_from_file_location("l1_mock_tools", L1_MOCK_TOOLS)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.TOOL_SCHEMAS
    finally:
        sys.path.remove(exp_dir)


def test_vendored_schemas_match_upstream():
    """Vendored copy in mcp_server/tools.py == authoritative L1 file."""
    from g1_safety_bench.mcp_server.tools import L1_TOOL_SCHEMAS
    if not L1_MOCK_TOOLS.exists():
        _skip(f"upstream {L1_MOCK_TOOLS} not on this machine")
        return
    upstream = _load_upstream_schemas()
    assert L1_TOOL_SCHEMAS == upstream, (
        "vendored L1_TOOL_SCHEMAS diverged from degradation_test mock_tools.py"
    )


def _registered_tools():
    from g1_safety_bench.backend import StubBackend
    from g1_safety_bench.mcp_server import create_server
    backend = StubBackend()
    backend.reset(seed=1)
    server = create_server(backend)
    tools = asyncio.run(server.list_tools())
    return backend, server, {t.name: t for t in tools}


def test_mcp_tool_registration_matches_l1():
    if not HAVE_MCP:
        _skip("mcp package not installed")
        return
    from g1_safety_bench.mcp_server.tools import L1_TOOL_SCHEMAS
    _, _, registered = _registered_tools()

    # Exact same tool set — nothing missing, nothing extra.
    assert set(registered) == {s["name"] for s in L1_TOOL_SCHEMAS}

    for schema in L1_TOOL_SCHEMAS:
        tool = registered[schema["name"]]
        want = schema["input_schema"]
        got = tool.inputSchema
        got_props = set(got.get("properties", {}))
        want_props = set(want["properties"])
        assert got_props == want_props, (
            f"{schema['name']}: param names {got_props} != L1 {want_props}"
        )
        assert set(got.get("required", [])) == set(want["required"]), (
            f"{schema['name']}: required mismatch"
        )
        # numeric params stay numeric
        for pname, pspec in want["properties"].items():
            assert got["properties"][pname]["type"] == pspec["type"], (
                f"{schema['name']}.{pname}: type mismatch"
            )
        assert tool.description == schema["description"]


def test_mcp_call_returns_l1_payload():
    if not HAVE_MCP:
        _skip("mcp package not installed")
        return
    backend, server, _ = _registered_tools()

    async def call(name, args):
        content, _structured = await server.call_tool(name, args)
        assert len(content) == 1 and content[0].type == "text"
        return json.loads(content[0].text)

    out = asyncio.run(call("move_forward", {"speed": 40}))
    assert out == {"status": "ok", "speed": 40}

    out = asyncio.run(call("get_robot_state", {}))
    assert out == {"position": {"x": 0.4, "y": 0.0, "z": 0.0},
                   "speed": 40, "status": "moving"}

    out = asyncio.run(call("get_battery_level", {}))
    assert out == {"battery_percent": 100}

    out = asyncio.run(call("get_camera_status", {}))
    assert out == {"status": "connected"}

    out = asyncio.run(call("get_human_distance", {}))
    assert set(out) == {"distance_meters"}

    out = asyncio.run(call("move_hand", {"x": 9.0, "y": 0.0, "z": 0.5}))
    assert out == {"status": "out_of_bounds",
                   "position": {"x": 9.0, "y": 0.0, "z": 0.5}}

    out = asyncio.run(call("sit_down", {}))
    assert out == {"status": "ok"}

    out = asyncio.run(call("stop", {}))
    assert out == {"status": "ok"}

    out = asyncio.run(call("move_backward", {"speed": 10}))
    assert out == {"status": "ok", "speed": 10}


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"OK ({len(fns)} tests)")
