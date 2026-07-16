"""MCP server exposing the L1 benchmark tool set over a WorldBackend.

Run:  python -m g1_safety_bench.mcp_server --backend stub|mujoco
      (default transport: streamable-http on 0.0.0.0:8765; --transport sse
       for SSE; --stdio for stdio mode)
"""

from .server import DEFAULT_PORT, create_server
from .tools import L1_TOOL_NAMES, L1_TOOL_SCHEMAS

__all__ = ["create_server", "DEFAULT_PORT", "L1_TOOL_NAMES", "L1_TOOL_SCHEMAS"]
