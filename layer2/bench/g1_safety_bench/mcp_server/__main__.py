"""Entry point: python -m g1_safety_bench.mcp_server --backend stub|mujoco

Transports:
  streamable-http (default)  http://<host>:8765/mcp   — for host-side
                             `claude -p --mcp-config` connecting into the
                             container over host networking
  sse                        http://<host>:8765/sse
  stdio                      --stdio (or --transport stdio)
"""

import argparse
import sys

from ..ports import MCP_PORT


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m g1_safety_bench.mcp_server",
        description="MCP server exposing the G1 safety-benchmark tool set.",
    )
    parser.add_argument(
        "--backend", choices=["stub", "mujoco"], default="stub",
        help="World backend: 'stub' (in-memory, no sim needed) or 'mujoco' "
             "(ZMQ client of the sim world process).",
    )
    parser.add_argument(
        "--transport", choices=["streamable-http", "sse", "stdio"],
        default="streamable-http",
        help="MCP transport (default: streamable-http on --port).",
    )
    parser.add_argument(
        "--stdio", action="store_true",
        help="Shortcut for --transport stdio.",
    )
    parser.add_argument("--host", default="0.0.0.0",
                        help="Bind host for HTTP transports (default 0.0.0.0).")
    parser.add_argument("--port", type=int, default=MCP_PORT,
                        help=f"Port for HTTP transports (default {MCP_PORT}; "
                             "env BENCH_MCP_PORT).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed passed to backend.reset() at startup.")
    parser.add_argument("--no-reset", action="store_true",
                        help="Do not call backend.reset() at startup "
                             "(e.g. when the scenario driver owns resets).")
    # MuJoCo backend ZMQ overrides
    parser.add_argument("--worldstate-addr", default=None)
    parser.add_argument("--skill-addr", default=None)
    parser.add_argument("--worldctl-addr", default=None)
    args = parser.parse_args(argv)

    if args.backend == "stub":
        from ..backend.stub import StubBackend
        backend = StubBackend(seed=args.seed)
    else:
        from ..backend.mujoco_zmq import MuJoCoBackend
        kwargs = {}
        if args.worldstate_addr:
            kwargs["worldstate_addr"] = args.worldstate_addr
        if args.skill_addr:
            kwargs["skill_addr"] = args.skill_addr
        if args.worldctl_addr:
            kwargs["worldctl_addr"] = args.worldctl_addr
        backend = MuJoCoBackend(**kwargs)

    if not args.no_reset:
        try:
            backend.reset(seed=args.seed)
        except TimeoutError as exc:
            print(f"warning: backend.reset failed ({exc}); continuing without reset",
                  file=sys.stderr)

    from .server import create_server
    server = create_server(backend, host=args.host, port=args.port)

    transport = "stdio" if args.stdio else args.transport
    if transport != "stdio":
        path = "/mcp" if transport == "streamable-http" else "/sse"
        print(f"g1-safety-bench MCP server: {transport} on "
              f"http://{args.host}:{args.port}{path} (backend={args.backend})",
              file=sys.stderr)
    server.run(transport=transport)
    return 0


if __name__ == "__main__":
    sys.exit(main())
