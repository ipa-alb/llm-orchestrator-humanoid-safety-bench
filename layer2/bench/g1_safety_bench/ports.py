"""Central env-var parametrization of every stack collision point (agent C1).

Allows multiple full stacks (sim + loco + battery + bench) to run on one host
simultaneously ("slots") without cross-talk. All defaults equal the historical
hard-coded values, so a stack started with no env vars behaves exactly as
before.

Environment variables (canonical list — scripts that cannot import this
module, e.g. scripts/run_sim_headless.py, duplicate this logic verbatim):

    BENCH_PORT_BASE     base ZMQ port    (default 5555)
    WORLDSTATE_PORT     sim PUB          (default BENCH_PORT_BASE + 0)
    SKILL_PORT          bench PUB->loco  (default BENCH_PORT_BASE + 1)
    WORLDCTL_PORT       sim REP          (default BENCH_PORT_BASE + 2)
    BENCH_DDS_DOMAIN    DDS domain id    (default 1)
    BENCH_DDS_IFACE     DDS interface    (default "lo")
    BENCH_MCP_PORT      MCP server port  (default 8765)

Slot convention used by scripts/campaign.py: slot k uses
BENCH_PORT_BASE = 5555 + 20*k, BENCH_DDS_DOMAIN = 1 + k,
BENCH_MCP_PORT = 8765 + k (slot 0 == the historical defaults).

DDS separation note: CycloneDDS discovery ports are a function of the domain
id (7400 + 250*domain), so distinct domains on the same interface never see
each other's rt/* topics.
"""

from __future__ import annotations

import os


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name, "").strip()
    if not v:
        return default
    try:
        return int(v)
    except ValueError as e:
        raise ValueError(f"env var {name} must be an integer, got {v!r}") from e


PORT_BASE = _env_int("BENCH_PORT_BASE", 5555)
WORLDSTATE_PORT = _env_int("WORLDSTATE_PORT", PORT_BASE)
SKILL_PORT = _env_int("SKILL_PORT", PORT_BASE + 1)
WORLDCTL_PORT = _env_int("WORLDCTL_PORT", PORT_BASE + 2)

DDS_DOMAIN = _env_int("BENCH_DDS_DOMAIN", 1)
DDS_IFACE = os.environ.get("BENCH_DDS_IFACE", "lo")

MCP_PORT = _env_int("BENCH_MCP_PORT", 8765)


def addr(port: int, host: str = "127.0.0.1") -> str:
    return f"tcp://{host}:{port}"
