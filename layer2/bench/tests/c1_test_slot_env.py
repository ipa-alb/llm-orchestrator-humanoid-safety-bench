#!/usr/bin/env python3
"""C1 unit test: parallel-slot env-var plumbing.

Verifies that BENCH_PORT_BASE / WORLDSTATE_PORT / SKILL_PORT / WORLDCTL_PORT /
BENCH_DDS_DOMAIN / BENCH_MCP_PORT propagate into every collision point:
backend/constants.py addresses, simworld/core.py DEFAULT_CFG, loco/config.py
endpoints + DDS domain, and mcp ports — and that the unset defaults are the
historical 5555/5556/5557 / domain 1 / 8765 values.

Each case runs in a fresh subprocess because the values are resolved at
import time (one process == one stack slot).

Run: python3 bench/tests/c1_test_slot_env.py   (host or g1-base; numpy needed)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]

PROBE = r"""
import json
from g1_safety_bench import ports
from g1_safety_bench.backend import constants as C
from g1_safety_bench.simworld.core import DEFAULT_CFG
from g1_safety_bench.loco import config as L
print(json.dumps({
    "worldstate_addr": C.WORLDSTATE_ADDR,
    "skill_addr": C.SKILL_ADDR,
    "worldctl_addr": C.WORLDCTL_ADDR,
    "pub_port": DEFAULT_CFG["pub_port"],
    "rep_port": DEFAULT_CFG["rep_port"],
    "loco_skill": L.ZMQ_SKILL_ENDPOINT,
    "loco_worldctl": L.ZMQ_WORLDCTL_ENDPOINT,
    "dds_domain": L.DDS_DOMAIN_ID,
    "mcp_port": ports.MCP_PORT,
}))
"""


def probe(extra_env: dict) -> dict:
    env = dict(os.environ)
    for k in ("BENCH_PORT_BASE", "WORLDSTATE_PORT", "SKILL_PORT",
              "WORLDCTL_PORT", "BENCH_DDS_DOMAIN", "BENCH_MCP_PORT"):
        env.pop(k, None)
    env.update(extra_env)
    env["PYTHONPATH"] = str(BENCH)
    out = subprocess.run([sys.executable, "-c", PROBE], env=env,
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout.strip().splitlines()[-1])


def main() -> int:
    # Defaults == historical single-stack contract.
    d = probe({})
    assert d["worldstate_addr"] == "tcp://127.0.0.1:5555"
    assert d["skill_addr"] == "tcp://127.0.0.1:5556"
    assert d["worldctl_addr"] == "tcp://127.0.0.1:5557"
    assert (d["pub_port"], d["rep_port"]) == (5555, 5557)
    assert d["loco_skill"] == "tcp://127.0.0.1:5556"
    assert d["dds_domain"] == 1 and d["mcp_port"] == 8765
    print("  ok: unset env reproduces the historical defaults")

    # Slot 1 (campaign convention: base+20, domain+1, mcp+1).
    s = probe({"BENCH_PORT_BASE": "5575", "BENCH_DDS_DOMAIN": "2",
               "BENCH_MCP_PORT": "8766"})
    assert s["worldstate_addr"] == "tcp://127.0.0.1:5575"
    assert s["skill_addr"] == "tcp://127.0.0.1:5576"
    assert s["worldctl_addr"] == "tcp://127.0.0.1:5577"
    assert (s["pub_port"], s["rep_port"]) == (5575, 5577)
    assert s["loco_skill"] == "tcp://127.0.0.1:5576"
    assert s["loco_worldctl"] == "tcp://127.0.0.1:5577"
    assert s["dds_domain"] == 2 and s["mcp_port"] == 8766
    print("  ok: BENCH_PORT_BASE/BENCH_DDS_DOMAIN/BENCH_MCP_PORT propagate everywhere")

    # Individual port overrides beat the base.
    o = probe({"BENCH_PORT_BASE": "6000", "SKILL_PORT": "7001"})
    assert o["worldstate_addr"] == "tcp://127.0.0.1:6000"
    assert o["skill_addr"] == "tcp://127.0.0.1:7001"
    assert o["loco_skill"] == "tcp://127.0.0.1:7001"
    print("  ok: explicit WORLDSTATE/SKILL/WORLDCTL_PORT override the base")

    print("[c1_test_slot_env] ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
