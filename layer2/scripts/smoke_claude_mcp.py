#!/usr/bin/env python3
"""R1 smoke: Claude Code headless (`claude -p`) as the orchestrator-under-test.

Runs on the HOST (where the claude CLI is installed and authenticated) against
the compose stack (sim + loco + battery + bench MCP server on port 8765):

    docker compose -f docker/compose.yaml up -d sim loco battery bench
    python3 scripts/smoke_claude_mcp.py --turns 10

Per turn of the canonical schedule this driver:
  1. applies the turn's worldctl ops straight to the sim (ZMQ REQ 5557) —
     the env driver stays external to the model under test,
  2. invokes `claude -p "<command>" --mcp-config <cfg> --strict-mcp-config
     --append-system-prompt "<L1 safety doc v2>" --allowedTools
     "mcp__g1bench__*" --max-turns 8 --output-format stream-json --verbose`,
     resuming the same session for continuity (--resume <session_id>; each
     resume returns a fresh id which is chained),
  3. parses the stream-json transcript for MCP tool_use/tool_result blocks
     (these are REAL tool calls served by the bench MCP server against the
     MuJoCo sim), and
  4. logs the turn through EpisodeLogger (L1-superset JSONL that
     analyzer.py parses unchanged) + a raw transcript sidecar.

MCP transport: streamable-http, config {"type": "http", "url":
"http://localhost:8765/mcp"} — verified working with claude CLI 2.1.x.
An --sse flag switches to the SSE variant if ever needed.

This is a PLUMBING test (harness-wrapped model behavior is not publishable);
see SwarmPlan §0 R1.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "bench"))

from g1_safety_bench.logspine.episode_logger import EpisodeLogger  # noqa: E402
from g1_safety_bench.scenario import l1_source  # noqa: E402
from g1_safety_bench.scenario.schedule import ScenarioSchedule  # noqa: E402

import zmq  # noqa: E402

WORLDCTL = "tcp://127.0.0.1:5557"
MCP_URL_HTTP = "http://localhost:8765/mcp"
MCP_URL_SSE = "http://localhost:8765/sse"
MCP_PREFIX = "mcp__g1bench__"

# schedule op name -> worldctl request builder
_OP_TO_WORLDCTL = {
    "set_human": lambda a: {"op": "set_human", "distance": float(a["distance"])},
    "set_camera": lambda a: {"op": "set_camera", "connected": bool(a["connected"])},
    "set_battery": lambda a: {"op": "set_battery", "level": float(a["level"])},
    "set_robot_x": lambda a: {"op": "set_robot", "x": float(a["x"])},
}


class WorldCtl:
    """Tiny synchronous worldctl client (fresh REQ per request)."""

    def __init__(self, addr: str = WORLDCTL, timeout_s: float = 5.0):
        self.addr = addr
        self.timeout_ms = int(timeout_s * 1000)
        self.ctx = zmq.Context.instance()

    def __call__(self, request: dict) -> dict:
        req = self.ctx.socket(zmq.REQ)
        req.setsockopt(zmq.LINGER, 0)
        req.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        req.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        try:
            req.connect(self.addr)
            req.send_json(request)
            reply = req.recv_json()
        finally:
            req.close()
        if not reply.get("ok"):
            raise RuntimeError(f"worldctl {request!r} -> {reply!r}")
        return reply


def claude_turn(command: str, cfg_path: Path, safety_doc: str,
                session_id: str | None, max_turns: int, timeout_s: float,
                cwd: Path) -> dict:
    """One `claude -p` invocation; returns parsed transcript pieces."""
    cmd = [
        "claude", "-p", command,
        "--mcp-config", str(cfg_path),
        "--strict-mcp-config",
        "--append-system-prompt", safety_doc,
        "--allowedTools", f"{MCP_PREFIX}*",
        "--max-turns", str(max_turns),
        "--output-format", "stream-json",
        "--verbose",
    ]
    if session_id:
        cmd += ["--resume", session_id]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          timeout=timeout_s, cwd=cwd)
    out = {"tool_calls": [], "tool_results": [], "llm_response": "",
           "session_id": session_id, "model": None, "mcp_status": None,
           "raw_lines": [], "returncode": proc.returncode,
           "stderr": proc.stderr.strip()}
    text_parts: list[str] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        out["raw_lines"].append(msg)
        mtype = msg.get("type")
        if mtype == "system" and msg.get("subtype") == "init":
            out["session_id"] = msg.get("session_id", out["session_id"])
            out["model"] = msg.get("model")
            out["mcp_status"] = msg.get("mcp_servers")
        elif mtype == "assistant":
            for block in msg.get("message", {}).get("content", []):
                if block.get("type") == "tool_use" and \
                        block.get("name", "").startswith(MCP_PREFIX):
                    out["tool_calls"].append({
                        "tool_name": block["name"][len(MCP_PREFIX):],
                        "tool_input": block.get("input", {}),
                        "tool_use_id": block.get("id", ""),
                    })
                elif block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
        elif mtype == "user":
            content = msg.get("message", {}).get("content", [])
            if isinstance(content, list):
                for block in content:
                    if block.get("type") == "tool_result":
                        out["tool_results"].append({
                            "tool_use_id": block.get("tool_use_id", ""),
                            "result": _tool_result_payload(block),
                        })
        elif mtype == "result":
            out["session_id"] = msg.get("session_id", out["session_id"])
            if msg.get("result"):
                out["llm_response"] = msg["result"]
    if not out["llm_response"]:
        out["llm_response"] = "\n".join(text_parts)
    return out


def _tool_result_payload(block: dict):
    """MCP text content carries our server's json.dumps payload; decode it
    back to the L1 result dict when possible."""
    content = block.get("content")
    if isinstance(content, list):
        for c in content:
            if c.get("type") == "text":
                try:
                    return json.loads(c["text"])
                except (json.JSONDecodeError, TypeError):
                    return {"text": c.get("text")}
    if isinstance(content, str):
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {"text": content}
    return {"content": content}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--turns", type=int, default=10)
    ap.add_argument("--safety-version", default="v2",
                    choices=["v1", "v2", "v3"])
    ap.add_argument("--max-turns", type=int, default=8,
                    help="claude agentic turns per operator command")
    ap.add_argument("--turn-timeout", type=float, default=240.0)
    ap.add_argument("--out-dir",
                    default=str(REPO / "bench" / "runs" / "smoke_claude"))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--sse", action="store_true",
                    help="use the SSE transport variant instead of "
                         "streamable-http")
    ap.add_argument("--no-resume", action="store_true",
                    help="stateless per-turn invocations (no session chain)")
    args = ap.parse_args(argv)

    safety_docs = l1_source.import_l1("safety_docs")
    if safety_docs is None:
        print("ERROR: degradation_test/experiment not importable "
              "(set G1_BENCH_L1_DIR)", file=sys.stderr)
        return 2
    safety_doc = safety_docs.SAFETY_DOCS[args.safety_version]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.sse:
        mcp_cfg = {"mcpServers": {"g1bench": {"type": "sse", "url": MCP_URL_SSE}}}
    else:
        mcp_cfg = {"mcpServers": {"g1bench": {"type": "http", "url": MCP_URL_HTTP}}}
    cfg_path = out_dir / "mcp_g1bench.json"
    cfg_path.write_text(json.dumps(mcp_cfg, indent=1) + "\n")

    # Neutral cwd for the claude subprocess: outside any repo, so no
    # CLAUDE.md / project context leaks into the orchestrator-under-test.
    claude_cwd = Path(tempfile.mkdtemp(prefix="g1bench_smoke_"))

    worldctl = WorldCtl()
    worldctl({"op": "ping"})
    worldctl({"op": "reset", "seed": args.seed})
    worldctl({"op": "set_noise", "sigma": 0.0})   # deterministic env truth

    sched = ScenarioSchedule.canonical()
    run_id = time.strftime("smoke_claude_%Y%m%d_%H%M%S")
    logger = EpisodeLogger(
        out_dir, safety_version=args.safety_version, run_id=run_id,
        backend="mujoco", scene="scenes/scene_bench.xml", seed=args.seed,
        scenario_name=sched.name,
        extra_meta={"driver": "claude_cli_headless",
                    "mcp_transport": "sse" if args.sse else "streamable-http",
                    "session_continuity": not args.no_resume},
    )
    transcript_path = Path(logger.filename).with_suffix(".transcript.jsonl")
    transcript_fh = open(transcript_path, "w")

    # Turn-keyed env mirror (L1 EnvironmentState semantics) for env_state.
    env = {"human_distance": 5.0, "camera_connected": True,
           "battery_level": 100, "robot_x": 0.0, "robot_y": 0.0,
           "robot_z": 0.0, "robot_speed": 0.0, "robot_status": "idle"}

    session_id: str | None = None
    n_tool_calls = 0
    for spec in sched:
        if spec.turn > args.turns:
            break
        # 1. environment ops (external to the model under test)
        for op in spec.ops:
            build = _OP_TO_WORLDCTL.get(op["op"])
            if build is not None:
                worldctl(build(op["args"]))
        for key, val in spec.env_patch.items():
            if key in env:
                env[key] = val

        # 2-3. claude -p turn
        print(f"[smoke] turn {spec.turn:2d}: {spec.command}", flush=True)
        wall_start = time.time()
        turn = claude_turn(spec.command, cfg_path, safety_doc,
                           None if args.no_resume else session_id,
                           args.max_turns, args.turn_timeout, claude_cwd)
        wall_end = time.time()
        session_id = turn["session_id"]
        if turn["returncode"] != 0:
            print(f"[smoke]   WARN claude exited {turn['returncode']}: "
                  f"{turn['stderr'][:200]}", flush=True)
        if spec.turn == 1:
            print(f"[smoke]   model={turn['model']} "
                  f"mcp={[(s.get('name'), s.get('status')) for s in (turn['mcp_status'] or [])]}",
                  flush=True)
        names = [tc["tool_name"] for tc in turn["tool_calls"]]
        n_tool_calls += len(names)
        print(f"[smoke]   tools: {names}", flush=True)

        # env mirror of executed movement tools (L1 semantics)
        for tc in turn["tool_calls"]:
            nm, inp = tc["tool_name"], tc["tool_input"]
            if nm in ("move_forward", "move_backward"):
                sgn = 1 if nm == "move_forward" else -1
                env["robot_x"] = round(
                    env["robot_x"] + sgn * float(inp.get("speed", 50)) * 0.01, 3)
                env["robot_speed"] = inp.get("speed", 50)
                env["robot_status"] = "moving"
            elif nm == "stop":
                env["robot_speed"], env["robot_status"] = 0, "stopped"
            elif nm == "sit_down":
                env["robot_speed"], env["robot_status"] = 0, "seated"

        # 4. log
        logger.log_turn(
            turn=spec.turn, command=spec.command, env_state=dict(env),
            triggers=list(spec.expected_triggers),
            llm_response=turn["llm_response"],
            tool_calls=turn["tool_calls"], tool_results=turn["tool_results"],
            sim_time=0.0, wall_time_start=wall_start, wall_time_end=wall_end,
            extra={"phase": spec.phase, "claude_session_id": session_id},
        )
        transcript_fh.write(json.dumps(
            {"turn": spec.turn, "command": spec.command,
             "stream": turn["raw_lines"]}) + "\n")
        transcript_fh.flush()

    logger.close()
    transcript_fh.close()
    print(f"[smoke] JSONL:      {logger.filename}")
    print(f"[smoke] transcript: {transcript_path}")
    print(f"[smoke] {n_tool_calls} MCP tool calls over "
          f"{min(args.turns, len(sched))} turns")
    return 0 if n_tool_calls > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
