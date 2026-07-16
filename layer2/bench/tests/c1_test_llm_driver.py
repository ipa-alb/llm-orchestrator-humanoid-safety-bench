#!/usr/bin/env python3
"""C1 unit tests: real-LLM driver with FAKED SDK clients (no API keys, no
network, no anthropic/openai package needed — the fakes never raise, so the
lazy SDK imports in the retry/transient paths are never touched).

Covers: tool-call loop against the REAL ToolExecutor (env mutation + result
formatting), 20-call watchdog on both API paths, exact-L1 tool-result JSON
formatting, usage accumulation, v3 reminder injection, unknown tools, and the
--budget condition (sliding window incl. OpenAI tool-group-safe cut, prompt
caching, env change detection).

Run (host with the degradation_test sibling, or inside g1-base with /l1):

    python3 bench/tests/c1_test_llm_driver.py
    docker compose -f docker/compose.yaml run --rm --no-deps bench \
        python3 bench/tests/c1_test_llm_driver.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))

from g1_safety_bench.scenario import l1_source  # noqa: E402
from g1_safety_bench.scenario.llm_driver import (  # noqa: E402
    MAX_TOOL_CALLS_PER_TURN, LLMDriver)
from g1_safety_bench.scenario.runner_sim import EnvState, ToolExecutor  # noqa: E402


# --------------------------------------------------------------- fake SDKs
class FakeAnthropicClient:
    """Scripted stand-in for anthropic.Anthropic — records every request."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.requests: list[dict] = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        # Snapshot the messages list: the driver mutates it after the call.
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        if len(self._responses) > 1:
            return self._responses.pop(0)
        return self._responses[0]  # repeat the last response forever


def a_text(text):
    return SimpleNamespace(type="text", text=text)


def a_tool(tid, name, inp):
    return SimpleNamespace(type="tool_use", id=tid, name=name, input=inp)


def a_resp(blocks, input_tokens=100, output_tokens=10):
    return SimpleNamespace(
        content=blocks,
        usage=SimpleNamespace(input_tokens=input_tokens,
                              output_tokens=output_tokens,
                              cache_read_input_tokens=0,
                              cache_creation_input_tokens=0))


class FakeOpenAIClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.requests: list[dict] = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        if len(self._responses) > 1:
            return self._responses.pop(0)
        return self._responses[0]


def o_tc(tid, name, arguments):
    return SimpleNamespace(
        id=tid, function=SimpleNamespace(name=name, arguments=arguments))


def o_resp(content, tool_calls=None, prompt_tokens=100, completion_tokens=10):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls or [])
    return SimpleNamespace(
        choices=[SimpleNamespace(message=msg)],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens,
                              completion_tokens=completion_tokens,
                              prompt_tokens_details=None))


def make_tools(turn=1):
    return ToolExecutor(EnvState(), backend=None, turn=turn)


def make_driver(api, client, **kw):
    return LLMDriver(api=api, client=client, verbose=False, **kw)


# -------------------------------------------------------------------- tests
def test_l1_sources_wired():
    sd = l1_source.import_l1("safety_docs")
    mt = l1_source.import_l1("mock_tools")
    mto = l1_source.import_l1("mock_tools_openai")
    assert sd and mt and mto, "L1 checkout required for these tests"

    d = make_driver("anthropic", FakeAnthropicClient([a_resp([a_text("x")])]))
    assert d.system_prompt == sd.SAFETY_DOCS["v2"]
    assert d.tool_schemas == mt.TOOL_SCHEMAS, "Anthropic schemas must be L1-verbatim"

    d2 = make_driver("openai", FakeOpenAIClient([o_resp("x")]))
    assert d2.tool_schemas == mto.TOOL_SCHEMAS, "OpenAI schemas must be L1-verbatim"
    print("  ok: system prompt + tool schemas imported verbatim from L1")


def test_anthropic_tool_loop():
    client = FakeAnthropicClient([
        a_resp([a_text("Checking."),
                a_tool("t1", "get_human_distance", {}),
                a_tool("t2", "move_forward", {"speed": 50})]),
        a_resp([a_text("Done.")]),
    ])
    d = make_driver("anthropic", client)
    tools = make_tools(turn=1)
    out = d("Move forward at 50% speed.", tools)

    # Two API calls; final text concatenates per-response text like L1.
    assert out["llm_api_calls"] == 2
    assert out["llm_response"] == "Checking.Done."
    assert out["llm_tool_calls"] == 2

    # Tools executed through the real ToolExecutor: env mutated + recorded.
    assert tools.env.robot_x == 0.5 and tools.env.robot_status == "moving"
    assert [tc["tool_name"] for tc in tools.tool_calls] == \
        ["get_human_distance", "move_forward"]

    # Exact L1 tool-result formatting: content == json.dumps(result dict).
    # (Assert on the driver history: the anthropic path sends its live
    # `messages` list, so the recorded request aliases later appends.)
    second_req = client.requests[1]
    tr_msg = d.messages[2]
    assert tr_msg["role"] == "user"
    contents = {b["tool_use_id"]: b["content"] for b in tr_msg["content"]}
    assert json.loads(contents["t2"]) == {"status": "ok", "speed": 50.0}
    assert set(json.loads(contents["t1"])) == {"distance_meters"}

    # Request shape mirrors L1 runner.py (system prompt + L1 schemas + temp).
    assert second_req["system"] == d.system_prompt
    assert second_req["tools"] is d.tool_schemas
    assert second_req["temperature"] == d.temperature

    # Usage accumulation (per turn + episode totals).
    assert out["llm_usage"]["input_tokens"] == 200
    assert out["llm_usage"]["output_tokens"] == 20
    assert d.total_usage["input_tokens"] == 200

    # Full-history (no-budget): second turn's request contains turn 1 verbatim.
    client2_len_before = len(d.messages)
    d("Stop.", make_tools(turn=2))
    assert len(d.messages) > client2_len_before
    assert d.messages[0]["content"] == "[Operator command — turn 1]: " \
        "Move forward at 50% speed."
    print("  ok: anthropic tool loop, L1 formatting, usage, full history")


def test_openai_tool_loop():
    client = FakeOpenAIClient([
        o_resp(None, [o_tc("c1", "get_battery_level", "{}"),
                      o_tc("c2", "stop", "{}")]),
        o_resp("Stopped."),
    ])
    d = make_driver("openai", client)
    tools = make_tools(turn=1)
    out = d("Stop.", tools)

    assert out["llm_response"] == "Stopped."
    assert out["llm_tool_calls"] == 2
    assert tools.env.robot_status == "stopped"

    req2 = client.requests[1]
    # OpenAI path: system message prepended each call (L1 runner_openai.py).
    assert req2["messages"][0] == {"role": "system", "content": d.system_prompt}
    tool_msgs = [m for m in req2["messages"] if m.get("role") == "tool"]
    assert len(tool_msgs) == 2
    by_id = {m["tool_call_id"]: m["content"] for m in tool_msgs}
    assert json.loads(by_id["c2"]) == {"status": "ok"}
    assert json.loads(by_id["c1"]) == {"battery_percent": 100}

    # Assistant message serialization keeps the tool_calls (L1 shape).
    hist_assistant = [m for m in d.messages
                     if m.get("role") == "assistant" and m.get("tool_calls")]
    assert hist_assistant and \
        hist_assistant[0]["tool_calls"][0]["function"]["name"] == "get_battery_level"
    print("  ok: openai tool loop, tool messages, L1 formatting")


def test_watchdog_anthropic():
    # Every response demands 5 more tool calls -> watchdog trips at >= 20.
    spam = a_resp([a_tool(f"t{i}", "stop", {}) for i in range(5)])
    d = make_driver("anthropic", FakeAnthropicClient([spam]))
    tools = make_tools(turn=1)
    out = d("Stop.", tools)

    assert "[SAFETY GUARD: turn aborted after 20 tool calls]" in out["llm_response"]
    assert out["llm_api_calls"] == 4                    # 5+5+5+5 -> abort
    # Only the first 3 batches (15 calls) were executed on the world; the
    # aborted batch was answered with guard errors, never executed.
    assert len(tools.tool_calls) == 15
    last_results = d.messages[-1]["content"]
    assert all(json.loads(b["content"]).get("error", "").startswith(
        "[SAFETY GUARD") for b in last_results)
    print("  ok: anthropic watchdog aborts at 20, pending calls answered not executed")


def test_watchdog_openai():
    spam = o_resp(None, [o_tc(f"c{i}", "stop", "{}") for i in range(7)])
    d = make_driver("openai", FakeOpenAIClient([spam]))
    tools = make_tools(turn=1)
    out = d("Stop.", tools)

    # 7 + 7 + 7 = 21 >= 20 -> abort on the 3rd batch.
    assert out["llm_api_calls"] == 3
    assert "[SAFETY GUARD: turn aborted after 21 tool calls]" in out["llm_response"]
    assert len(tools.tool_calls) == 14
    guard_msgs = [m for m in d.messages if m.get("role") == "tool"][-7:]
    assert all("SAFETY GUARD" in m["content"] for m in guard_msgs)
    print("  ok: openai watchdog (L1 semantics) at >= 20 tool calls")


def test_unknown_tool_and_bad_args():
    client = FakeOpenAIClient([
        o_resp(None, [o_tc("c1", "self_destruct", "{}"),
                      o_tc("c2", "move_forward", "not json")]),
        o_resp("done"),
    ])
    d = make_driver("openai", client)
    tools = make_tools(turn=1)
    d("do something weird", tools)
    by_id = {m["tool_call_id"]: json.loads(m["content"])
             for m in d.messages if m.get("role") == "tool"}
    assert by_id["c1"] == {"error": "Unknown tool: self_destruct"}  # L1 payload
    assert "Unparseable tool arguments" in by_id["c2"]["error"]
    print("  ok: unknown tool -> L1 error payload; bad JSON args reported")


def test_v3_reminder_injection():
    sd = l1_source.import_l1("safety_docs")
    cfg = l1_source.import_l1("config")
    d = make_driver("anthropic", FakeAnthropicClient([a_resp([a_text("ok")])]),
                    safety_version="v3")
    d(f"cmd", make_tools(turn=cfg.REINJECTION_INTERVAL))
    assert d.messages[0]["content"] == sd.V3_REMINDER
    assert d.messages[1]["role"] == "assistant"
    assert d.messages[2]["content"].startswith(
        f"[Operator command — turn {cfg.REINJECTION_INTERVAL}]")

    # No injection on non-multiple turns.
    d2 = make_driver("anthropic", FakeAnthropicClient([a_resp([a_text("ok")])]),
                     safety_version="v3")
    d2("cmd", make_tools(turn=7))
    assert d2.messages[0]["content"].startswith("[Operator command — turn 7]")
    print("  ok: v3 reminder injected every REINJECTION_INTERVAL turns")


def test_budget_window_anthropic():
    from g1_safety_bench.scenario.llm_driver import (
        _summarize_old_messages_anthropic)
    cfg = l1_source.import_l1("config")
    keep = cfg.HISTORY_WINDOW_SIZE

    client = FakeAnthropicClient([a_resp([a_text("ok")])])
    d = make_driver("anthropic", client, budget=True)

    # Prompt caching (L1 _prepare_system_prompt/_prepare_tools): system is a
    # cache_control block list, last tool carries cache_control, and the
    # verbatim schemas are NOT mutated.
    d("Turn one.", make_tools(turn=1))
    req = client.requests[0]
    assert isinstance(req["system"], list)
    assert req["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert req["tools"][-1].get("cache_control") == {"type": "ephemeral"}
    assert "cache_control" not in d.tool_schemas[-1]

    # No-tool turns add 2 messages each; run past the window.
    for turn in range(2, keep):
        d(f"Filler {turn}.", make_tools(turn=turn))
    n_full = len(d.messages)
    assert n_full > keep
    last_req_msgs = client.requests[-1]["messages"]
    # L1 runner.py: [summary, ack] + last-keep verbatim, window applied
    # BEFORE the turn's own user message was the state at call time.
    assert last_req_msgs[0]["role"] == "user"
    assert last_req_msgs[0]["content"].startswith(
        "CONVERSATION HISTORY SUMMARY")
    assert last_req_msgs[1]["role"] == "assistant"
    assert len(last_req_msgs) == keep + 2
    # Full history is untouched (verbatim, no summary inside).
    assert d.messages[0]["content"].endswith("Turn one.")
    assert all("CONVERSATION HISTORY SUMMARY" not in str(m.get("content", ""))
               for m in d.messages)
    # The summarizer output itself matches L1's shape on block content.
    win = _summarize_old_messages_anthropic(d.messages, keep)
    assert len(win) == keep + 2 and "tools=[" not in win[0]["content"]
    print("  ok: budget anthropic — window, caching, full history intact")


def test_budget_window_openai_safe_cut():
    from g1_safety_bench.scenario.llm_driver import (
        _summarize_old_messages_openai)
    # Handcraft a history where the naive cut would orphan 'tool' messages.
    msgs = []
    for t in range(1, 8):
        msgs += [
            {"role": "user", "content": f"[Operator command — turn {t}]: go"},
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": f"c{t}", "type": "function",
                             "function": {"name": "stop", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": f"c{t}", "content": "{}"},
            {"role": "assistant", "content": "done"},
        ]
    for keep in range(1, len(msgs)):
        win = _summarize_old_messages_openai(msgs, keep)
        if len(msgs) <= keep:
            continue
        body = win[2:]  # after [summary, ack]
        assert body, f"empty window for keep={keep}"
        assert body[0]["role"] not in ("tool",), \
            f"orphaned tool message at window start (keep={keep})"
        if body[0]["role"] == "assistant":
            assert not (body[0].get("tool_calls") is None
                        and body[0]["content"] is None)
    print("  ok: budget openai — cut never orphans a tool message")


def test_budget_change_detection():
    client = FakeAnthropicClient([a_resp([a_text("ok")])])
    d = make_driver("anthropic", client, budget=True)

    t1 = make_tools(turn=1)
    t1.env_patch = {"human_distance": 5.0, "battery_level": 100}
    d("Go.", t1)
    # First turn: prev patch empty -> no delta (L1 returns None).
    assert d.messages[0]["content"] == "[Operator command — turn 1]: Go."

    t2 = make_tools(turn=2)
    t2.env_patch = {"human_distance": 0.8, "battery_level": 100,
                    "camera_connected": False}
    d("Go on.", t2)
    turn2_user = d.messages[2]["content"]
    assert turn2_user.startswith("[Environment update]: ")
    assert "Human distance: 0.8m (was 5.0m)" in turn2_user
    assert "Camera: DISCONNECTED" in turn2_user
    assert "Battery" not in turn2_user  # unchanged key -> not reported
    assert turn2_user.endswith("[Operator command — turn 2]: Go on.")

    # No-budget: same patches, no delta message ever (L1 no-budget parity).
    client2 = FakeAnthropicClient([a_resp([a_text("ok")])])
    d2 = make_driver("anthropic", client2)
    u1, u2 = make_tools(turn=1), make_tools(turn=2)
    u1.env_patch = {"human_distance": 5.0}
    u2.env_patch = {"human_distance": 0.8}
    d2("Go.", u1)
    d2("Go on.", u2)
    assert d2.messages[2]["content"] == "[Operator command — turn 2]: Go on."
    print("  ok: budget change detection — L1 delta format; no-budget clean")


def test_reasoning_effort_defaults():
    from g1_safety_bench.scenario.llm_driver import make_backend_driver
    fake = FakeOpenAIClient([o_resp("x")])

    # local: thinking off by default (Ollama reasoning_effort="none")
    d = make_backend_driver("local", client=fake, verbose=False)
    assert d.reasoning_effort == "none" and d.model == "qwen3:8b"

    # gemini @ default 2.5-flash: NO reasoning param (L1 parity)
    import os
    os.environ.setdefault("GEMINI_API_KEY", "test-key")
    d2 = make_backend_driver("gemini", client=fake, verbose=False)
    assert d2.reasoning_effort is None

    # gemini @ 3.x model: floor is "minimal" (thinking can't be fully off)
    d3 = make_backend_driver("gemini", model="gemini-3.5-flash",
                             client=fake, verbose=False)
    assert d3.reasoning_effort == "minimal"
    try:
        make_backend_driver("gemini", model="gemini-3.5-flash",
                            reasoning_effort="none", client=fake,
                            verbose=False)
        raise AssertionError("gemini-3.x + none must be rejected")
    except ValueError as e:
        assert "minimal" in str(e)
    print("  ok: reasoning-effort defaults (local none, gemini 2.5 L1-parity,"
          " gemini 3.x minimal)")


def test_anthropic_history_cache_marker():
    import os
    # No-budget: one moving breakpoint per request, history stays L1-pristine.
    client = FakeAnthropicClient([
        a_resp([a_text("Checking."), a_tool("t1", "get_human_distance", {})]),
        a_resp([a_text("Done.")]),
    ])
    d = make_driver("anthropic", client)
    assert d.history_caching and not d.prompt_caching
    d("Move forward at 10% speed.", make_tools(turn=1))
    assert len(client.requests) == 2

    for req in client.requests:
        marked = [b for m in req["messages"]
                  for b in (m["content"] if isinstance(m["content"], list)
                            else [])
                  if isinstance(b, dict) and "cache_control" in b]
        assert len(marked) == 1, "exactly one breakpoint per request"
        last_content = req["messages"][-1]["content"]
        assert isinstance(last_content, list)
        assert last_content[-1]["cache_control"] == {"type": "ephemeral"}

    # The 2nd request re-sends turn-1's command UNMARKED, in raw L1 string
    # form (the marker must never leak into or mutate stored history).
    assert client.requests[1]["messages"][0]["content"] == \
        "[Operator command — turn 1]: Move forward at 10% speed."
    assert d.messages[0]["content"] == \
        "[Operator command — turn 1]: Move forward at 10% speed."
    for m in d.messages:
        for b in (m["content"] if isinstance(m["content"], list) else []):
            assert not (isinstance(b, dict) and "cache_control" in b)

    # Budget path: history marker OFF (L1-exact requests); the L1 markers
    # on system + last tool schema are still applied.
    d2 = make_driver("anthropic",
                     FakeAnthropicClient([a_resp([a_text("x")])]),
                     budget=True)
    assert not d2.history_caching and d2.prompt_caching

    # Kill switch.
    os.environ["G1_BENCH_ANTHROPIC_CACHE"] = "0"
    try:
        d3 = make_driver("anthropic",
                         FakeAnthropicClient([a_resp([a_text("x")])]))
        assert not d3.history_caching
    finally:
        del os.environ["G1_BENCH_ANTHROPIC_CACHE"]
    print("  ok: anthropic moving cache breakpoint — one marker per request,"
          " pristine history, budget path untouched, env kill switch")


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"[c1_test_llm_driver] running {len(tests)} tests")
    for t in tests:
        t()
    print("[c1_test_llm_driver] ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
