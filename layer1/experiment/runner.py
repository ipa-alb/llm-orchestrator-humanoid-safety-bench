"""Main experiment loop — feeds commands to Claude API with tool-use.

Budget-optimized version:
  - Sliding window: only the last N messages are kept; older ones are summarized.
  - Change detection: environment state is only injected when it changes.
  - Prompt caching: system prompt and tool schemas use cache_control headers.
"""

import sys
import json
import time
import dataclasses
from typing import Any

import anthropic

from config import (
    MODEL, TEMPERATURE, MAX_TOKENS, REINJECTION_INTERVAL,
    HISTORY_WINDOW_SIZE, CHANGE_DETECTION_ENABLED, PROMPT_CACHING_ENABLED,
    HISTORY_CACHING_ENABLED,
)
from mock_tools import EnvironmentState, TOOL_SCHEMAS, handle_tool_call
from safety_docs import SAFETY_DOCS, V3_REMINDER
from scenarios import SCENARIO
from logger import ExperimentLogger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _summarize_old_messages(messages: list[dict], keep: int) -> list[dict]:
    """Replace messages older than the window with a compact summary message.

    Returns a new list: [summary_msg] + messages[-keep:]
    If the history is already within the window, returns it unchanged.
    """
    if len(messages) <= keep:
        return list(messages)  # always return a copy to avoid aliasing

    old = messages[:-keep]
    recent = messages[-keep:]

    # Build a compact summary of the old conversation
    summary_parts = []
    turn_label = ""
    for msg in old:
        role = msg["role"]
        content = msg.get("content", "")

        if role == "user":
            # Could be a string (operator command) or list (tool results)
            if isinstance(content, str):
                # Extract turn number if present
                if "[Operator command" in content:
                    turn_label = content.split("]")[0] + "]"
                    summary_parts.append(f"{turn_label}: {content.split(']: ')[-1][:80]}")
                elif "SAFETY REMINDER" in content:
                    summary_parts.append("[Safety reminder re-injected]")
                else:
                    summary_parts.append(f"User: {content[:80]}")
            elif isinstance(content, list):
                # Tool results — just note how many
                n = len(content)
                summary_parts.append(f"  [{n} tool result(s) returned]")
        elif role == "assistant":
            if isinstance(content, str):
                summary_parts.append(f"  Assistant: {content[:80]}")
            else:
                # Content blocks from API (tool_use + text)
                tool_names = []
                text_snippet = ""
                for block in content:
                    if hasattr(block, "type"):
                        if block.type == "tool_use":
                            tool_names.append(block.name)
                        elif block.type == "text" and block.text:
                            text_snippet = block.text[:60]
                    elif isinstance(block, dict):
                        if block.get("type") == "tool_use":
                            tool_names.append(block.get("name", "?"))
                        elif block.get("type") == "text":
                            text_snippet = block.get("text", "")[:60]
                parts = []
                if tool_names:
                    parts.append(f"tools=[{','.join(tool_names)}]")
                if text_snippet:
                    parts.append(text_snippet)
                summary_parts.append(f"  Assistant: {' | '.join(parts)}")

    # Cap summary to last 40 entries to keep it compact
    capped = summary_parts[-40:] if len(summary_parts) > 40 else summary_parts
    summary_text = (
        "CONVERSATION HISTORY SUMMARY (older turns, condensed to save context):\n"
        + "\n".join(capped)
    )

    summary_msg = {"role": "user", "content": summary_text}
    ack_msg = {
        "role": "assistant",
        "content": "Understood. I have the conversation history context and will continue following all safety rules.",
    }

    return [summary_msg, ack_msg] + recent


def _build_env_delta_message(prev: dict, curr: dict) -> str | None:
    """Return a human-readable string of what changed, or None if nothing did.

    On the very first turn (prev is empty), returns None so we don't send a
    noisy delta full of 'was None' values — the model can discover state via tools.
    """
    if not prev:
        return None
    changes = []
    for key in curr:
        old_val = prev.get(key)
        new_val = curr[key]
        if old_val != new_val:
            # Friendly names
            labels = {
                "human_distance": "Human distance",
                "camera_connected": "Camera",
                "battery_level": "Battery",
                "robot_x": "Robot X position",
            }
            label = labels.get(key, key)
            if key == "camera_connected":
                changes.append(f"{label}: {'connected' if new_val else 'DISCONNECTED'}")
            elif key == "human_distance":
                changes.append(f"{label}: {new_val}m (was {old_val}m)")
            elif key == "battery_level":
                changes.append(f"{label}: {new_val}% (was {old_val}%)")
            else:
                changes.append(f"{label}: {new_val} (was {old_val})")
    if not changes:
        return None
    return "[Environment update]: " + "; ".join(changes)


def _prepare_system_prompt(text: str) -> list[dict]:
    """Wrap the system prompt with cache_control if caching is enabled."""
    if PROMPT_CACHING_ENABLED:
        return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]
    return text


def _prepare_tools(schemas: list[dict]) -> list[dict]:
    """Add cache_control to the last tool schema (Anthropic caches up to that point)."""
    if not PROMPT_CACHING_ENABLED:
        return schemas
    # Deep copy to avoid mutating the original
    tools = [dict(s) for s in schemas]
    tools[-1] = dict(tools[-1])
    tools[-1]["cache_control"] = {"type": "ephemeral"}
    return tools


def _messages_with_cache_marker(msgs: list) -> list:
    """Return msgs with an ephemeral cache breakpoint on the LAST message.

    Anthropic caches the request prefix up to the marked block, so marking
    the newest message caches tools + system + the entire history; the next
    call reads that prefix back at 10% of the input price. The stored
    history is NEVER mutated — the marker lives only in a shallow per-call
    copy, so exactly one history breakpoint exists per request. Billing
    only: the tokens the model samples from are unchanged. (Ported verbatim
    from the L2 bench driver, humanoid_testing commit 3c883c2.)
    """
    if not HISTORY_CACHING_ENABLED or not msgs:
        return msgs
    last = msgs[-1]
    content = last.get("content")
    if isinstance(content, str):
        marked = [{"type": "text", "text": content,
                   "cache_control": {"type": "ephemeral"}}]
    elif isinstance(content, list) and content and isinstance(content[-1], dict):
        marked = list(content)
        marked[-1] = dict(marked[-1])
        marked[-1]["cache_control"] = {"type": "ephemeral"}
    else:
        return msgs
    return msgs[:-1] + [{**last, "content": marked}]


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(
    safety_version: str,
    num_turns: int = 100,
    run_id: str | None = None,
    verbose: bool = True,
) -> str:
    """Run the full experiment and return the log file path."""

    system_prompt = _prepare_system_prompt(SAFETY_DOCS[safety_version])
    tools = _prepare_tools(TOOL_SCHEMAS)
    client = anthropic.Anthropic(timeout=120.0)
    logger = ExperimentLogger(safety_version, run_id)
    env = EnvironmentState()

    # Conversation history (persists across turns)
    messages: list[dict] = []

    # Track previous env_patch for change detection
    prev_env_patch: dict = {}

    turns_to_run = min(num_turns, len(SCENARIO))

    for turn_idx in range(turns_to_run):
        turn_num = turn_idx + 1
        turn_spec = SCENARIO[turn_idx]

        # --- Apply environment patch ---
        for key, val in turn_spec["env_patch"].items():
            setattr(env, key, val)

        # --- V3: re-inject safety reminder every N turns ---
        if safety_version == "v3" and turn_num > 1 and turn_num % REINJECTION_INTERVAL == 0:
            messages.append({"role": "user", "content": V3_REMINDER})
            messages.append({
                "role": "assistant",
                "content": "Understood. Safety rules acknowledged and active.",
            })

        # --- Change detection: prepend env delta to the command ---
        command = turn_spec["command"]
        env_prefix = ""
        if CHANGE_DETECTION_ENABLED:
            delta_msg = _build_env_delta_message(prev_env_patch, turn_spec["env_patch"])
            if delta_msg:
                env_prefix = delta_msg + "\n"
            prev_env_patch = dict(turn_spec["env_patch"])

        user_content = f"{env_prefix}[Operator command — turn {turn_num}]: {command}"
        messages.append({"role": "user", "content": user_content})

        if verbose:
            print(f"\n{'='*60}", flush=True)
            print(f"Turn {turn_num} | Phase: {_phase_name(turn_idx)} | Triggers: {turn_spec['triggers']}")
            print(f"  Env: dist={env.human_distance}m cam={'ON' if env.camera_connected else 'OFF'} bat={env.battery_level}%")
            print(f"  Cmd: {command}", flush=True)
            if env_prefix:
                print(f"  Delta: {env_prefix.strip()}", flush=True)

        # --- Sliding window: trim history before API call ---
        windowed_messages = _summarize_old_messages(messages, HISTORY_WINDOW_SIZE)

        # --- Agentic tool-use loop ---
        all_tool_calls: list[dict] = []
        all_tool_results: list[dict] = []
        final_text = ""

        while True:
            # Retry with backoff on transient API errors (overloaded, rate limit)
            max_retries = 10
            for attempt in range(max_retries):
                try:
                    response = client.messages.create(
                        model=MODEL,
                        max_tokens=MAX_TOKENS,
                        temperature=TEMPERATURE,
                        system=system_prompt,
                        tools=tools,
                        messages=_messages_with_cache_marker(
                            windowed_messages),
                    )
                    break
                except (anthropic.APIStatusError, anthropic.APIConnectionError) as e:
                    # Only retry on transient errors (429, 529, 5xx)
                    if isinstance(e, anthropic.APIStatusError) and e.status_code < 429:
                        raise
                    wait = min(2 ** attempt, 60)  # cap at 60s
                    if verbose:
                        print(f"  [retry] {type(e).__name__}, waiting {wait}s (attempt {attempt+1}/{max_retries})", flush=True)
                    time.sleep(wait)
            else:
                raise RuntimeError(f"API still unavailable after {max_retries} retries")

            # Collect text and tool-use blocks from the response
            assistant_content = response.content
            text_parts = []
            tool_use_blocks = []

            for block in assistant_content:
                if block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "tool_use":
                    tool_use_blocks.append(block)

            # Append assistant message to BOTH full history and windowed copy
            messages.append({"role": "assistant", "content": assistant_content})
            windowed_messages.append({"role": "assistant", "content": assistant_content})

            if text_parts:
                final_text += " ".join(text_parts)

            # If no tool calls, we're done with this turn
            if not tool_use_blocks:
                break

            # Execute tool calls and build tool results
            tool_result_contents = []
            for tool_block in tool_use_blocks:
                tc = {
                    "tool_name": tool_block.name,
                    "tool_input": tool_block.input,
                    "tool_use_id": tool_block.id,
                }
                all_tool_calls.append(tc)

                result = handle_tool_call(tool_block.name, tool_block.input, env)
                all_tool_results.append({"tool_use_id": tool_block.id, "result": result})

                tool_result_contents.append({
                    "type": "tool_result",
                    "tool_use_id": tool_block.id,
                    "content": json.dumps(result),
                })

                if verbose:
                    print(f"  Tool: {tool_block.name}({tool_block.input}) → {result}", flush=True)

            # Send tool results back — append to both histories
            tool_msg = {"role": "user", "content": tool_result_contents}
            messages.append(tool_msg)
            windowed_messages.append(tool_msg)

        if verbose:
            print(f"  LLM: {final_text[:200]}{'...' if len(final_text) > 200 else ''}", flush=True)
            # Show token usage from response
            if hasattr(response, "usage"):
                u = response.usage
                cache_read = getattr(u, "cache_read_input_tokens", 0) or 0
                cache_create = getattr(u, "cache_creation_input_tokens", 0) or 0
                print(f"  Tokens: in={u.input_tokens} out={u.output_tokens} cache_read={cache_read} cache_create={cache_create}", flush=True)

        # --- Log turn ---
        logger.log_turn(
            turn=turn_num,
            command=command,
            env_state=dataclasses.asdict(env),
            triggers=turn_spec["triggers"],
            llm_response=final_text,
            tool_calls=all_tool_calls,
            tool_results=all_tool_results,
        )

    logger.close()
    print(f"\nExperiment complete. Log: {logger.filename}")
    return logger.filename


def _phase_name(turn_idx: int) -> str:
    if turn_idx < 25:
        return "baseline"
    if turn_idx < 50:
        return "proximity"
    if turn_idx < 75:
        return "stress"
    return "degradation"
