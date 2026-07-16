"""Main experiment loop — feeds commands to OpenAI API with tool-use.

Budget-optimized version:
  - Sliding window: only the last N messages are kept; older ones are summarized.
  - Change detection: environment state is only injected when it changes.
"""

import sys
import json
import time
import dataclasses
from typing import Any

import openai

from config import (
    MODEL, TEMPERATURE, MAX_TOKENS, REINJECTION_INTERVAL, BASE_URL,
    HISTORY_WINDOW_SIZE, CHANGE_DETECTION_ENABLED, REASONING_EFFORT,
)
from mock_tools_openai import EnvironmentState, TOOL_SCHEMAS, handle_tool_call
from safety_docs import SAFETY_DOCS, V3_REMINDER
from scenarios import SCENARIO
from logger import ExperimentLogger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _summarize_old_messages(messages: list[dict], keep: int) -> list[dict]:
    """Replace messages older than the window with a compact summary message.

    Returns a new list: [summary_msg, ack_msg] + messages[-keep:]
    If the history is already within the window, returns it unchanged.

    OpenAI requires that 'tool' messages follow the 'assistant' message with
    'tool_calls' that produced them. So we adjust the cut point to avoid
    splitting in the middle of a tool-call group.
    """
    if len(messages) <= keep:
        return list(messages)

    # Find a safe cut point: walk the boundary backward so recent[] doesn't
    # start with an orphaned 'tool' or 'assistant' response to tool calls.
    cut = len(messages) - keep
    while cut < len(messages) and messages[cut]["role"] in ("tool", "assistant") and cut > 0:
        # If it's a tool message, it's orphaned from its assistant — include it
        # If it's an assistant message, check if it's a continuation (after tools)
        if messages[cut]["role"] == "tool":
            cut -= 1
        elif messages[cut]["role"] == "assistant" and cut > 0 and messages[cut - 1]["role"] == "tool":
            cut -= 1
        else:
            break
    # Also pull in the assistant message that started the tool_calls group
    if cut > 0 and messages[cut]["role"] == "tool":
        cut -= 1

    old = messages[:cut]
    recent = messages[cut:]

    summary_parts = []
    for msg in old:
        role = msg["role"]
        content = msg.get("content", "")

        if role == "user":
            if isinstance(content, str):
                if "[Operator command" in content:
                    turn_label = content.split("]")[0] + "]"
                    summary_parts.append(f"{turn_label}: {content.split(']: ')[-1][:80]}")
                elif "SAFETY REMINDER" in content:
                    summary_parts.append("[Safety reminder re-injected]")
                else:
                    summary_parts.append(f"User: {content[:80]}")
        elif role == "assistant":
            if isinstance(content, str) and content:
                summary_parts.append(f"  Assistant: {content[:80]}")
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                names = [tc["function"]["name"] for tc in tool_calls]
                summary_parts.append(f"  Assistant: tools=[{','.join(names)}]")
        elif role == "tool":
            summary_parts.append(f"  [tool result]")

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
    """Return a human-readable string of what changed, or None if nothing did."""
    if not prev:
        return None
    changes = []
    for key in curr:
        old_val = prev.get(key)
        new_val = curr[key]
        if old_val != new_val:
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

    system_prompt = SAFETY_DOCS[safety_version]
    client_kwargs = {"timeout": 120.0}
    if BASE_URL:
        client_kwargs["base_url"] = BASE_URL
    client = openai.OpenAI(**client_kwargs)
    logger = ExperimentLogger(safety_version, run_id)
    env = EnvironmentState()

    # Conversation history (persists across turns)
    messages: list[dict] = []

    # Opt-in thinking cap (local Ollama models). Mutable copy: dropped for
    # the rest of the episode if the server 400s on it (mirrors the L2
    # bench driver's fallback).
    reasoning_effort = REASONING_EFFORT

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
        MAX_TOOL_CALLS_PER_TURN = 20

        tool_call_count = 0
        while True:
            # Retry with exponential backoff for rate limits
            for attempt in range(15):
                try:
                    response = client.chat.completions.create(
                        model=MODEL,
                        max_tokens=MAX_TOKENS,
                        temperature=TEMPERATURE,
                        tools=TOOL_SCHEMAS,
                        messages=[{"role": "system", "content": system_prompt}] + windowed_messages,
                        **({"reasoning_effort": reasoning_effort}
                           if reasoning_effort else {}),
                    )
                    break
                except openai.BadRequestError as e:
                    # Non-thinking models may reject the reasoning_effort
                    # field; drop it for the rest of the episode (L2 parity).
                    if reasoning_effort and ("reason" in str(e).lower()
                                             or "think" in str(e).lower()):
                        if verbose:
                            print(f"  [note] {MODEL} rejected reasoning_effort="
                                  f"{reasoning_effort!r}; dropping it", flush=True)
                        reasoning_effort = None
                        continue
                    raise
                except openai.RateLimitError:
                    if attempt == 14:
                        raise
                    wait = min(60, 2 ** attempt)  # 1..60s, capped; TPM window resets each minute
                    if verbose:
                        print(f"  [rate limit] waiting {wait}s before retry ({attempt+1}/15)...", flush=True)
                    time.sleep(wait)

            message = response.choices[0].message

            # Collect text and tool calls from the response
            if message.content:
                final_text += message.content

            tool_calls = message.tool_calls or []

            # Append assistant message to BOTH full history and windowed copy
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": message.content}
            if tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ]
            messages.append(assistant_msg)
            windowed_messages.append(assistant_msg)

            # If no tool calls, we're done with this turn
            if not tool_calls:
                break

            # Guard against infinite tool-call loops (e.g. Gemini stop() spam)
            tool_call_count += len(tool_calls)
            if tool_call_count >= MAX_TOOL_CALLS_PER_TURN:
                final_text += f"\n[SAFETY GUARD: turn aborted after {tool_call_count} tool calls]"
                break

            # Execute tool calls and send results back
            for tc in tool_calls:
                tool_name = tc.function.name
                tool_input = json.loads(tc.function.arguments)

                all_tool_calls.append({
                    "tool_name": tool_name,
                    "tool_input": tool_input,
                    "tool_use_id": tc.id,
                })

                result = handle_tool_call(tool_name, tool_input, env)
                all_tool_results.append({"tool_use_id": tc.id, "result": result})

                # Each tool result is a separate message in OpenAI format
                tool_msg = {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                }
                messages.append(tool_msg)
                windowed_messages.append(tool_msg)

                if verbose:
                    print(f"  Tool: {tool_name}({tool_input}) → {result}", flush=True)

        if verbose:
            print(f"  LLM: {final_text[:200]}{'...' if len(final_text) > 200 else ''}", flush=True)

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
