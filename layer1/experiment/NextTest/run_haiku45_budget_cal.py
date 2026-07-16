#!/usr/bin/env python3
"""Run Claude 3.5 Haiku with a CALIBRATED budget window.

Window size 20 is chosen to match ~5 effective turns of context, equivalent
to what the other corrective budget runs see. Rationale:
  - Anthropic runner generates ~4 messages per turn (batched tool calls,
    1 assistant + N tool results + 1 tool_use block ≈ 4 total)
  - 4 msg/turn × 5 turns = 20

This run tests whether Claude 3.5 Haiku regresses under context windowing
the same way GPT-4o-mini did (+20 issues, sensor neglect 4→69), or whether
it handles the sliding window more robustly.
Expected cost: ~$0.50 (window limits token growth).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

config.MODEL = "claude-haiku-4-5-20251001"
config.HISTORY_WINDOW_SIZE = 20          # calibrated: ≈ 5 effective turns
config.CHANGE_DETECTION_ENABLED = True
config.PROMPT_CACHING_ENABLED = False    # keep variables clean
config.RESULTS_DIR = "results_haiku45_budget_cal"

os.makedirs(config.RESULTS_DIR, exist_ok=True)

from runner import run_experiment
from analyzer import analyze

print("=" * 60)
print("Claude 3.5 Haiku — CALIBRATED BUDGET run")
print(f"  Model:            {config.MODEL}")
print(f"  Sliding window:   ON (window={config.HISTORY_WINDOW_SIZE} ≈ 5 effective turns)")
print(f"  Change detection: {config.CHANGE_DETECTION_ENABLED}")
print(f"  Prompt caching:   {config.PROMPT_CACHING_ENABLED}")
print(f"  Results dir:      {config.RESULTS_DIR}")
print("  NOTE: Window=20 because Anthropic runner uses ~4 messages/turn.")
print("=" * 60)

log_path = run_experiment(
    safety_version="v2",
    num_turns=100,
    run_id="haiku45_budget_cal",
    verbose=True,
)

print("\n--- Running analysis ---")
analyze(log_path)
