#!/usr/bin/env python3
"""Run GPT-4o-mini with a CALIBRATED budget window.

Window size 32 is chosen to match ~5 effective turns of context, equivalent
to what the Anthropic budget runs see at window=20. Rationale:
  - GPT-4o-mini generates ~6 messages per turn (sequential tool calls)
  - 6 msg/turn × 5 turns = 30; use 32 for buffer
  - Anthropic runner: ~4 msg/turn × 5 turns = 20 (the existing budget setting)

This run tests whether the sensor-neglect regression (20→75) in the original
budget run disappears when the effective context depth is equalised.
Expected cost: ~$0.10.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import config

config.MODEL = "gpt-4o-mini"
config.BASE_URL = None
config.HISTORY_WINDOW_SIZE = 32          # calibrated: ≈ 5 effective turns
config.CHANGE_DETECTION_ENABLED = True
config.RESULTS_DIR = "results_chatgpt_budget_cal"

import os
os.makedirs(config.RESULTS_DIR, exist_ok=True)

from runner_openai import run_experiment
from analyzer import analyze

print("=" * 60)
print("GPT-4o-mini — CALIBRATED BUDGET run")
print(f"  Model:            {config.MODEL}")
print(f"  Sliding window:   ON (window={config.HISTORY_WINDOW_SIZE} ≈ 5 effective turns)")
print(f"  Change detection: {config.CHANGE_DETECTION_ENABLED}")
print(f"  Results dir:      {config.RESULTS_DIR}")
print("  NOTE: Window calibrated to match Anthropic budget depth.")
print("=" * 60)

log_path = run_experiment(
    safety_version="v2",
    num_turns=100,
    run_id="chatgpt_budget_cal",
    verbose=True,
)

print("\n--- Running analysis ---")
analyze(log_path)
