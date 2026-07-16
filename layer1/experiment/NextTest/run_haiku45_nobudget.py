#!/usr/bin/env python3
"""Run Claude 3.5 Haiku (haiku-4-5) with ALL budget optimizations disabled.

Controlled no-budget baseline — mirrors run_haiku3.py but for the newer model.
Replaces the original 92-turn run that crashed due to credit exhaustion.
Expected cost: ~$5 (quadratic token growth, same as original Run 1).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

config.MODEL = "claude-haiku-4-5-20251001"
config.HISTORY_WINDOW_SIZE = 99999       # no sliding window
config.CHANGE_DETECTION_ENABLED = False  # full env state injected every turn
config.PROMPT_CACHING_ENABLED = False    # no caching
config.RESULTS_DIR = "results_haiku45_nobudget"

import os
os.makedirs(config.RESULTS_DIR, exist_ok=True)

from runner import run_experiment
from analyzer import analyze

print("=" * 60)
print("Claude 3.5 Haiku — CONTROLLED NO-BUDGET baseline")
print(f"  Model:            {config.MODEL}")
print(f"  Sliding window:   OFF (window={config.HISTORY_WINDOW_SIZE})")
print(f"  Change detection: {config.CHANGE_DETECTION_ENABLED}")
print(f"  Prompt caching:   {config.PROMPT_CACHING_ENABLED}")
print(f"  Results dir:      {config.RESULTS_DIR}")
print("  NOTE: Expected cost ~$5 due to quadratic token growth.")
print("=" * 60)

log_path = run_experiment(
    safety_version="v2",
    num_turns=100,
    run_id="haiku45_nobudget",
    verbose=True,
)

print("\n--- Running analysis ---")
analyze(log_path)
