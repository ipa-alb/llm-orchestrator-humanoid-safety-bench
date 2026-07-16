#!/usr/bin/env python3
"""Run GPT-4o-mini with ALL budget optimizations disabled.

Controlled no-budget baseline. Replaces the original 88-turn run that crashed
on a rate limit with no retry logic and had an unknown window configuration.
Expected cost: ~$0.40.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import config

config.MODEL = "gpt-4o-mini"
config.BASE_URL = None
config.HISTORY_WINDOW_SIZE = 99999       # no sliding window
config.CHANGE_DETECTION_ENABLED = False  # full env state injected every turn
config.RESULTS_DIR = "results_chatgpt_nobudget"

import os
os.makedirs(config.RESULTS_DIR, exist_ok=True)

from runner_openai import run_experiment
from analyzer import analyze

print("=" * 60)
print("GPT-4o-mini — CONTROLLED NO-BUDGET baseline")
print(f"  Model:            {config.MODEL}")
print(f"  Sliding window:   OFF (window={config.HISTORY_WINDOW_SIZE})")
print(f"  Change detection: {config.CHANGE_DETECTION_ENABLED}")
print(f"  Results dir:      {config.RESULTS_DIR}")
print("  NOTE: Expected cost ~$0.40.")
print("=" * 60)

log_path = run_experiment(
    safety_version="v2",
    num_turns=100,
    run_id="chatgpt_nobudget",
    verbose=True,
)

print("\n--- Running analysis ---")
analyze(log_path)
