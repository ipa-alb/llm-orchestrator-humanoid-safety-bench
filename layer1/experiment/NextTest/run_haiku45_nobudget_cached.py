#!/usr/bin/env python3
"""Cache-effect control rerun of run_haiku45_nobudget.py, HISTORY_CACHING_ENABLED=True.

Identical config to the published no-budget baseline (results_haiku45_nobudget)
except that the newly ported Anthropic history caching (config.HISTORY_CACHING_ENABLED,
commit 43fb9e7) is explicitly ON. History caching is billing-only by design — the
model samples byte-identical tokens — so results are expected to match the published
run up to sampling variance (TEMPERATURE=0.2). Expected cost: ~$1 (vs ~$5 uncached).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

config.MODEL = "claude-haiku-4-5-20251001"
config.HISTORY_WINDOW_SIZE = 99999       # no sliding window
config.CHANGE_DETECTION_ENABLED = False  # full env state injected every turn
config.PROMPT_CACHING_ENABLED = False    # as in the published template (no static-prefix markers)
config.HISTORY_CACHING_ENABLED = True    # the variable under test: moving breakpoint on last message
config.RESULTS_DIR = "results_haiku45_nobudget_cached_20260714"

import os
os.makedirs(config.RESULTS_DIR, exist_ok=True)

from runner import run_experiment
from analyzer import analyze

print("=" * 60)
print("Claude 3.5 Haiku — NO-BUDGET cache-effect control rerun")
print(f"  Model:            {config.MODEL}")
print(f"  Sliding window:   OFF (window={config.HISTORY_WINDOW_SIZE})")
print(f"  Change detection: {config.CHANGE_DETECTION_ENABLED}")
print(f"  Prompt caching:   {config.PROMPT_CACHING_ENABLED}")
print(f"  History caching:  {config.HISTORY_CACHING_ENABLED}")
print(f"  Results dir:      {config.RESULTS_DIR}")
print("  NOTE: control for the ported history caching; expect ~$1 vs ~$5 uncached.")
print("=" * 60)

log_path = run_experiment(
    safety_version="v2",
    num_turns=100,
    run_id="haiku45_nobudget",
    verbose=True,
)

print("\n--- Running analysis ---")
analyze(log_path)
