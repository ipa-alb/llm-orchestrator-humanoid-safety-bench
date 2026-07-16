#!/usr/bin/env python3
"""Cache-effect control rerun of run_haiku45_budget_cal.py, HISTORY_CACHING_ENABLED=True.

Identical calibrated-budget config to the published run (results_haiku45_budget_cal:
window=20, change detection ON, prompt caching OFF) except that the newly ported
Anthropic history caching (config.HISTORY_CACHING_ENABLED, commit 43fb9e7) is
explicitly ON. History caching is billing-only by design — the model samples
byte-identical tokens. With the sliding window trimming the prefix every turn,
little/no cache reuse is expected here; this arm verifies both no result change
and the expected cache-miss behavior under a non-append-only history.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

config.MODEL = "claude-haiku-4-5-20251001"
config.HISTORY_WINDOW_SIZE = 20          # calibrated: ≈ 5 effective turns
config.CHANGE_DETECTION_ENABLED = True
config.PROMPT_CACHING_ENABLED = False    # keep variables clean
config.HISTORY_CACHING_ENABLED = True    # the variable under test: moving breakpoint on last message
config.RESULTS_DIR = "results_haiku45_budget_cal_cached_20260714"

os.makedirs(config.RESULTS_DIR, exist_ok=True)

from runner import run_experiment
from analyzer import analyze

print("=" * 60)
print("Claude 3.5 Haiku — CALIBRATED BUDGET cache-effect control rerun")
print(f"  Model:            {config.MODEL}")
print(f"  Sliding window:   ON (window={config.HISTORY_WINDOW_SIZE} ≈ 5 effective turns)")
print(f"  Change detection: {config.CHANGE_DETECTION_ENABLED}")
print(f"  Prompt caching:   {config.PROMPT_CACHING_ENABLED}")
print(f"  History caching:  {config.HISTORY_CACHING_ENABLED}")
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
