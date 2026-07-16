#!/usr/bin/env python3
"""Run Gemini 2.5 Flash with a CALIBRATED budget window.

Window size 50 is chosen to match ~5 effective turns of context, equivalent
to what the Anthropic budget runs see at window=20. Rationale:
  - Gemini generates ~10 messages per turn (462 API calls / 100 turns = 4.62
    calls/turn; each tool call is a separate API round-trip with 2 messages)
  - 10 msg/turn × 5 turns = 50
  - Anthropic runner: ~4 msg/turn × 5 turns = 20 (the existing budget setting)

The original Gemini budget run used window=20, which gave only ~2 effective
turns of context — the most aggressive truncation of any model in the study.
This run tests whether Gemini's regression (45→61 issues, 4 flat refusals)
disappears when given a fair context window.
Expected cost: ~$0.25.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

gemini_key = os.environ.get("GEMINI_API_KEY")
if not gemini_key:
    raise RuntimeError("GEMINI_API_KEY is not set")
os.environ["OPENAI_API_KEY"] = gemini_key

import config

config.MODEL = "gemini-2.5-flash"
config.BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
config.HISTORY_WINDOW_SIZE = 50          # calibrated: ≈ 5 effective turns
config.CHANGE_DETECTION_ENABLED = True
config.RESULTS_DIR = "results_gemini_budget_cal"

os.makedirs(config.RESULTS_DIR, exist_ok=True)

from runner_openai import run_experiment
from analyzer import analyze

print("=" * 60)
print("Gemini 2.5 Flash — CALIBRATED BUDGET run")
print(f"  Model:            {config.MODEL}")
print(f"  Base URL:         {config.BASE_URL}")
print(f"  Sliding window:   ON (window={config.HISTORY_WINDOW_SIZE} ≈ 5 effective turns)")
print(f"  Change detection: {config.CHANGE_DETECTION_ENABLED}")
print(f"  Results dir:      {config.RESULTS_DIR}")
print("  NOTE: Window=50 because Gemini makes ~10 messages/turn.")
print("=" * 60)

log_path = run_experiment(
    safety_version="v2",
    num_turns=100,
    run_id="gemini_budget_cal",
    verbose=True,
)

print("\n--- Running analysis ---")
analyze(log_path)
