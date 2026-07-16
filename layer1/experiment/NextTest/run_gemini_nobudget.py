#!/usr/bin/env python3
"""Run Gemini 2.5 Flash with ALL budget optimizations disabled.

Controlled no-budget baseline. Provides a verified-config replacement for the
original run whose window/change-detection settings were not preserved.
Expected cost: ~$1.00.
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
config.HISTORY_WINDOW_SIZE = 99999       # no sliding window
config.CHANGE_DETECTION_ENABLED = False  # full env state injected every turn
config.RESULTS_DIR = "results_gemini_nobudget"

os.makedirs(config.RESULTS_DIR, exist_ok=True)

from runner_openai import run_experiment
from analyzer import analyze

print("=" * 60)
print("Gemini 2.5 Flash — CONTROLLED NO-BUDGET baseline")
print(f"  Model:            {config.MODEL}")
print(f"  Base URL:         {config.BASE_URL}")
print(f"  Sliding window:   OFF (window={config.HISTORY_WINDOW_SIZE})")
print(f"  Change detection: {config.CHANGE_DETECTION_ENABLED}")
print(f"  Results dir:      {config.RESULTS_DIR}")
print("  NOTE: Expected cost ~$1.00. Gemini makes ~4.6 API calls/turn.")
print("=" * 60)

log_path = run_experiment(
    safety_version="v2",
    num_turns=100,
    run_id="gemini_nobudget",
    verbose=True,
)

print("\n--- Running analysis ---")
analyze(log_path)
