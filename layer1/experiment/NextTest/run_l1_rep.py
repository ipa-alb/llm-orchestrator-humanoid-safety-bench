#!/usr/bin/env python3
"""Generic single-rep L1 runner for the 5-repetition campaign (2026-07-15).

Usage: run_l1_rep.py <model_key> <arm> <rep>
  model_key: haiku45 | chatgpt | gemini | qwen8b | qwen30b

qwen8b is the local-model arm actually run for the 2026-07-15 campaign
(qwen3:8b via Ollama — the model of the archived pre-fix L2 campaigns,
fully GPU-resident on a 24 GB card so latencies are clean). The qwen30b
entries are config-only (kept for a possible later run of the L2
post-fix campaign model qwen3:30b); no qwen30b results exist.
  arm:       nobudget | budget_cal
  rep:       1..5

Reproduces EXACTLY the configs of the corrective one-off scripts
(run_<model>_<arm>.py in this directory) but writes each repetition into
its own fresh directory:

    results_l1_5rep_20260715/<model_key>_<arm>/rep<N>/

One dir per rep because logger.py opens the jsonl in APPEND mode — the
root cause of the 153-line contamination found in the published
results_gemini_budget_cal log. A pre-existing jsonl in the target dir is
an error, never appended to.

Anthropic runs keep HISTORY_CACHING_ENABLED = True (config default): the
2026-07-14 cache-effect control (CACHE_CONTROL_FINDINGS.md) showed the
moving cache breakpoint is billing-only. PROMPT_CACHING_ENABLED stays
False exactly as in the one-off scripts.
"""

import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".env"))

model_key, arm, rep = sys.argv[1], sys.argv[2], int(sys.argv[3])

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"
CFG = {
    ("haiku45", "nobudget"):   dict(api="anthropic", model="claude-haiku-4-5-20251001",
                                    window=99999, chdet=False),
    ("haiku45", "budget_cal"): dict(api="anthropic", model="claude-haiku-4-5-20251001",
                                    window=20, chdet=True),
    ("chatgpt", "nobudget"):   dict(api="openai", model="gpt-4o-mini", base_url=None,
                                    window=99999, chdet=False),
    ("chatgpt", "budget_cal"): dict(api="openai", model="gpt-4o-mini", base_url=None,
                                    window=32, chdet=True),
    ("gemini", "nobudget"):    dict(api="openai", model="gemini-2.5-flash", base_url=GEMINI_BASE,
                                    window=99999, chdet=False, use_gemini_key=True),
    ("gemini", "budget_cal"):  dict(api="openai", model="gemini-2.5-flash", base_url=GEMINI_BASE,
                                    window=50, chdet=True, use_gemini_key=True),
    # Local Ollama models, thinking OFF via reasoning_effort="none" — the
    # exact mechanism the L2 bench driver uses for its local backend
    # (humanoid_testing llm_driver.py default_reasoning_effort="none",
    # auto-dropped on 400). Ollama ignores the API key but the OpenAI SDK
    # requires a non-empty one (use_ollama_key).
    # qwen8b budget_cal window 20 = 5 effective turns x 4 msgs/turn
    # measured in the 2026-07-15 calibration probe (1 tool call/turn in
    # baseline -> user + assistant(tool_calls) + tool + assistant = 4;
    # same figure as claude). See QWEN8B_NOTE.md.
    ("qwen8b", "nobudget"):    dict(api="openai", model="qwen3:8b",
                                    base_url="http://localhost:11434/v1",
                                    window=99999, chdet=False,
                                    use_ollama_key=True, reasoning_effort="none"),
    ("qwen8b", "budget_cal"):  dict(api="openai", model="qwen3:8b",
                                    base_url="http://localhost:11434/v1",
                                    window=20, chdet=True,
                                    use_ollama_key=True, reasoning_effort="none"),
    # qwen30b: config-only, not yet run (see docstring).
    ("qwen30b", "nobudget"):   dict(api="openai", model="qwen3:30b",
                                    base_url="http://localhost:11434/v1",
                                    window=99999, chdet=False,
                                    use_ollama_key=True, reasoning_effort="none"),
    ("qwen30b", "budget_cal"): dict(api="openai", model="qwen3:30b",
                                    base_url="http://localhost:11434/v1",
                                    window=30, chdet=True,
                                    use_ollama_key=True, reasoning_effort="none"),
}
cfg = CFG[(model_key, arm)]

if cfg.get("use_gemini_key"):
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    os.environ["OPENAI_API_KEY"] = key
if cfg.get("use_ollama_key"):
    os.environ["OPENAI_API_KEY"] = "ollama"

import config

config.MODEL = cfg["model"]
config.HISTORY_WINDOW_SIZE = cfg["window"]
config.CHANGE_DETECTION_ENABLED = cfg["chdet"]
config.PROMPT_CACHING_ENABLED = False
if "base_url" in cfg:
    config.BASE_URL = cfg["base_url"]
if "reasoning_effort" in cfg:
    config.REASONING_EFFORT = cfg["reasoning_effort"]

results_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results_l1_5rep_20260715", f"{model_key}_{arm}", f"rep{rep}")
config.RESULTS_DIR = results_dir

run_id = f"{model_key}_{arm}_rep{rep}"
jsonl_path = os.path.join(results_dir, f"run_v2_{run_id}.jsonl")
if os.path.exists(jsonl_path):
    raise RuntimeError(f"refusing to append to existing log: {jsonl_path}")
os.makedirs(results_dir, exist_ok=True)

if cfg["api"] == "anthropic":
    from runner import run_experiment
else:
    from runner_openai import run_experiment
from analyzer import analyze

print(f"[{run_id}] model={config.MODEL} window={config.HISTORY_WINDOW_SIZE} "
      f"chdet={config.CHANGE_DETECTION_ENABLED} -> {results_dir}", flush=True)

log_path = run_experiment(
    safety_version="v2",
    num_turns=100,
    run_id=run_id,
    verbose=False,
)

with open(log_path) as fh:
    n_lines = sum(1 for _ in fh)
if n_lines != 100:
    raise RuntimeError(f"[{run_id}] expected 100 turns, log has {n_lines} lines: {log_path}")

result = analyze(log_path, save_plots=False)
print(f"[{run_id}] DONE 100 turns, violations={result.get('total_violations')}", flush=True)
