# Reproducing the paper's results

There are two tiers of reproduction, and it matters to keep them apart:

- **Tier A — exact.** Every published number is re-derived from the shipped
  raw logs by the unchanged scorer. This is bit-exact, needs no API keys, and
  is the strong verification path: our numbers follow from our data.
- **Tier B — statistical.** Re-running the campaigns against the live APIs
  reproduces the *findings* (means, confidence intervals, per-rule violation
  mixes), not identical transcripts: LLM sampling is stochastic and hosted
  models drift over time. Everything that can be pinned is pinned (fixed
  seed 7, deterministic scenario schedule, safety prompt v2, exact model
  IDs); what cannot be pinned is documented here.

Model IDs used for the paper (2026-07): `claude-haiku-4-5-20251001`,
`gpt-4o-mini`, `gemini-2.5-flash`, and local `qwen3:8b` / `qwen3:30b` via
Ollama.

---

## Tier A — re-derive every published number (offline)

Setup (Python ≥ 3.10):

```bash
pip install -r layer1/experiment/requirements.txt   # anthropic, openai, matplotlib
pip install python-dotenv    # used by the campaign launchers in NextTest/
```

### Layer 1

From `layer1/experiment/`:

```bash
# One accepted-paper run (violations, per-rule breakdown, behavioral issues):
python run_experiment.py --analyze-only results_haiku45_nobudget/run_v2_haiku45_nobudget.jsonl

# The full 5-repetition campaign aggregate (the camera-ready's Layer-1 numbers):
python NextTest/aggregate_l1_5rep.py
```

The scorer is `analyzer.py` — post-hoc, deterministic, and identical for
every model and layer. Nothing about scoring happens during a run.

### Layer 2

From `layer2/` (the Layer-1 analyzer is auto-discovered at
`../layer1/experiment` — i.e. this repo's own copy):

```bash
# One episode:
python3 scripts/run_l1_analyzer.py bench/runs/l2_campaign_20260714_171745/gpt_rep1/run_v2_gpt_rep1.jsonl

# All shipped campaigns, rescored from raw JSONL (also regenerates the
# paper's figures and figures_data.json in bench/analysis/):
python3 bench/analysis/make_figures.py
```

`make_figures.py` never trusts stored aggregates: it re-scores every episode
from the raw JSONL with the unchanged analyzer and records any disagreement
with the stored `analysis.json` files under `figures_data.json["discrepancies"]`.

### What maps to what

| paper artifact | data | command |
|---|---|---|
| Layer-1 violation counts (per model × condition) | `layer1/experiment/results_l1_5rep_20260715/` | `aggregate_l1_5rep.py` |
| Accepted-paper single runs (incl. latency traces) | `layer1/experiment/results_{haiku45,chatgpt,gemini}_*/` | `run_experiment.py --analyze-only` |
| Layer-2 replication (per model × condition) | `layer2/bench/runs/l2_campaign_*/` | `run_l1_analyzer.py` / `make_figures.py` |
| qwen3:30b refusal-shield result (0/500 no-budget, 62 % flat refusals) | `layer2/bench/runs/qwen30b_campaign_5rep/` | `run_l1_analyzer.py` per episode |
| Violations & behavioral-issues figure; budget-arrows figure | both of the above | `make_figures.py` |

Per-session API costs on the budget-arrows figure are provider-dashboard
estimates recorded at experiment time (token prices are not in the logs);
they are hardcoded with a provenance note inside `make_figures.py`.

---

## Tier B — re-run the campaigns

### Keys and environment

```bash
cp .env.example .env       # fill in your keys; .env is gitignored
```

`ANTHROPIC_API_KEY` (claude), `OPENAI_API_KEY` (GPT), `GEMINI_API_KEY`
(Gemini, via its OpenAI-compatible endpoint). Local models need an
[Ollama](https://ollama.com) server with the model pulled
(`ollama pull qwen3:8b`).

### Layer 1 (text only; no Docker needed)

From `layer1/experiment/`. The exact launchers used for the paper
are in `NextTest/`:

```bash
# The 5-repetition campaign (per model, per arm, 5 reps):
for r in 1 2 3 4 5; do python NextTest/run_l1_rep.py haiku45 nobudget   $r; done
for r in 1 2 3 4 5; do python NextTest/run_l1_rep.py haiku45 budget_cal $r; done
# model keys: haiku45 | chatgpt | gemini | qwen8b     arms: nobudget | budget_cal

# The accepted-paper single runs (one-off scripts, identical configs):
python NextTest/run_haiku45_nobudget.py     # etc. for chatgpt / gemini, budget_cal
```

Aggregate with `python NextTest/aggregate_l1_5rep.py`.

Approximate cost/time per 100-turn run (2026-07 prices): claude ≈ $1–2,
gpt-4o-mini ≈ $0.1, gemini-2.5-flash ≈ $0.2–0.5; 10–25 min per run.
A cheap pre-flight: add `--turns 10` via `run_experiment.py` directly.

### Layer 2 (MuJoCo, dockerized)

From `layer2/`. Requires Docker + docker compose and the
submodules checked out (`git submodule update --init`).

```bash
docker/up.sh                          # builds g1-base and g1-sim images
# (a "C++ unitree_mujoco build failed" warning during the g1-sim build is
#  expected and harmless: that is an optional alternative simulator — the
#  benchmark's simulator is the Python one run by the compose sim service)

# 1. Pre-flight (checks images, L1 checkout, keys):
scripts/run_campaign.sh --preflight

# 2. Deterministic self-test — must reproduce EXACTLY 0 and 70 violations
#    per 100 turns (no LLM involved; if this fails, the setup is broken):
scripts/run_campaign.sh --backends mock-compliant,mock-violator --reps 1 --parallel 2

# 3. Cheap smoke against a real backend:
scripts/run_campaign.sh --backends gpt --reps 1 --turns 10

# 4. The published campaigns (both arms):
scripts/run_campaign.sh --backends claude,gpt,gemini --reps 5 --parallel 2
scripts/run_campaign.sh --backends claude,gpt,gemini --reps 5 --parallel 2 --budget
scripts/run_campaign.sh --backends local --reps 5 --parallel 1           # qwen3:8b via Ollama
scripts/run_campaign.sh --backends local --reps 5 --parallel 1 --budget
# other local models: LOCAL_LLM_MODEL=qwen3:30b scripts/run_campaign.sh --backends local ...
```

Each campaign writes `bench/runs/l2_campaign_<timestamp>/` with one directory
per (backend, rep): raw `*.jsonl` + `*.meta.json` + `*.state.csv`, the
per-episode `analysis.json`, and a `campaign_summary.json`. Infrastructure
failures (sim not ready, physics divergence, API auth) are retried once with
a fresh stack; **model safety violations are data and are never retried**.

Published Layer-2 campaign wall-clock: ~30–60 min per (model, arm) at
`--parallel 10` for cloud models; local qwen3:8b ≈ 7 min/rep on a 24 GB GPU.

### What "successful reproduction" means

- The mock anchors must match exactly (0 and 70 violations / 100 turns).
- For cloud models, expect per-rep variance of the same order as the shipped
  per-rep tables (e.g. gpt-4o-mini Layer-2 no-budget ranged 0–21 violations
  per rep) — compare means and per-rule mixes over 5 reps, not single runs.
- Hosted models are moving targets: results for a model ID retired or
  retrained after 2026-07 may differ for reasons outside this benchmark.
  The shipped logs are the frozen record of what the paper measured.
