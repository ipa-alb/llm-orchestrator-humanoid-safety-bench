#!/usr/bin/env python3
"""Paper-figure generator for the CBS 2026 camera-ready / journal follow-up.

Rerunnable, idempotent: run from this directory (bench/analysis/) on the host
(needs python3 + matplotlib; no docker, no API keys). It only READS run
artifacts and only OVERWRITES its own outputs (fig_*.pdf / fig_*.png /
figures_data.json) in this directory.

What it does
------------
1. Auto-discovers Layer-2 campaigns under ../runs/ (any dir with a
   campaign_summary.json), skipping the explicit PRE-FIX skip-list below,
   anything under archive/, smoke runs (< 100 turns) and mock backends.
   Newly pulled campaigns (e.g. local-model runs from another machine)
   are picked up automatically on the next run.
2. Auto-discovers Layer-1 runs in the sibling degradation_test checkout:
   the results_l1_5rep_* campaign dirs (n=5 per arm, preferred) first, then
   any older corrective-series results_*_nobudget / results_*_budget_cal
   dirs — arms covered by a 5-rep campaign supersede the older singles of
   the same (model, condition). Legacy/confounded runs (InitialResults,
   window-20 "budget" family, partial runs, foreign L2 campaign copies)
   are cataloged but never pooled.
3. RE-SCORES every included run from its JSONL with the *unchanged* L1
   analyzer (imported live via g1_safety_bench.scenario.l1_source), never
   trusting stored analysis.json / campaign_summary.json aggregates.
   Aborted attempts are excluded two ways: `*.attempt1_aborted.jsonl`
   files are never read, and appended-retry contamination inside a JSONL
   (turn counter resets) is cut by scoring the FINAL contiguous attempt
   only. Any disagreement with stored aggregates is recorded in
   figures_data.json["discrepancies"].
4. Emits IEEEtran-friendly figures (PDF + PNG, ~3.5 in column width, no
   titles — captions live in LaTeX) plus figures_data.json with every
   number behind every mark.

Scope honesty: L2 data is simulation (MuJoCo G1); figures label it
"L2 (sim)". As of 2026-07-15 every pooled arm in both layers has n=5
reps, so 95% t-CIs are drawn everywhere; the n<3 no-interval policy
remains in force should a sparser arm ever appear. Nothing here refers
to Layer 3.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent            # bench/analysis
BENCH = HERE.parent                               # bench
RUNS_DIR = BENCH / "runs"
REPO_ROOT = BENCH.parents[1]                      # paper_review_cbs
L1_EXPERIMENT = Path(
    os.environ.get("G1_BENCH_L1_DIR", REPO_ROOT / "layer1" / "experiment")
)

# Import the unchanged L1 analyzer live (repo convention, see
# g1_safety_bench/scenario/l1_source.py).
sys.path.insert(0, str(BENCH))
from g1_safety_bench.scenario import l1_source  # noqa: E402

if l1_source.add_l1_to_path() is None:
    sys.exit("FATAL: degradation_test/experiment not found (set G1_BENCH_L1_DIR)")
import analyzer as l1_analyzer  # noqa: E402  (the unchanged L1 analyzer)

# ---------------------------------------------------------------------------
# PRE-FIX skip-list (L2). In every campaign below the robot fell at turn 9
# (far-reach move_hand overlay destabilized the whole-body policy; fixed in
# humanoid_testing commits 222591a/19554af/6c9c604 on 2026-07-14). They stay
# decision-layer valid but are NOT the current dataset — see
# bench/runs/RESULTS.md. Anything under runs/archive/ is also skipped.
# ---------------------------------------------------------------------------
PRE_FIX_SKIP = {
    "campaign_20260711_024626",
    "campaign_20260711_201032",
    "campaign_20260711_201508",
    "campaign_20260711_202206",
    "campaign_20260711_213135",
    "campaign_20260711_224244",
    "campaign_20260711_233203",
    "campaign_20260711_235108",
    "campaign_20260713_191932",
    "campaign_20260713_192445",
    "campaign_20260713_205050",
    "campaign_20260714_103632",
    "campaign_20260714_114542",
    "campaign_20260714_114814",
    "campaign_20260714_121447",
    "l2_campaign_20260714_125340",
}

REQUIRED_TURNS = 100          # smoke runs (< 100 turns) are skipped
ISSUE_TYPES = ["over_rejection", "flat_refusal", "sensor_neglect"]
RULES = ["S1", "S2", "S3", "S4", "S5"]

# Display names. The L2 artifacts pin exact model ids; L1 logs do not record
# the model, so the L1 mapping below is keyed on results-dir naming and backed
# by the preserved run scripts (NextTest/*.py) — see ANALYSIS.md catalog.
MODEL_DISPLAY = [
    (re.compile(r"claude-haiku-4-5"), "Claude Haiku 4.5"),
    (re.compile(r"claude-3-haiku"), "Claude 3 Haiku"),
    (re.compile(r"gpt-4o-mini"), "GPT-4o-mini"),
    (re.compile(r"gpt-4o$"), "GPT-4o"),
    (re.compile(r"gemini-2\.5-flash"), "Gemini 2.5 Flash"),
    (re.compile(r"qwen3:30b"), "Qwen3 30B"),
    (re.compile(r"qwen3:8b"), "Qwen3 8B"),
]
MODEL_ORDER = ["Claude Haiku 4.5", "Gemini 2.5 Flash", "GPT-4o-mini"]  # then others
CONDITIONS = ["no-budget", "budget"]

# L1 results-dir name -> (model id, budget family). Only the corrective-series
# families are pool-eligible; the legacy window-20 "budget" family and
# InitialResults are cataloged as excluded. New local-model runs pulled from
# the colleague's machine are picked up as long as they follow the
# results_<modeltoken>_{nobudget|budget_cal}[_...] convention.
L1_MODEL_TOKENS = [
    (re.compile(r"^qwen8b"), "qwen3:8b"),
    (re.compile(r"^haiku45"), "claude-haiku-4-5-20251001"),
    (re.compile(r"^haiku3"), "claude-3-haiku-20240307"),
    (re.compile(r"^chatgpt_4o"), "gpt-4o"),
    (re.compile(r"^chatgpt"), "gpt-4o-mini"),
    (re.compile(r"^gemini"), "gemini-2.5-flash"),
]

# Validated colorblind-safe categorical palette (dataviz skill reference
# palette, light mode; worst adjacent CVD dE 24.2). Fixed slot order.
PAL = {
    "blue": "#2a78d6",     # slot 1
    "aqua": "#1baf7a",     # slot 2
    "yellow": "#eda100",   # slot 3
    "green": "#008300",    # slot 4
    "violet": "#4a3aa7",   # slot 5
    "red": "#e34948",      # slot 6
}
RULE_COLORS = dict(zip(RULES, [PAL["blue"], PAL["aqua"], PAL["yellow"], PAL["green"], PAL["violet"]]))
TYPE_COLORS = dict(zip(ISSUE_TYPES, [PAL["blue"], PAL["aqua"], PAL["yellow"]]))
COND_COLORS = {"no-budget": PAL["blue"], "budget": PAL["yellow"]}
LAYER_COLORS = {"L1": PAL["blue"], "L2": PAL["aqua"]}
DOT_KW = dict(s=7, color="#333333", edgecolors="white", linewidths=0.4, zorder=5,
              clip_on=False)  # never silently clip a data dot at a fixed axis edge

# 95% two-sided t critical values (df = n-1); CI only drawn for n >= 3.
T95 = {2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}

plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.dpi": 200,
    "savefig.dpi": 300,
})

COL_W = 3.5  # IEEEtran column width in inches


# ---------------------------------------------------------------------------
# Scoring helpers (all numbers re-derived from JSONL via the L1 analyzer)
# ---------------------------------------------------------------------------
def load_jsonl(path: Path) -> list[dict]:
    entries = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def split_attempts(entries: list[dict]) -> list[list[dict]]:
    """Split appended retries: a new attempt starts when the turn counter
    resets (the episode/L1 loggers append; see DATA_NOTE.md in
    l2_campaign_20260714_180626)."""
    segs: list[list[dict]] = [[]]
    for e in entries:
        if segs[-1] and e["turn"] <= segs[-1][-1]["turn"]:
            segs.append([])
        segs[-1].append(e)
    return segs


def score(entries: list[dict]) -> dict:
    """Re-score one attempt with the unchanged L1 analyzer."""
    rules: Counter = Counter()
    types: Counter = Counter()
    n_v = n_i = 0
    for e in entries:
        for v in l1_analyzer.check_violations(e):
            rules[v["rule"]] += 1
            n_v += 1
        for i in l1_analyzer.check_behavioral_issues(e):
            types[i["type"]] += 1
            n_i += 1
    return {
        "turns": len(entries),
        "violations": n_v,
        "rule_counts": {r: rules.get(r, 0) for r in RULES if rules.get(r, 0)},
        "issues": n_i,
        "issue_type_counts": {t: types.get(t, 0) for t in ISSUE_TYPES},
        "safety_versions": sorted({str(e.get("safety_version")) for e in entries}),
    }


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def display_model(model_id: str) -> str:
    for rx, name in MODEL_DISPLAY:
        if rx.search(model_id or ""):
            return name
    return model_id or "unknown"


# ---------------------------------------------------------------------------
# Discovery — Layer 2
# ---------------------------------------------------------------------------
def discover_l2(discrepancies: list, catalog: list) -> list[dict]:
    reps: list[dict] = []
    for summary_path in sorted(RUNS_DIR.glob("*/campaign_summary.json")):
        camp_dir = summary_path.parent
        name = camp_dir.name
        if "archive" in camp_dir.parts:
            continue
        if name in PRE_FIX_SKIP:
            catalog.append({"layer": "L2", "path": str(camp_dir), "pooled": False,
                            "reason": "pre-fix physics (explicit skip-list; see RESULTS.md)"})
            continue
        try:
            summary = json.loads(summary_path.read_text())
        except Exception as exc:  # unreadable summary -> skip loudly
            discrepancies.append({"path": str(summary_path), "note": f"unreadable summary: {exc}"})
            continue
        cfg_turns = int(summary.get("config", {}).get("turns") or 0)
        if cfg_turns < REQUIRED_TURNS:
            catalog.append({"layer": "L2", "path": str(camp_dir), "pooled": False,
                            "reason": f"smoke campaign ({cfg_turns} turns < {REQUIRED_TURNS})"})
            continue

        for rep_dir in sorted(p for p in camp_dir.iterdir() if p.is_dir()):
            m = re.match(r"(?P<backend>.+)_rep(?P<rep>\d+)$", rep_dir.name)
            if not m:
                continue
            backend = m.group("backend")
            if backend.startswith("mock"):
                catalog.append({"layer": "L2", "path": str(rep_dir), "pooled": False,
                                "reason": "mock backend (reference policy, not an LLM)"})
                continue
            jsonls = [p for p in rep_dir.glob("*.jsonl")
                      if not p.name.endswith(".attempt1_aborted.jsonl")]
            if len(jsonls) != 1:
                discrepancies.append({"path": str(rep_dir),
                                      "note": f"expected exactly 1 main jsonl, found {len(jsonls)}"})
                continue
            jsonl = jsonls[0]
            meta = {}
            mp = jsonl.parent / (jsonl.stem + ".meta.json")
            if mp.is_file():
                meta = json.loads(mp.read_text())
            model_id = meta.get("llm_model") or summary.get("per_backend", {}).get(
                backend, {}).get("model", backend)
            budget = meta.get("budget")
            if budget is None:
                budget = bool(summary.get("config", {}).get("budget"))
            condition = "budget" if budget else "no-budget"

            attempts = split_attempts(load_jsonl(jsonl))
            final = attempts[-1]
            if len(attempts) > 1:
                discrepancies.append({
                    "path": str(jsonl),
                    "note": (f"{len(attempts)} appended attempts in main jsonl; scored final "
                             f"attempt only ({len(final)} turns; "
                             f"{sum(len(a) for a in attempts[:-1])} contaminating lines excluded)")})
            sc = score(final)
            if sc["turns"] != REQUIRED_TURNS:
                catalog.append({"layer": "L2", "path": str(jsonl), "pooled": False,
                                "reason": f"partial run ({sc['turns']} turns)"})
                continue

            # Cross-check the stored per-rep analysis.json, if any.
            ana_path = rep_dir / "analysis.json"
            if ana_path.is_file():
                try:
                    stored = json.loads(ana_path.read_text())
                    if (stored.get("total_violations") != sc["violations"]
                            or stored.get("total_turns") != sc["turns"]):
                        discrepancies.append({
                            "path": str(ana_path),
                            "note": (f"stored analysis.json disagrees (stored "
                                     f"{stored.get('total_violations')} viol/"
                                     f"{stored.get('total_turns')} turns vs recomputed "
                                     f"{sc['violations']}/{sc['turns']}); recomputation used")})
                except Exception as exc:
                    discrepancies.append({"path": str(ana_path), "note": f"unreadable: {exc}"})

            rec = {
                "layer": "L2",
                "model_id": model_id,
                "model": display_model(model_id),
                "condition": condition,
                "campaign": name,
                "rep": int(m.group("rep")),
                "path": str(jsonl.relative_to(RUNS_DIR)),
                **sc,
            }
            reps.append(rec)
            catalog.append({"layer": "L2", "path": str(jsonl), "pooled": True,
                            "model": model_id, "condition": condition,
                            "reason": "post-fix 100-turn rep"})
    return reps


# ---------------------------------------------------------------------------
# Discovery — Layer 1
# ---------------------------------------------------------------------------
def l1_family(dirname: str) -> tuple[str | None, str | None, str | None]:
    """Return (model_id, condition, exclusion_reason) from a results dir name."""
    m = re.match(r"^results_(?P<rest>.+)$", dirname)
    if not m:
        return None, None, "not a results_* dir"
    rest = m.group("rest")
    if re.search(r"budget_cal", rest):
        condition = "budget"
        token = re.sub(r"_budget_cal.*$", "", rest)
    elif re.search(r"nobudget", rest):
        condition = "no-budget"
        token = re.sub(r"_nobudget.*$", "", rest)
    elif re.search(r"budget", rest):
        return None, None, ("legacy window-20 budget family (InitialTest config), "
                            "not the published calibrated-budget condition")
    else:
        return None, None, "unrecognized condition family (no nobudget/budget_cal token)"
    model_id = None
    for rx, mid in L1_MODEL_TOKENS:
        if rx.search(token):
            model_id = mid
            break
    if model_id is None:
        model_id = token  # future local models: display the raw token
    return model_id, condition, None


def discover_l1(discrepancies: list, catalog: list) -> list[dict]:
    reps: list[dict] = []
    seen_hashes: dict[str, str] = {}
    if not L1_EXPERIMENT.is_dir():
        return reps

    # 5-rep campaign dirs (results_l1_5rep_<date>/<model>_<cond>/rep<N>/*.jsonl)
    # are the preferred replication set; arms found here supersede the older
    # corrective single runs of the same (model, condition).
    campaign_arms: set[tuple[str, str]] = set()
    for camp in sorted(L1_EXPERIMENT.glob("results_l1_5rep_*")):
        if not camp.is_dir():
            continue
        for arm_dir in sorted(p for p in camp.iterdir() if p.is_dir()):
            # Foreign L2 campaign copies parked inside the L1 campaign dir
            # (e.g. qwen30b_campaign_5rep*): L2 layout, "layer": "L2" meta,
            # byte-identical to bench/runs/<name>. Never an L1 arm.
            if (arm_dir / "campaign_summary.json").is_file():
                twin = RUNS_DIR / arm_dir.name
                catalog.append({
                    "layer": "L1", "path": str(arm_dir), "pooled": False,
                    "reason": ("foreign Layer-2 campaign copy (L2 layout/meta), "
                               "not an L1 arm"
                               + (f"; scored from its original at bench/runs/{arm_dir.name}"
                                  if twin.is_dir() else ""))})
                continue
            model_id, condition, reason = l1_family(f"results_{arm_dir.name}")
            if reason is not None:
                catalog.append({"layer": "L1", "path": str(arm_dir), "pooled": False,
                                "reason": reason})
                continue
            for jsonl in sorted(arm_dir.glob("rep*/*.jsonl")):
                h = md5(jsonl)
                if h in seen_hashes:
                    catalog.append({"layer": "L1", "path": str(jsonl), "pooled": False,
                                    "reason": f"byte-identical duplicate of {seen_hashes[h]}"})
                    continue
                seen_hashes[h] = str(jsonl)
                attempts = split_attempts(load_jsonl(jsonl))
                sc = score(attempts[-1])
                if len(attempts) > 1 or sc["turns"] != REQUIRED_TURNS:
                    catalog.append({"layer": "L1", "path": str(jsonl), "pooled": False,
                                    "reason": f"non-clean campaign log "
                                              f"({len(attempts)} attempts, {sc['turns']} turns)"})
                    continue
                campaign_arms.add((model_id, condition))
                reps.append({
                    "layer": "L1", "model_id": model_id,
                    "model": display_model(model_id), "condition": condition,
                    "campaign": camp.name, "rep": int(jsonl.parent.name[3:]),
                    "path": str(jsonl.relative_to(L1_EXPERIMENT)),
                    "model_source": "campaign dir layout (<model>_<cond>/rep<N>)",
                    **sc,
                })
                catalog.append({"layer": "L1", "path": str(jsonl), "pooled": True,
                                "model": model_id, "condition": condition,
                                "reason": "L1 5-rep campaign (2026-07-15)"})

    # Pool-eligible corrective-series dirs live at experiment/ top level.
    for d in sorted(p for p in L1_EXPERIMENT.iterdir() if p.is_dir()
                    and p.name.startswith("results")
                    and not p.name.startswith("results_l1_5rep_")):
        model_id, condition, reason = l1_family(d.name)
        if reason is None and (model_id, condition) in campaign_arms:
            catalog.append({"layer": "L1", "path": str(d), "pooled": False,
                            "reason": "superseded by the L1 5-rep campaign "
                                      "(same config, n=5 replication)"})
            # Keep the appended-attempt provenance visible even though the
            # dir is no longer pooled: published numbers derived from
            # whole-file scoring of these logs mixed attempts (e.g. the
            # gemini budget "33 issues" scored a dead 53-turn attempt plus
            # the final 100-turn attempt in a 153-line file).
            for jsonl in sorted(d.glob("*.jsonl")):
                attempts = split_attempts(load_jsonl(jsonl))
                if len(attempts) > 1:
                    discrepancies.append({
                        "path": str(jsonl),
                        "note": (f"{len(attempts)} appended attempts "
                                 f"({sum(len(a) for a in attempts[:-1])} lines from dead "
                                 f"attempts + final {len(attempts[-1])}-turn attempt). "
                                 "SUPERSEDED by the L1 5-rep campaign and NOT pooled — "
                                 "recorded for provenance only: any published number "
                                 "scored from this whole file mixed attempts.")})
            continue
        jsonls = sorted(d.glob("*.jsonl"))
        if reason is not None:
            catalog.append({"layer": "L1", "path": str(d), "pooled": False, "reason": reason})
            continue
        for jsonl in jsonls:
            h = md5(jsonl)
            if h in seen_hashes:
                catalog.append({"layer": "L1", "path": str(jsonl), "pooled": False,
                                "reason": f"byte-identical duplicate of {seen_hashes[h]}"})
                continue
            seen_hashes[h] = str(jsonl)
            attempts = split_attempts(load_jsonl(jsonl))
            final = attempts[-1]
            if len(attempts) > 1:
                discrepancies.append({
                    "path": str(jsonl),
                    "note": (f"{len(attempts)} appended attempts (L1 logger appends on rerun); "
                             f"scored final attempt only ({len(final)} turns; "
                             f"{sum(len(a) for a in attempts[:-1])} contaminating lines excluded). "
                             "Any earlier analysis of the whole file mixed attempts.")})
            sc = score(final)
            if sc["turns"] != REQUIRED_TURNS:
                catalog.append({"layer": "L1", "path": str(jsonl), "pooled": False,
                                "reason": f"partial run ({sc['turns']} turns) — never pooled"})
                continue
            rec = {
                "layer": "L1",
                "model_id": model_id,
                "model": display_model(model_id),
                "condition": condition,
                "campaign": d.name,
                "rep": len([r for r in reps if r["model_id"] == model_id
                            and r["condition"] == condition]) + 1,
                "path": str(jsonl.relative_to(L1_EXPERIMENT)),
                "model_source": ("results-dir name + preserved run script "
                                 "(L1 logs do not record the model id)"),
                **sc,
            }
            reps.append(rec)
            catalog.append({"layer": "L1", "path": str(jsonl), "pooled": True,
                            "model": model_id, "condition": condition,
                            "reason": "corrective-series config (cached reruns are "
                                      "billing-only config controls and pool as reps)"})

    # Catalog (never pool) the InitialResults series — methodologically
    # confounded, see comparison_normal_vs_budget/METHODOLOGICAL_REPORT.md.
    init = L1_EXPERIMENT / "InitialResults"
    if init.is_dir():
        for jsonl in sorted(init.rglob("*.jsonl")):
            h = md5(jsonl)
            reason = "InitialResults series (confounded; see METHODOLOGICAL_REPORT.md)"
            if h in seen_hashes:
                reason = f"byte-identical duplicate of {seen_hashes[h]}"
            else:
                seen_hashes[h] = str(jsonl)
            catalog.append({"layer": "L1", "path": str(jsonl), "pooled": False,
                            "reason": reason})
    return reps


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def arm_key(rec: dict) -> tuple[str, str]:
    return (rec["model"], rec["condition"])


def arms_from(reps: list[dict]) -> dict[tuple[str, str], list[dict]]:
    arms: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in reps:
        arms[arm_key(r)].append(r)
    return dict(arms)


# Models excluded from ALL figures (still discovered, scored, and present in
# figures_data.json's catalog — just not plotted). Author decision 2026-07-15.
EXCLUDED_FROM_FIGURES = {"Qwen3 30B"}


def ordered_models(arms: dict) -> list[str]:
    models = {m for m, _ in arms} - EXCLUDED_FROM_FIGURES
    return [m for m in MODEL_ORDER if m in models] + sorted(models - set(MODEL_ORDER))


def mean_ci(values: list[float]) -> dict:
    n = len(values)
    mean = float(np.mean(values)) if n else float("nan")
    out = {"n": n, "mean": mean, "values": values, "sd": None, "ci95_half": None}
    if n >= 2:
        out["sd"] = float(np.std(values, ddof=1))
    if n >= 3:
        out["ci95_half"] = T95.get(n, 1.96) * out["sd"] / math.sqrt(n)
    return out


def jitter(n: int, width: float = 0.10) -> np.ndarray:
    rng = np.random.default_rng(7)  # deterministic
    return rng.uniform(-width, width, size=n)


def draw_ci(ax, x, st, elinewidth=0.9, capsize=2.0):
    """95% t-CI whisker (only defined for n >= 3), lower arm clipped at 0
    because violation/issue counts cannot be negative (noted in
    figures_data.json)."""
    if st["ci95_half"] is None:
        return
    lo = min(st["mean"], st["ci95_half"])  # clip at zero
    ax.errorbar(x, st["mean"], yerr=[[lo], [st["ci95_half"]]], fmt="none",
                ecolor="#222222", elinewidth=elinewidth, capsize=capsize,
                capthick=elinewidth, zorder=4)


def draw_bar_with_reps(ax, x, values: list[float], color: str, width=0.62):
    """Mean bar + 95% CI whisker (n>=3 only) + jittered per-rep dots."""
    st = mean_ci(values)
    ax.bar(x, st["mean"], width=width, color=color, zorder=2)
    draw_ci(ax, x, st)
    ax.scatter(x + jitter(len(values)), values, **DOT_KW)
    return st


def finish_counts_axis(ax, headroom=1.18):
    """Counts cannot be negative; give annotations/dots headroom above."""
    ax.set_ylim(bottom=0)
    ax.set_ylim(0, max(ax.get_ylim()[1], 1.0) * headroom)
    ax.grid(axis="x", visible=False)


def stat_top(st: dict) -> float:
    """Highest y this stat reaches on the canvas (dots and CI cap)."""
    top = max(st["values"]) if st["values"] else 0.0
    if st["ci95_half"]:
        top = max(top, st["mean"] + st["ci95_half"])
    return float(top)


def fixed_counts_axis(ax, top: float, needed: float, name: str):
    """Author-mandated fixed per-session y-scale (2026-07-15): a 100-turn
    session has a natural per-session ceiling of 100 events, so per-session
    panels share a fixed 0-100 scale (0-105 where ~99-value dots need
    headroom off the frame edge); tick labels always stop at 100. Never
    silently clamps: if the data outgrows the fixed range it reports loudly
    (and the overshoot stays visible because rep dots draw with
    clip_on=False)."""
    if needed > top:
        print(f"  WARNING: {name}: data reaches {needed:.1f}, above the fixed "
              f"0-{top:g} scale — NOT clamped; raise with the authors")
    elif top - needed < 1:
        print(f"  NOTE: {name}: data reaches {needed:.1f}, touching the fixed "
              f"0-{top:g} frame edge (dot drawn unclipped)")
    ax.set_ylim(0, top)
    ax.set_yticks(range(0, 101, 20))
    ax.grid(axis="x", visible=False)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def save(fig, stem: str):
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"{stem}.{ext}", bbox_inches="tight",
                    facecolor="white", pad_inches=0.02)
    plt.close(fig)
    print(f"  wrote {stem}.pdf / .png")


def arm_positions(models: list[str]):
    """x positions: conditions adjacent within a model group.

    Returns (xs, labels, groups, cond_sep). With 4+ model groups in a
    3.5 in column ("dense", e.g. the L2 figures since qwen3:30b landed)
    the within-group spacing is widened relative to the group gap so the
    full/budget tick labels stop colliding; group labels are then wrapped
    to two lines by draw_arm_xlabels.
    """
    dense = len(models) >= 4
    cond_sep = 1.2 if dense else 0.8
    group_gap = 0.4 if dense else 0.55
    xs, labels, groups = [], [], []
    x = 0.0
    for m in models:
        for c in CONDITIONS:
            xs.append(x)
            labels.append("full" if c == "no-budget" else "budget")
            x += cond_sep
        groups.append((m, (xs[-2] + xs[-1]) / 2))
        x += group_gap
    return xs, labels, groups, cond_sep


def wrap_model(name: str) -> str:
    """Break a long model name at the space nearest its middle."""
    if len(name) <= 11 or " " not in name:
        return name
    parts = name.split(" ")
    best = min(range(1, len(parts)),
               key=lambda i: abs(len(" ".join(parts[:i])) - len(" ".join(parts[i:]))))
    return " ".join(parts[:best]) + "\n" + " ".join(parts[best:])


def draw_arm_xlabels(ax, xs, labels, groups, y_group: float):
    """Condition ticks + centered model-group labels below them."""
    dense = len(groups) >= 4
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=7.0 if dense else 7.5)
    for m, gx in groups:
        ax.text(gx, y_group, wrap_model(m) if dense else m, ha="center",
                va="top", fontsize=7.5, transform=ax.get_xaxis_transform())


def fig_l2_violations(arms, data):
    """(a) L2 violations per 100-turn session: mean over reps, 95% CI, rep dots."""
    models = ordered_models(arms)
    xs, labels, groups, cond_sep = arm_positions(models)
    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    entry = {}
    i = 0
    for m in models:
        for c in CONDITIONS:
            reps = arms.get((m, c), [])
            vals = [r["violations"] for r in reps]
            if vals:
                st = draw_bar_with_reps(ax, xs[i], vals, COND_COLORS[c],
                                        width=0.78 * cond_sep)
                entry[f"{m} | {c}"] = st
            i += 1
    draw_arm_xlabels(ax, xs, labels, groups, y_group=-0.24)
    ax.set_ylabel("Violations / session")
    fixed_counts_axis(ax, 100, max((stat_top(s) for s in entry.values()), default=0),
                      "fig_l2_violations")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COND_COLORS[c]) for c in CONDITIONS]
    ax.legend(handles, ["full context", "context budget"], frameon=False,
              loc="upper right", handlelength=1.0, handleheight=0.8)
    ax.text(0.01, 0.98, "L2 (sim), n = 5 reps/arm", transform=ax.transAxes,
            ha="left", va="top", fontsize=7, color="#555555")
    save(fig, "fig_l2_violations")
    data["fig_l2_violations"] = {
        "description": "Mean violations per 100-turn session per arm; 95% t-CI over "
                       "5 reps; dots = individual reps. Layer 2 (MuJoCo sim). "
                       "Fixed 0-100 per-session scale (author convention).",
        "arms": entry,
    }


def fig_l2_violations_by_rule(arms, data):
    """(b) L2 violations by safety rule, stacked per arm (totals over 5x100 turns)."""
    models = ordered_models(arms)
    xs, labels, groups, cond_sep = arm_positions(models)
    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    entry = {}
    i = 0
    for m in models:
        for c in CONDITIONS:
            reps = arms.get((m, c), [])
            totals = Counter()
            for r in reps:
                totals.update(r["rule_counts"])
            bottom = 0
            for rule in RULES:
                v = totals.get(rule, 0)
                if v:
                    ax.bar(xs[i], v, width=0.78 * cond_sep, bottom=bottom,
                           color=RULE_COLORS[rule], edgecolor="white",
                           linewidth=0.6, zorder=2)
                    bottom += v
            if reps and bottom == 0:
                ax.text(xs[i], 0.4, "0", ha="center", va="bottom",
                        fontsize=7, color="#555555")
            if reps:
                entry[f"{m} | {c}"] = {"total_turns": sum(r["turns"] for r in reps),
                                       "rule_totals": {r: totals.get(r, 0) for r in RULES}}
            i += 1
    draw_arm_xlabels(ax, xs, labels, groups, y_group=-0.24)
    ax.set_ylabel("Violations by rule (5 sessions)")
    finish_counts_axis(ax, headroom=1.30)
    handles = [plt.Rectangle((0, 0), 1, 1, color=RULE_COLORS[r]) for r in RULES]
    ax.legend(handles, RULES, frameon=False, ncol=5, loc="upper right",
              handlelength=1.0, handleheight=0.8, columnspacing=0.8)
    ax.text(0.01, 0.98, "L2 (sim)", transform=ax.transAxes, ha="left", va="top",
            fontsize=7, color="#555555")
    save(fig, "fig_l2_violations_by_rule")
    data["fig_l2_violations_by_rule"] = {
        "description": "Total violations by safety rule per arm over the five "
                       "100-turn reps. Layer 2 (MuJoCo sim).",
        "arms": entry,
    }


def fig_l2_issues_by_type(arms, data):
    """(c) L2 behavioral issues by type: mean per rep, 95% CI, rep dots."""
    models = ordered_models(arms)
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    entry = {}
    xs_group, labels, groups, cond_sep = arm_positions(models)
    group_w = 0.325 * cond_sep
    i = 0
    for m in models:
        for c in CONDITIONS:
            reps = arms.get((m, c), [])
            if reps:
                for j, t in enumerate(ISSUE_TYPES):
                    vals = [r["issue_type_counts"].get(t, 0) for r in reps]
                    x = xs_group[i] + (j - 1) * group_w
                    st = mean_ci(vals)
                    ax.bar(x, st["mean"], width=group_w * 0.86,
                           color=TYPE_COLORS[t], zorder=2)
                    draw_ci(ax, x, st, elinewidth=0.8, capsize=1.5)
                    ax.scatter(x + jitter(len(vals), 0.05), vals,
                               **{**DOT_KW, "s": 5})
                    entry.setdefault(f"{m} | {c}", {})[t] = st
            i += 1
    draw_arm_xlabels(ax, xs_group, labels, groups, y_group=-0.22)
    ax.set_ylabel("Behavioral issues / session")
    fixed_counts_axis(ax, 105,
                      max((stat_top(s) for d in entry.values() for s in d.values()),
                          default=0), "fig_l2_issues_by_type")
    handles = [plt.Rectangle((0, 0), 1, 1, color=TYPE_COLORS[t]) for t in ISSUE_TYPES]
    ax.legend(handles, ["over-rejection", "flat refusal", "sensor neglect"],
              frameon=False, loc="upper right", handlelength=1.0, handleheight=0.8,
              bbox_to_anchor=(1.0, 1.14))  # clear of the ~86-issue qwen8b dots
    ax.text(0.01, 0.98, "L2 (sim), n = 5 reps/arm", transform=ax.transAxes,
            ha="left", va="top", fontsize=7, color="#555555")
    save(fig, "fig_l2_issues_by_type")
    data["fig_l2_issues_by_type"] = {
        "description": "Behavioral issues by type per 100-turn session: mean over "
                       "5 reps, 95% t-CI, dots = individual reps. Layer 2 (MuJoCo sim). "
                       "Fixed 0-105 per-session scale, tick labels to 100 "
                       "(author convention).",
        "arms": entry,
    }


def fig_l1_overview(arms, data):
    """L1 companion (honest upgrade of corrective_bars.png): violations (top)
    and behavioral-issue totals (bottom); mean bars, 95% t-CI (n >= 3; every
    current arm has n = 5), per-run dots. n is stated once in the corner
    note (author decision 2026-07-15: no per-bar n labels); both panels use
    the fixed 0-100 per-session scale."""
    models = ordered_models(arms)
    xs, labels, groups, cond_sep = arm_positions(models)
    fig, (ax_v, ax_i) = plt.subplots(
        2, 1, figsize=(COL_W, 3.2), sharex=True,
        gridspec_kw={"height_ratios": [1, 1.6], "hspace": 0.14})
    entry = {}
    i = 0
    for m in models:
        for c in CONDITIONS:
            reps = arms.get((m, c), [])
            if reps:
                v_vals = [r["violations"] for r in reps]
                i_vals = [r["issues"] for r in reps]
                st_v = draw_bar_with_reps(ax_v, xs[i], v_vals, COND_COLORS[c],
                                          width=0.78 * cond_sep)
                st_i = draw_bar_with_reps(ax_i, xs[i], i_vals, COND_COLORS[c],
                                          width=0.78 * cond_sep)
                entry[f"{m} | {c}"] = {"violations": st_v, "issues": st_i}
            i += 1
    ax_v.set_ylabel("Violations")
    ax_i.set_ylabel("Behavioral issues")
    draw_arm_xlabels(ax_i, xs, labels, groups, y_group=-0.24)
    fixed_counts_axis(ax_v, 100, max((stat_top(e["violations"]) for e in entry.values()),
                                     default=0), "fig_l1_overview/violations")
    fixed_counts_axis(ax_i, 100, max((stat_top(e["issues"]) for e in entry.values()),
                                     default=0), "fig_l1_overview/issues")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COND_COLORS[c]) for c in CONDITIONS]
    ax_v.legend(handles, ["full context", "context budget"], frameon=False,
                loc="upper right", handlelength=1.0, handleheight=0.8)
    ns = sorted({e["violations"]["n"] for e in entry.values()})
    n_note = f"n = {ns[0]}" if len(ns) == 1 else f"n = {ns[0]}–{ns[-1]}"
    ax_v.text(0.01, 0.96, f"L1 (text), per 100-turn session, {n_note}",
              transform=ax_v.transAxes, ha="left", va="top", fontsize=7,
              color="#555555")
    save(fig, "fig_l1_overview")
    data["fig_l1_overview"] = {
        "description": "Layer-1 arms (5-rep campaign 2026-07-15): violations (top) "
                       "and total behavioral issues (bottom) per 100-turn session. "
                       "Bars = mean over reps, 95% t-CI (n >= 3; all arms n = 5), "
                       "dots = individual runs; n stated once in the corner note. "
                       "Fixed 0-100 per-session scale on both panels "
                       "(author convention).",
        "arms": entry,
    }


def fig_l1_l2_violations(l1_arms, l2_arms, data):
    """(d) Cross-layer replication: violations (top) and behavioral-issue
    totals (bottom) per 100-turn session, L1 vs L2 — same two-panel layout
    as fig_l1_overview but paired by layer instead of condition.

    Includes every model with data in EITHER layer; a model that has only
    one layer so far (qwen3:30b: L2 done, L1 pending) shows a single bar
    per arm until its other layer lands."""
    models = ordered_models({**l2_arms, **l1_arms})
    # >3 model groups don't fit a single IEEE column; widen (use \figure* /
    # scale to \columnwidth in LaTeX as appropriate).
    fig_w = COL_W if len(models) <= 3 else COL_W + 0.75 * (len(models) - 3)
    fig, (ax_v, ax_i) = plt.subplots(
        2, 1, figsize=(fig_w, 2.05), sharex=True,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.14})
    entry = {}
    xs_ticks, tick_labels, groups, cond_sep = arm_positions(models)
    bw = 0.375 * cond_sep
    i = 0
    for m in models:
        for c in CONDITIONS:
            for k, (layer, arms) in enumerate((("L1", l1_arms), ("L2", l2_arms))):
                reps = arms.get((m, c), [])
                if not reps:
                    continue
                x = xs_ticks[i] + (k - 0.5) * bw
                stats_by_metric = {}
                for ax, metric in ((ax_v, "violations"), (ax_i, "issues")):
                    vals = [r[metric] for r in reps]
                    st = mean_ci(vals)
                    ax.bar(x, st["mean"], width=bw * 0.88,
                           color=LAYER_COLORS[layer], zorder=2)
                    draw_ci(ax, x, st, elinewidth=0.8, capsize=1.5)
                    ax.scatter(x + jitter(len(vals), 0.05), vals,
                               **{**DOT_KW, "s": 5})
                    stats_by_metric[metric] = st
                entry.setdefault(f"{m} | {c}", {})[layer] = stats_by_metric
            i += 1
    draw_arm_xlabels(ax_i, xs_ticks, tick_labels, groups, y_group=-0.30)
    ax_v.set_ylabel("Violations")
    ax_i.set_ylabel("Behavioral issues")
    for ax, metric in ((ax_v, "violations"), (ax_i, "issues")):
        needed = max((stat_top(by_m[metric]) for by_l in entry.values()
                      for by_m in by_l.values()), default=0)
        fixed_counts_axis(ax, 105, needed, f"fig_l1_l2_violations/{metric}")
    handles = [plt.Rectangle((0, 0), 1, 1, color=LAYER_COLORS[l]) for l in ("L1", "L2")]
    n_by_layer = {}
    for l, arms in (("L1", l1_arms), ("L2", l2_arms)):
        ns = {len(reps) for reps in arms.values() if reps}
        n_by_layer[l] = (f"n = {min(ns)}" if len(ns) == 1
                         else f"n = {min(ns)}–{max(ns)}") if ns else "n = 0"
    ax_v.legend(handles, [f"L1 (text), {n_by_layer['L1']}",
                          f"L2 (sim), {n_by_layer['L2']}"], frameon=False,
                loc="upper left", handlelength=1.0, handleheight=0.8)
    save(fig, "fig_l1_l2_violations")
    data["fig_l1_l2_violations"] = {
        "description": "Cross-layer replication per 100-turn session, Layer 1 "
                       "(text) vs Layer 2 (MuJoCo sim), per arm: violations (top) "
                       "and total behavioral issues (bottom). Bars = mean over "
                       "runs; 95% t-CI where n >= 3; dots = individual runs; "
                       "legend n computed from the discovered data. Fixed 0-105 "
                       "per-session scale on both panels, tick labels to 100 "
                       "(author convention).",
        "arms": entry,
    }


# ---------------------------------------------------------------------------
# Budget arrow chart (paper Fig. 4-right template, n=5 data)
# ---------------------------------------------------------------------------
# Per-session API cost (USD) on the arrow chart's x axis. PROVENANCE: the L1
# logger records NO token usage, and the 5-rep campaign kept no console
# logs, so costs CANNOT be recomputed from the campaign JSONLs. These are the
# corrective-series FINDINGS.md per-session estimates — the exact basis the
# original corrective_arrow_chart.png used — carried over because the 5-rep
# campaign is config-identical (same windows, scenario, 100 turns). Public
# prices as of 2026-07-11 (see PRICE_TABLE_AS_OF in scripts/campaign.py).
# Cross-check against the only real L1 token counts on disk (the
# config-identical claude cached-rerun console logs, caching is billing-only):
# 2.171M in / 12.4k out tokens no-budget and 487k / 21.4k budget, priced
# uncached at $1/$5 per Mtok = $2.23 and $0.59 per session — i.e. the ~$5
# no-budget estimate is conservative-high; treat the axis as
# order-of-magnitude. Local models (qwen3:8b) have no API cost and cannot
# sit on a log-cost axis; like the original (strictly 3 backends), they are
# omitted from this chart.
L1_COST_EST = {  # model display name -> {condition: USD/session}
    "Claude Haiku 4.5": {"no-budget": 5.00, "budget": 0.50},
    "Gemini 2.5 Flash": {"no-budget": 1.00, "budget": 0.10},
    "GPT-4o-mini": {"no-budget": 0.40, "budget": 0.10},
}
ARROW_IMPROVE = PAL["green"]   # fewer issues under budget
ARROW_WORSEN = PAL["red"]      # more issues under budget
MODEL_ACCENT = {"Claude Haiku 4.5": PAL["violet"], "GPT-4o-mini": PAL["aqua"],
                "Gemini 2.5 Flash": PAL["blue"]}

# Candidate label anchors relative to a marker, tried IN ORDER (deterministic
# flip on collision, author spec 2026-07-15): touch-distance sides first,
# then farther positions that get a thin leader line (offset > 12 pt).
LBL_CANDIDATES = [  # (dx_pt, dy_pt, ha, va, leader)
    (8, 0, "left", "center", False),
    (-8, 0, "right", "center", False),
    (0, 9, "center", "bottom", False),
    (0, -9, "center", "top", False),
    (18, 0, "left", "center", True),
    (-18, 0, "right", "center", True),
    (0, 18, "center", "bottom", True),
    (0, -18, "center", "top", True),
    (14, 12, "left", "bottom", True),
    (-14, 12, "right", "bottom", True),
    (14, -12, "left", "top", True),
    (-14, -12, "right", "top", True),
]


def _rects_overlap(a, b, pad=2.0):
    return not (a[2] + pad < b[0] or b[2] + pad < a[0]
                or a[3] + pad < b[1] or b[3] + pad < a[1])


def place_label(ax, renderer, occupied, xy_data, text, color, fontsize=6.0,
                bold=False, bbox_kw=None):
    """Anchor `text` next to the marker at `xy_data`: try LBL_CANDIDATES in
    order, keep the first position whose box stays inside the axes and
    overlaps nothing placed so far; draw a thin leader line for far anchors.
    Returns the chosen (dx_pt, dy_pt). Placement is fully derived from the
    marker's own coordinates — no per-model hand offsets."""
    fig = ax.figure
    px_per_pt = fig.dpi / 72.0
    probe = ax.annotate(text, xy_data, textcoords="offset points",
                        xytext=(0, 0), fontsize=fontsize,
                        fontweight="bold" if bold else "normal",
                        linespacing=1.15, ma="left", bbox=bbox_kw)
    ext = probe.get_window_extent(renderer)
    w, h = ext.width, ext.height
    probe.remove()
    mx, my = ax.transData.transform(xy_data)
    ax_box = ax.get_window_extent(renderer)
    chosen = LBL_CANDIDATES[0]
    for dx, dy, ha, va, leader in LBL_CANDIDATES:
        px, py = mx + dx * px_per_pt, my + dy * px_per_pt
        x0 = px - (w if ha == "right" else w / 2 if ha == "center" else 0)
        y0 = py - (h if va == "top" else h / 2 if va == "center" else 0)
        rect = (x0, y0, x0 + w, y0 + h)
        if not (ax_box.x0 + 2 <= rect[0] and rect[2] <= ax_box.x1 - 2
                and ax_box.y0 + 2 <= rect[1] and rect[3] <= ax_box.y1 - 2):
            continue
        if any(_rects_overlap(rect, o) for o in occupied):
            continue
        chosen = (dx, dy, ha, va, leader)
        break
    dx, dy, ha, va, leader = chosen
    arrowprops = (dict(arrowstyle="-", color=color, lw=0.5, alpha=0.7,
                       shrinkA=0, shrinkB=2) if leader else None)
    ax.annotate(text, xy_data, textcoords="offset points", xytext=(dx, dy),
                fontsize=fontsize, fontweight="bold" if bold else "normal",
                color=color, ha=ha, va=va, linespacing=1.15, ma="left",
                bbox=bbox_kw, zorder=7, arrowprops=arrowprops)
    px, py = mx + dx * px_per_pt, my + dy * px_per_pt
    x0 = px - (w if ha == "right" else w / 2 if ha == "center" else 0)
    y0 = py - (h if va == "top" else h / 2 if va == "center" else 0)
    occupied.append((x0, y0, x0 + w, y0 + h))
    return dx, dy


def fig_budget_arrows(l1_arms, data):
    """Fig. 4-right template (corrective_arrow_chart.png) rebuilt from the
    n=5 L1 campaign: per model, an arrow from the no-budget run (circle) to
    the budget run (diamond) in (per-session API cost [log x], mean
    behavioral issues/session [y]) space; arrow color = direction of the
    issue change (palette green improve / red worsen, double-encoded by the
    signed delta-issues label); marker size grows with mean violations.
    Cloud models only — see the L1_COST_EST provenance note."""
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches

    # Aspect matches the accepted paper's corrective_arrow_chart template
    # (1921x1029 ~ 0.53); extra width + right xlim headroom gives the labels
    # horizontal room so they sit beside their markers without leader lines.
    fig, ax = plt.subplots(figsize=(4.0, 2.13))
    # Axes geometry FIRST so transData is final before label placement.
    ax.set_xscale("log")
    ax.set_xlim(0.06, 14)
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))
    xticks = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"${v:g}" for v in xticks], fontsize=6.5)
    ax.minorticks_off()
    ax.set_xlabel("Est. API cost / session (USD, log)")
    ax.set_ylabel("Behavioral issues / session")
    ax.grid(which="major", alpha=0.2)

    # Phase 1: one loop builds every record AND draws its marks — labels are
    # placed in phase 2 from the very same records (binding by construction).
    entry = {}
    records = []
    worsened_any = False
    skipped = []
    for m in ordered_models(l1_arms):
        if m not in L1_COST_EST:
            skipped.append(m)
            continue
        color = MODEL_ACCENT.get(m, "#555555")
        pts = {}
        for c in CONDITIONS:
            reps = l1_arms.get((m, c), [])
            if not reps:
                continue
            st_i = mean_ci([r["issues"] for r in reps])
            st_v = mean_ci([r["violations"] for r in reps])
            pts[c] = {"cost": L1_COST_EST[m][c], "issues": st_i["mean"],
                      "violations": st_v["mean"], "issues_stat": st_i,
                      "violations_stat": st_v, "n": st_i["n"]}
        if len(pts) != 2:
            continue
        nb, b = pts["no-budget"], pts["budget"]
        improved = b["issues"] < nb["issues"]
        worsened_any = worsened_any or not improved
        ax.annotate("", xy=(b["cost"], b["issues"]),
                    xytext=(nb["cost"], nb["issues"]),
                    arrowprops=dict(arrowstyle="-|>",
                                    color=ARROW_IMPROVE if improved else ARROW_WORSEN,
                                    lw=1.3, mutation_scale=9,
                                    connectionstyle="arc3,rad=0.12"), zorder=4)
        s_nb = 25 + math.sqrt(nb["violations"]) * 25
        s_b = 20 + math.sqrt(b["violations"]) * 25
        ax.scatter(nb["cost"], nb["issues"], s=s_nb, c=color, alpha=0.9,
                   marker="o", edgecolors="#333333", linewidths=0.8, zorder=6)
        ax.scatter(b["cost"], b["issues"], s=s_b, c=color, alpha=0.65,
                   marker="D", edgecolors="#555555", linewidths=0.6, zorder=6)
        di = b["issues"] - nb["issues"]
        records.append({
            "name": m, "color": color,
            "start_xy": (nb["cost"], nb["issues"]),
            "end_xy": (b["cost"], b["issues"]),
            "start_size": s_nb, "end_size": s_b,
            "start_label": f"{m}\n({nb['violations']:g} viol.)",
            "end_label": f"(budget) {b['violations']:g} viol.",
            "delta_label": f"{'+' if di >= 0 else '−'}{abs(di):g} issues",
            "mid_xy": (math.exp((math.log(nb['cost']) + math.log(b['cost'])) / 2),
                       (nb["issues"] + b["issues"]) / 2),
        })
        entry[m] = {"no-budget": {k: v for k, v in nb.items()},
                    "budget": {k: v for k, v in b.items()},
                    "delta_issues": di, "arrow": "improved" if improved else "worsened"}

    handles = [
        mlines.Line2D([], [], color="#777777", marker="o", linestyle="None",
                      markersize=4.5, label="full context"),
        mlines.Line2D([], [], color="#777777", marker="D", linestyle="None",
                      markersize=4, alpha=0.7, label="context budget"),
        mpatches.Patch(color=ARROW_IMPROVE, label="fewer issues"),
    ]
    if worsened_any:
        handles.append(mpatches.Patch(color=ARROW_WORSEN, label="more issues"))
    legend = ax.legend(handles=handles, loc="upper right", frameon=False,
                       fontsize=6, handlelength=1.0, borderaxespad=0.2,
                       labelspacing=0.3)
    n_all = sorted({e["no-budget"]["n"] for e in entry.values()}
                   | {e["budget"]["n"] for e in entry.values()})
    n_note = f"n = {n_all[0]}" if len(n_all) == 1 else f"n = {n_all[0]}–{n_all[-1]}"
    corner = ax.text(0.01, 0.98, f"L1 (text), mean of {n_note} sessions",
                     transform=ax.transAxes, ha="left", va="top", fontsize=6,
                     color="#555555")
    # Local model: zero marginal API cost — a log dollar axis cannot honestly
    # place $0, so no arrow (author decision 2026-07-15); in-panel note only.
    qwen_note = None
    qwen_txt = None
    q_nb = l1_arms.get(("Qwen3 8B", "no-budget"), [])
    q_b = l1_arms.get(("Qwen3 8B", "budget"), [])
    if q_nb and q_b:
        qv_nb = mean_ci([r["violations"] for r in q_nb])
        qv_b = mean_ci([r["violations"] for r in q_b])
        qwen_note = (f"qwen3:8b (local, no API cost): "
                     f"{qv_nb['mean']:g} → {qv_b['mean']:g} viol.")
        qwen_txt = ax.text(0.98, 0.03, qwen_note, transform=ax.transAxes,
                           ha="right", va="bottom", fontsize=6.0,
                           color="#555555", ma="left")

    # Phase 2: programmatic label placement from the SAME records. Every
    # label anchors to its own marker's coordinates; candidates flip
    # deterministically on collision (see place_label).
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    px_per_pt = fig.dpi / 72.0
    occupied = [tuple(legend.get_window_extent(renderer).extents),
                tuple(corner.get_window_extent(renderer).extents)]
    if qwen_txt is not None:
        occupied.append(tuple(qwen_txt.get_window_extent(renderer).extents))
    for rec in records:  # marker keep-out boxes
        for xy, s in ((rec["start_xy"], rec["start_size"]),
                      (rec["end_xy"], rec["end_size"])):
            cx, cy = ax.transData.transform(xy)
            r = (math.sqrt(s) / 2 + 1.5) * px_per_pt
            occupied.append((cx - r, cy - r, cx + r, cy + r))
    placements = {}
    for rec in records:
        placements[rec["name"]] = {
            "start_label_offset_pt": place_label(
                ax, renderer, occupied, rec["start_xy"], rec["start_label"],
                rec["color"], bold=True),
            "end_label_offset_pt": place_label(
                ax, renderer, occupied, rec["end_xy"], rec["end_label"],
                rec["color"]),
        }
    chip_bbox = dict(boxstyle="round,pad=0.15", facecolor="white",
                     alpha=0.75, edgecolor="#cccccc", linewidth=0.4)
    for rec in records:  # delta chips after all endpoint labels
        placements[rec["name"]]["delta_chip_offset_pt"] = place_label(
            ax, renderer, occupied, rec["mid_xy"], rec["delta_label"],
            "#555555", fontsize=5.8, bbox_kw=chip_bbox)
    for name, p in placements.items():
        print(f"    arrow-chart labels {name}: start {p['start_label_offset_pt']} pt, "
              f"end {p['end_label_offset_pt']} pt, "
              f"delta {p['delta_chip_offset_pt']} pt")
    save(fig, "fig_budget_arrows")
    data["fig_budget_arrows"] = {
        "description": "Budget arrow chart (original corrective_arrow_chart.png "
                       "template, rebuilt from the n=5 L1 campaign): arrow = "
                       "full-context -> budget per model in (per-session API "
                       "cost [log], mean behavioral issues/session) space; "
                       "marker size grows with mean violations; arrow color = "
                       "sign of the issue change (double-encoded by the signed "
                       "label). Fixed 0-100 per-session y scale.",
        "cost_basis": "Corrective-series FINDINGS.md per-session estimates "
                      "(public prices as of 2026-07-11, PRICE_TABLE_AS_OF in "
                      "scripts/campaign.py); L1 logs record no token usage so "
                      "costs are NOT recomputed from the 5-rep JSONLs. "
                      "Cross-check from the config-identical claude cached-rerun "
                      "console logs: $2.23 no-budget / $0.59 budget per session "
                      "at $1/$5 per Mtok (vs the ~$5 / ~$0.50 estimates).",
        "excluded_from_arrows": [
            f"{m} (no per-session API-cost basis: local models have no API "
            "cost and cannot sit on a log-cost axis; the original template "
            "is strictly cloud/3-backend)" for m in skipped],
        "qwen_note": qwen_note,
        "label_placements_pt": placements,
        "arms": entry,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    discrepancies: list[dict] = []
    catalog: list[dict] = []
    print(f"L2 runs root: {RUNS_DIR}")
    print(f"L1 experiment root: {L1_EXPERIMENT}")

    l2_reps = discover_l2(discrepancies, catalog)
    l1_reps = discover_l1(discrepancies, catalog)
    l2_arms = arms_from(l2_reps)
    l1_arms = arms_from(l1_reps)

    print(f"discovered: {len(l2_reps)} L2 reps in {len(l2_arms)} arms; "
          f"{len(l1_reps)} L1 runs in {len(l1_arms)} arms")
    for (m, c), reps in sorted(l2_arms.items()):
        print(f"  L2 {m:>18s} {c:>9s}: n={len(reps)} viol={[r['violations'] for r in reps]}")
    for (m, c), reps in sorted(l1_arms.items()):
        print(f"  L1 {m:>18s} {c:>9s}: n={len(reps)} viol={[r['violations'] for r in reps]}")

    data: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator": "bench/analysis/make_figures.py",
        "scoring": "recomputed from JSONL with the unchanged L1 analyzer "
                   "(check_violations / check_behavioral_issues); final attempt only; "
                   "*.attempt1_aborted.jsonl excluded",
        "scope": "L2 = MuJoCo simulation (Layer 2); L1 = text-prompting (Layer 1). "
                 "No Layer 3 data exists.",
        "pre_fix_skip_list": sorted(PRE_FIX_SKIP),
        "error_bars": "95% two-sided t-CI over repetitions, drawn only where "
                      "n >= 3 (every pooled arm currently has n = 5); the lower "
                      "whisker is clipped at 0 (counts cannot be negative). "
                      "Should an n = 1..2 arm ever appear, no interval is drawn: "
                      "the individual-run dots are the only honest spread statement.",
    }

    print("rendering figures...")
    fig_l2_violations(l2_arms, data)
    fig_l2_violations_by_rule(l2_arms, data)
    fig_l2_issues_by_type(l2_arms, data)
    if l1_arms:
        fig_l1_overview(l1_arms, data)
        fig_l1_l2_violations(l1_arms, l2_arms, data)
        fig_budget_arrows(l1_arms, data)

    data["l2_reps"] = l2_reps
    data["l1_runs"] = l1_reps
    data["discrepancies"] = discrepancies
    data["discovery_catalog"] = catalog
    (HERE / "figures_data.json").write_text(json.dumps(data, indent=1) + "\n")
    print(f"wrote figures_data.json ({len(discrepancies)} discrepancies, "
          f"{len(catalog)} catalog entries)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
