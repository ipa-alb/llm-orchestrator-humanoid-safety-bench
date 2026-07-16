#!/usr/bin/env python3
"""Aggregate the 2026-07-15 L1 5-repetition campaign into a results summary.

Reads results_l1_5rep_20260715/<model>_<arm>/rep<N>/run_v2_*.jsonl (must be
exactly 100 lines each), re-scores every rep with the unchanged analyzer,
and writes results_l1_5rep_20260715/RESULTS_L1_5REP.md plus a machine-
readable summary.json. Refuses to aggregate a rep whose log is not exactly
100 turns (guards against the append-contamination bug class).
"""

import sys, os, json, glob, statistics
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analyzer import load_log, check_violations, check_behavioral_issues

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "results_l1_5rep_20260715")
ARMS = [("haiku45", "claude-haiku-4-5-20251001"), ("chatgpt", "gpt-4o-mini"),
        ("gemini", "gemini-2.5-flash"), ("qwen8b", "qwen3:8b")]
CONDS = ["nobudget", "budget_cal"]

summary = {"campaign": "l1_5rep_20260715", "layer": "L1", "arms": {}}
lines = ["# L1 5-repetition campaign — 2026-07-15",
         "",
         "4 models x 2 conditions x 5 reps x 100 turns, text-only L1 harness",
         "(mock tools), unchanged analyzer. Cloud configs identical to the",
         "published corrective runs (windows: claude 20 / gpt 32 / gemini 50",
         "msgs, change detection on in budget_cal, off in nobudget); qwen3:8b",
         "is local via Ollama, thinking off, window 20 (see QWEN8B_NOTE.md).",
         "One fresh dir per rep; every log verified at exactly 100 turns.", ""]

for mk, model in ARMS:
    for cond in CONDS:
        reps = []
        for repdir in sorted(glob.glob(f"{ROOT}/{mk}_{cond}/rep*")):
            logs = glob.glob(f"{repdir}/*.jsonl")
            assert len(logs) == 1, f"expected exactly 1 jsonl in {repdir}: {logs}"
            entries = load_log(logs[0])
            if len(entries) != 100 and os.environ.get("L1AGG_ALLOW_PARTIAL"):
                print(f"  [partial preview] skipping {logs[0]} ({len(entries)} turns)")
                continue
            assert len(entries) == 100, f"{logs[0]} has {len(entries)} turns"
            viols, issues, rules = [], [], Counter()
            for e in entries:
                for v in check_violations(e):
                    v["turn"] = e["turn"]; viols.append(v); rules[v["rule"]] += 1
                issues.extend(check_behavioral_issues(e))
            reps.append({"rep": int(os.path.basename(repdir)[3:]), "violations": len(viols),
                         "rules": dict(rules), "issues": len(issues),
                         "detail": [{"turn": v["turn"], "rule": v["rule"],
                                     "desc": v["desc"]} for v in viols]})
        if not reps:
            continue
        vs = [r["violations"] for r in reps]
        iss = [r["issues"] for r in reps]
        rules_total = Counter()
        for r in reps:
            rules_total.update(r["rules"])
        arm = {"model": model, "condition": cond, "n": len(reps),
               "violations_per_rep": vs, "violations_total": sum(vs),
               "violations_mean": statistics.mean(vs),
               "violations_stdev": statistics.stdev(vs) if len(vs) > 1 else 0.0,
               "rule_counts": dict(rules_total),
               "issues_per_rep": iss, "issues_total": sum(iss), "reps": reps}
        summary["arms"][f"{mk}_{cond}"] = arm
        rules_s = ", ".join(f"{k}={v}" for k, v in sorted(rules_total.items())) or "—"
        lines.append(f"## {model} — {cond} (n={len(reps)})")
        lines.append("")
        lines.append(f"- violations per rep: {vs}  (total **{sum(vs)}/500**, "
                     f"mean {statistics.mean(vs):.1f} ± "
                     f"{(statistics.stdev(vs) if len(vs)>1 else 0):.1f} sd)")
        lines.append(f"- by rule: {rules_s}")
        lines.append(f"- behavioral issues per rep: {iss} (total {sum(iss)})")
        lines.append("")

with open(f"{ROOT}/summary.json", "w") as fh:
    json.dump(summary, fh, indent=1)
with open(f"{ROOT}/RESULTS_L1_5REP.md", "w") as fh:
    fh.write("\n".join(lines) + "\n")
print(f"wrote {ROOT}/RESULTS_L1_5REP.md")
for k, a in summary["arms"].items():
    print(f"{k:22s} n={a['n']} viol={a['violations_per_rep']} total={a['violations_total']}/500")
