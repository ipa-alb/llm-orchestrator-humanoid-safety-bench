#!/usr/bin/env python3
"""Run the UNCHANGED L1 analyzer (degradation_test/experiment/analyzer.py)
on a bench JSONL episode log.

The analyzer module is imported verbatim from the degradation_test checkout
(sibling of humanoid_testing, or G1_BENCH_L1_DIR) — nothing is copied or
patched. Needs matplotlib (analyzer imports it at module level), so this
normally runs on the HOST, not in the g1-base container.

Usage:
    python3 scripts/run_l1_analyzer.py <log.jsonl> [--expect-zero]
                                       [--expect-violations] [--plots]

Exit code 0 iff the stated expectation holds (or no expectation given).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "bench"))

from g1_safety_bench.scenario import l1_source  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("log", help="bench JSONL episode log")
    ap.add_argument("--expect-zero", action="store_true",
                    help="fail unless 0 violations and 0 behavioral issues")
    ap.add_argument("--expect-violations", action="store_true",
                    help="fail unless at least one violation is detected")
    ap.add_argument("--plots", action="store_true",
                    help="also save the analyzer's plots (into L1's "
                         "RESULTS_DIR relative to the cwd)")
    args = ap.parse_args(argv)

    analyzer = l1_source.import_l1("analyzer")
    if analyzer is None:
        print("ERROR: degradation_test/experiment not importable "
              "(set G1_BENCH_L1_DIR)", file=sys.stderr)
        return 2

    res = analyzer.analyze(args.log, save_plots=args.plots)

    summary = {
        "log": args.log,
        "total_turns": res["total_turns"],
        "total_violations": res["total_violations"],
        "rule_counts": res["rule_counts"],
        "phase_counts": res["phase_counts"],
        "behavioral_issues": len(res["behavioral_issues"]),
    }
    print("\n[run_l1_analyzer] " + json.dumps(summary))

    if args.expect_zero and (res["total_violations"] != 0
                             or res["behavioral_issues"]):
        print("[run_l1_analyzer] FAIL: expected a clean run", file=sys.stderr)
        return 1
    if args.expect_violations and res["total_violations"] == 0:
        print("[run_l1_analyzer] FAIL: expected violations, found none",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
