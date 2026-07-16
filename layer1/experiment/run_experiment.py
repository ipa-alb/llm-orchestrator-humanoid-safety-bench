#!/usr/bin/env python3
"""Entry point: pick safety doc version, run experiment, analyze results."""

import argparse
import sys

from runner import run_experiment
from analyzer import analyze, compare_runs


def main() -> None:
    parser = argparse.ArgumentParser(description="Safety Degradation PoC")
    parser.add_argument(
        "--safety-version",
        choices=["v1", "v2", "v3"],
        required=True,
        help="Safety document version: v1 (naive), v2 (structured), v3 (redundant)",
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=100,
        help="Number of turns to run (default: 100)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Custom run identifier (default: timestamp)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Override results directory (default: from config.py)",
    )
    parser.add_argument(
        "--analyze-only",
        type=str,
        default=None,
        help="Skip experiment, just analyze this log file",
    )
    parser.add_argument(
        "--compare",
        nargs="+",
        type=str,
        default=None,
        help="Compare multiple log files",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-turn output",
    )
    parser.add_argument(
        "--live-log",
        action="store_true",
        help="Enable live prompt/response logger (tail -f in another terminal)",
    )

    args = parser.parse_args()

    if args.compare:
        compare_runs(args.compare)
        return

    if args.analyze_only:
        analyze(args.analyze_only)
        return

    # Override results dir if requested
    if args.results_dir:
        import config
        config.RESULTS_DIR = args.results_dir

    # Start live logger if requested
    if args.live_log:
        from live_logger import install_live_logger
        install_live_logger()

    log_path = run_experiment(
        safety_version=args.safety_version,
        num_turns=args.turns,
        run_id=args.run_id,
        verbose=not args.quiet,
    )

    if args.live_log:
        from live_logger import uninstall_live_logger
        uninstall_live_logger()

    print("\n--- Running analysis ---")
    analyze(log_path)


if __name__ == "__main__":
    main()
