"""Run all A2 tests: python3 bench/tests/a2_run_all.py"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TESTS = [
    "a2_test_schedule_parity.py",
    "a2_test_policies_analyzer.py",
    "a2_test_logger_schema.py",
]


def main() -> int:
    failed = []
    for t in TESTS:
        print(f"\n===== {t} =====")
        rc = subprocess.call([sys.executable, str(HERE / t)])
        if rc != 0:
            failed.append(t)
    print("\n===== summary =====")
    for t in TESTS:
        print(f"  {'FAIL' if t in failed else 'PASS'}  {t}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
