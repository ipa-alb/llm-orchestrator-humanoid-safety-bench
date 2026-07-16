#!/usr/bin/env bash
# One-command closed loop (R2): compose stack up -> canonical 100-turn run(s)
# against MuJoCoBackend -> UNCHANGED L1 analyzer -> summary.
#
#   scripts/run_closed_loop.sh                    # compliant + violator, 100 turns
#   scripts/run_closed_loop.sh --policy compliant # one policy only
#   scripts/run_closed_loop.sh --turns 10         # smoke subset
#   scripts/run_closed_loop.sh --dwell 1.0        # faster sim dwell per turn
#   scripts/run_closed_loop.sh --keep-up          # leave the stack running
#
# Requires: docker compose (images g1-base/g1-sim built, see docker/up.sh),
# host python3 with matplotlib for the analyzer step, and the
# degradation_test checkout as a sibling of humanoid_testing.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
COMPOSE=(docker compose -f "$ROOT/docker/compose.yaml")

POLICIES=(compliant violator)
TURNS=""
DWELL="2.0"
KEEP_UP=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy) POLICIES=("$2"); shift 2 ;;
    --turns) TURNS="$2"; shift 2 ;;
    --dwell) DWELL="$2"; shift 2 ;;
    --keep-up) KEEP_UP=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_HOST="$ROOT/bench/runs/closed_loop_$STAMP"
OUT_CTR="/ws/bench/runs/closed_loop_$STAMP"

cleanup() {
  if [[ $KEEP_UP -eq 0 ]]; then
    echo "[run_closed_loop] stopping stack ..."
    "${COMPOSE[@]}" down --remove-orphans >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

FAIL=0
declare -A JSONL
for POLICY in "${POLICIES[@]}"; do
  RUN_ID="cl_${STAMP}_${POLICY}"
  echo
  # Fresh physics per episode: a fallen/wandered robot from a previous run
  # must not leak into the next (worldctl reset does not reset physics).
  echo "[run_closed_loop] (re)starting sim + loco + battery ..."
  "${COMPOSE[@]}" down --remove-orphans >/dev/null 2>&1 || true
  "${COMPOSE[@]}" up -d sim loco battery
  echo "[run_closed_loop] === $POLICY run (turns: ${TURNS:-100}, dwell ${DWELL}s) ==="
  "${COMPOSE[@]}" run --rm bench \
      python3 scripts/closed_loop_driver.py \
        --policy "$POLICY" --dwell "$DWELL" --run-id "$RUN_ID" \
        --out-dir "$OUT_CTR" ${TURNS:+--turns "$TURNS"} \
      || FAIL=1
  JSONL[$POLICY]="$OUT_HOST/run_v2_${RUN_ID}.jsonl"
done

echo
echo "[run_closed_loop] === UNCHANGED L1 analyzer ==="
for POLICY in "${POLICIES[@]}"; do
  LOG="${JSONL[$POLICY]}"
  if [[ ! -f "$LOG" ]]; then
    echo "[run_closed_loop] MISSING LOG for $POLICY: $LOG" >&2; FAIL=1; continue
  fi
  EXPECT="--expect-violations"
  [[ "$POLICY" == "compliant" ]] && EXPECT="--expect-zero"
  python3 "$HERE/run_l1_analyzer.py" "$LOG" "$EXPECT" || FAIL=1
done

echo
if [[ $FAIL -eq 0 ]]; then
  echo "[run_closed_loop] PASS — artifacts in $OUT_HOST"
else
  echo "[run_closed_loop] FAIL — see messages above (artifacts in $OUT_HOST)" >&2
fi
exit $FAIL
