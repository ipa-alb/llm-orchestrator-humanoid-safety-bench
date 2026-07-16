#!/usr/bin/env bash
# Build the g1 images and start the compose stack.
#
#   ./docker/up.sh                 # build g1-base + g1-sim, start core services
#   ./docker/up.sh --sim-only      # start only the sim service
#   ./docker/up.sh --profiles gui  # additionally enable compose profiles (csv: gui,ros,gpu)
#   ./docker/up.sh --no-build      # skip image builds
#   ./docker/up.sh -- -d           # everything after -- is passed to `docker compose up`
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"

BUILD=1
SIM_ONLY=0
PROFILES=""
UP_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-build) BUILD=0; shift ;;
    --sim-only) SIM_ONLY=1; shift ;;
    --profiles) PROFILES="$2"; shift 2 ;;
    --) shift; UP_ARGS=("$@"); break ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ $BUILD -eq 1 ]]; then
  echo "[up.sh] building g1-base..."
  docker build -f "$HERE/Dockerfile.base" -t g1-base "$ROOT"
  echo "[up.sh] building g1-sim..."
  docker build -f "$HERE/Dockerfile.sim" -t g1-sim "$ROOT"
fi

COMPOSE=(docker compose -f "$HERE/compose.yaml")
if [[ -n "$PROFILES" ]]; then
  IFS=',' read -ra PP <<< "$PROFILES"
  for p in "${PP[@]}"; do COMPOSE+=(--profile "$p"); done
fi

if [[ $SIM_ONLY -eq 1 ]]; then
  exec "${COMPOSE[@]}" up "${UP_ARGS[@]}" sim
else
  exec "${COMPOSE[@]}" up "${UP_ARGS[@]}"
fi
