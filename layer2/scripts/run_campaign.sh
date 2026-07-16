#!/usr/bin/env bash
# Real-LLM campaign wrapper (agent C1) — see scripts/campaign.py for all flags.
#
#   scripts/run_campaign.sh --preflight
#   scripts/run_campaign.sh --backends mock-compliant,mock-violator --reps 1 --parallel 2
#   scripts/run_campaign.sh --backends gpt --reps 1 --turns 10          # smoke
#   scripts/run_campaign.sh --backends claude,gpt,gemini --reps 5 --parallel 2
#
# Requires docker compose (images built via docker/up.sh), the
# degradation_test sibling checkout, host python3 with matplotlib, and for
# real backends: ANTHROPIC_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$HERE/campaign.py" "$@"
