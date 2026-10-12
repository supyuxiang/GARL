#!/usr/bin/env bash
set -euo pipefail

BASE="${1:?base hf model}"
RL_DATA="${2:?deepscaler jsonl}"
WORK="${3:-outputs/garl_rl}"

garl-train garl --base-model "$BASE" --rl-data "$RL_DATA" --work-dir "$WORK" --rounds "${ROUNDS:-5}"
