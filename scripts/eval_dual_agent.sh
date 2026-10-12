#!/usr/bin/env bash
set -euo pipefail

ACTOR="${1:?actor checkpoint}"
COACH="${2:?coach checkpoint}"
DATA="${3:?eval jsonl}"
OUT="${4:-outputs/eval_predictions.jsonl}"

garl-eval --actor "$ACTOR" --coach "$COACH" --data "$DATA" --output "$OUT"
