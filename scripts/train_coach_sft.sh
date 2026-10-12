#!/usr/bin/env bash
set -euo pipefail

DATA="${1:?coach coldstart jsonl}"
MODEL="${2:?base model}"
OUT="${3:-outputs/coach_sft}"

garl-train coach-sft --data "$DATA" --model "$MODEL" --output-dir "$OUT" --epochs "${EPOCHS:-3}"
