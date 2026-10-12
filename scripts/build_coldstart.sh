#!/usr/bin/env bash
set -euo pipefail

# Build Coach cold-start JSONL (requires GPU + HF model).
# Example:
#   bash scripts/build_coldstart.sh \
#     /path/to/openthoughts_sample.jsonl \
#     outputs/coach_coldstart.jsonl \
#     Qwen/Qwen2.5-7B-Instruct

INPUT="${1:?input jsonl}"
OUTPUT="${2:?output jsonl}"
MODEL="${3:?hf model id}"

garl-train build-coldstart \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --model "$MODEL" \
  --max-rows "${MAX_ROWS:-1000}" \
  ${USE_OPENAI_JUDGE:+--use-openai-judge}
