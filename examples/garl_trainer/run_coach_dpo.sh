#!/usr/bin/env bash
set -xeuo pipefail

GARL_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${GARL_ROOT}:${GARL_ROOT}/verl:${PYTHONPATH:-}"

NPROC=${NPROC:-8}
MODEL=${MODEL:-checkpoints/garl/coach_sft}
REF=${REF:-Qwen/Qwen2.5-7B-Instruct}
TRAIN=${TRAIN:-$HOME/data/garl/coach_dpo.parquet}
SAVE=${SAVE:-checkpoints/garl/coach_dpo}

torchrun --nnodes=1 --nproc_per_node="${NPROC}" \
  -m garl.verl.coach_dpo_trainer \
  model.path="${MODEL}" \
  model.ref_path="${REF}" \
  data.train_files="${TRAIN}" \
  trainer.default_local_dir="${SAVE}" \
  trainer.total_training_steps="${STEPS:-300}" \
  trainer.dpo_beta="${BETA:-0.1}" \
  "$@"
