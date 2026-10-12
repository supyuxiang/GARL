#!/usr/bin/env bash
# Coach cold-start SFT via verl MultiTurnSFTDataset
set -xeuo pipefail

NPROC=${NPROC:-8}
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
TRAIN=${TRAIN:-$HOME/data/garl/coach_sft.parquet}
SAVE=${SAVE:-checkpoints/garl/coach_sft}

GARL_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${GARL_ROOT}:${GARL_ROOT}/verl:${PYTHONPATH:-}"
cd "${GARL_ROOT}/verl"
torchrun --nnodes=1 --nproc_per_node="${NPROC}" \
  -m verl.trainer.sft_trainer \
  data.train_files="${TRAIN}" \
  data.val_files=null \
  data.messages_key=messages \
  model.path="${MODEL}" \
  model.use_remove_padding=true \
  engine=fsdp \
  trainer.default_local_dir="${SAVE}" \
  trainer.project_name=garl \
  trainer.experiment_name=coach_sft \
  trainer.total_epochs=3 \
  optim.lr=1e-5 \
  optim.weight_decay=1e-2 \
  "$@"
