#!/usr/bin/env bash
# Hybrid-mode Actor GRPO (solo + guided parquet) via verl
set -xeuo pipefail

MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
TRAIN=${TRAIN:-$HOME/data/garl/actor_hybrid.parquet}
TEST=${TEST:-$HOME/data/garl/math500.parquet}
NPROC=${NPROC:-8}
SAVE=${SAVE:-checkpoints/garl/actor_grpo}

GARL_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${GARL_ROOT}:${GARL_ROOT}/verl:${PYTHONPATH:-}"
REWARD="${GARL_ROOT}/garl/verl/reward.py"

cd "${GARL_ROOT}/verl"
torchrun --nnodes=1 --nproc_per_node="${NPROC}" \
  -m verl.trainer.main_ppo \
  data.train_files="${TRAIN}" \
  data.val_files="${TEST}" \
  data.return_raw_chat=True \
  data.train_batch_size=256 \
  data.max_prompt_length=2048 \
  data.max_response_length=4096 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path="${MODEL}" \
  actor_rollout_ref.actor.optim.lr=1e-5 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.04 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  reward.custom_reward_function.path="${REWARD}" \
  reward.custom_reward_function.name=compute_score \
  algorithm.adv_estimator=grpo \
  trainer.project_name=garl \
  trainer.experiment_name=actor_grpo \
  trainer.default_local_dir="${SAVE}" \
  trainer.total_epochs=1 \
  trainer.n_gpus_per_node="${NPROC}" \
  "$@"
