#!/usr/bin/env bash
# One GARL co-evolution round: (optional) refresh guided data -> Coach DPO -> Actor GRPO
set -euo pipefail

GARL_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${GARL_ROOT}:${GARL_ROOT}/verl:${PYTHONPATH:-}"

BASE_RL=${BASE_RL:-$HOME/data/garl/deepscaler_base.jsonl}
GUIDED_JSONL=${GUIDED_JSONL:-$HOME/data/garl/round_guided.jsonl}
OUT_DATA=${OUT_DATA:-$HOME/data/garl}

mkdir -p "${OUT_DATA}"

python -m garl.verl.preprocess --input "${BASE_RL}" --out-dir "${OUT_DATA}" --task actor_solo
python -m garl.verl.preprocess --input "${GUIDED_JSONL}" --out-dir "${OUT_DATA}" --task actor_guided
python -m garl.verl.preprocess --input "${BASE_RL}" --out-dir "${OUT_DATA}" --task actor_hybrid --guided-input "${GUIDED_JSONL}"

bash "$(dirname "$0")/run_coach_dpo.sh"
bash "$(dirname "$0")/run_actor_grpo.sh" data.train_files="${OUT_DATA}/actor_hybrid.parquet"
