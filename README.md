# GARL: Guidance-Aware Reinforcement Learning

Dual-agent math reasoning (**Actor** + **Coach**) with co-evolutionary RL, implemented on **[verl](https://github.com/volcengine/verl)** (GRPO / SFT / rollout).

Paper: `paper/main.tex` — Coach produces `<reflection>` + `<plan>`; Actor trains with hybrid GRPO (question-only + guidance-conditioned); Coach trains with DPO from downstream improvement preferences.

## Layout

| Path | Role |
|------|------|
| `garl/` | Prompts, format checks, math grading, inference |
| `garl/verl/` | verl reward, Parquet preprocess, Coach DPO dataset & trainer |
| `verl/` | Pinned verl fork (Actor GRPO via `main_ppo`, Coach SFT via `sft_trainer`) |
| `examples/garl_trainer/` | End-to-end shell scripts |
| `paper/` | ACL paper source |

## Install

```bash
cd /root/GARL
pip install -e .
pip install -e verl/   # or: pip install -r verl/requirements.txt
export PYTHONPATH=/root/GARL:/root/GARL/verl:$PYTHONPATH
```

## Data pipeline (JSONL → Parquet)

Raw fields: `question`, `answer`; for Coach SFT add `initial_solution`, `guidance`; for DPO add `chosen` / `rejected`.

```bash
python -m garl.verl.preprocess --input data/openthoughts.jsonl --out-dir ~/data/garl --task coach_sft
python -m garl.verl.preprocess --input data/deepscaler.jsonl --out-dir ~/data/garl --task actor_solo
python -m garl.verl.preprocess --input data/round_guided.jsonl --out-dir ~/data/garl --task actor_hybrid \
  --guided-input data/round_guided.jsonl
python -m garl.verl.preprocess --input data/coach_prefs.jsonl --out-dir ~/data/garl --task coach_dpo
```

## Training (verl)

**1. Coach cold-start SFT** (3 epochs, paper default):

```bash
bash examples/garl_trainer/run_coach_sft.sh
```

**2. Actor hybrid GRPO** (custom reward = outcome + format):

```bash
bash examples/garl_trainer/run_actor_grpo.sh
```

Uses `reward.custom_reward_function` → `garl/verl/reward.py`. Dataset `data_source` can be `garl/math` (registered in `verl/verl/utils/reward_score/__init__.py`).

**3. Coach DPO** (preference Parquet, β=0.1, 300 steps):

```bash
bash examples/garl_trainer/run_coach_dpo.sh
```

**4. Alternating round** (refresh hybrid Parquet, then Coach DPO + Actor GRPO):

```bash
bash examples/garl_trainer/run_garl_alternate.sh
```

Guided rows (`round_guided.jsonl`) should contain `initial_solution` and `guidance` from the current Coach (e.g. verl `main_generation_server` on Actor rollouts). Re-run each co-evolution round as in the paper.

## Inference (local HF)

```bash
garl-infer --actor checkpoints/garl/actor_grpo --coach checkpoints/garl/coach_dpo/coach_dpo_hf --question "..."
```

## Tests

```bash
pytest tests/
```

## Legacy trainers

`garl/training/*` remains as a small HF reference; **production training should use verl scripts above**.

## License

Apache-2.0 — see [LICENSE](LICENSE).
