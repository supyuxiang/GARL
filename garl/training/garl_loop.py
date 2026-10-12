"""Alternating Coach DPO + Actor GRPO (GARL co-evolution)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from garl.data.io import write_jsonl
from garl.data.preferences import PreferenceConfig, build_preference_dataset
from garl.training.actor_grpo import ActorGRPOConfig, train_actor_grpo
from garl.training.coach_dpo import CoachDPOConfig, train_coach_dpo
from garl.training.load import load_causal_lm, load_tokenizer


@dataclass
class GARLTrainConfig:
    base_model: str
    rl_data_path: str
    work_dir: str
    alternating_rounds: int = 5
    coach_dpo_steps: int = 300
    actor_grpo_steps: int = 300
    alpha: float = 0.3
    preference_batch: int = 256


def run_garl_training(cfg: GARLTrainConfig) -> Path:
    work = Path(cfg.work_dir)
    work.mkdir(parents=True, exist_ok=True)

    import json

    rows = [
        json.loads(line)
        for line in Path(cfg.rl_data_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    actor_path = cfg.base_model
    coach_path = cfg.base_model

    for rnd in range(cfg.alternating_rounds):
        tokenizer = load_tokenizer(actor_path)
        actor = load_causal_lm(actor_path)
        coach = load_causal_lm(coach_path)

        pref_rows = build_preference_dataset(
            rows,
            actor,
            coach,
            tokenizer,
            cfg=PreferenceConfig(alpha=cfg.alpha),
            max_rows=cfg.preference_batch,
        )
        pref_path = work / f"round_{rnd}_preferences.jsonl"
        write_jsonl(pref_path, pref_rows)

        coach_out = train_coach_dpo(
            pref_path,
            CoachDPOConfig(
                model_name=coach_path,
                ref_model_name=cfg.base_model,
                output_dir=str(work / f"round_{rnd}"),
                max_steps=cfg.coach_dpo_steps,
            ),
        )
        coach_path = str(coach_out)

        actor_out = train_actor_grpo(
            cfg.rl_data_path,
            ActorGRPOConfig(
                model_name=actor_path,
                ref_model_name=cfg.base_model,
                output_dir=str(work / f"round_{rnd}"),
                max_steps=cfg.actor_grpo_steps,
            ),
        )
        actor_path = str(actor_out)

    final = work / "final"
    final.mkdir(exist_ok=True)
    (final / "actor_path.txt").write_text(actor_path, encoding="utf-8")
    (final / "coach_path.txt").write_text(coach_path, encoding="utf-8")
    return final
