"""Convert raw math JSONL into verl Parquet schemas for SFT / GRPO / Coach DPO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from garl.prompts import (
    ACTOR_REGENERATION_PROMPT,
    ACTOR_USER_PROMPT,
    COACH_GENERATION_PROMPT,
    SYSTEM_PROMPT,
)


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _get_q_a(row: dict) -> tuple[str, str]:
    q = (row.get("question") or row.get("problem") or "").strip()
    a = (row.get("answer") or row.get("ground_truth") or row.get("solution") or "").strip()
    return q, a


def build_coach_sft_parquet(rows: list[dict], out: Path) -> None:
    records = []
    for i, row in enumerate(rows):
        q, _ = _get_q_a(row)
        guidance = row.get("guidance", "").strip()
        initial = row.get("initial_solution", "").strip()
        if not guidance or not initial:
            continue
        user = COACH_GENERATION_PROMPT.format(question=q, solution=initial)
        records.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": guidance},
                ]
            }
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(out)
    print(f"coach SFT: {len(records)} -> {out}")


def build_actor_solo_parquet(rows: list[dict], out: Path, data_source: str = "garl/math") -> None:
    records = []
    for i, row in enumerate(rows):
        q, gt = _get_q_a(row)
        if not q or not gt:
            continue
        user = ACTOR_USER_PROMPT.format(question=q)
        records.append(
            {
                "data_source": data_source,
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": gt},
                "extra_info": {"index": i, "garl_mode": "solo"},
            }
        )
    pd.DataFrame(records).to_parquet(out)
    print(f"actor solo RL: {len(records)} -> {out}")


def build_actor_guided_parquet(
    rows: list[dict],
    out: Path,
    data_source: str = "garl/math",
) -> None:
    records = []
    for i, row in enumerate(rows):
        q, gt = _get_q_a(row)
        initial = row.get("initial_solution", "").strip()
        guidance = row.get("guidance", "").strip()
        if not all([q, gt, initial, guidance]):
            continue
        user = ACTOR_REGENERATION_PROMPT.format(
            question=q,
            solution=initial,
            guidance=guidance,
        )
        records.append(
            {
                "data_source": data_source,
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": gt},
                "extra_info": {"index": i, "garl_mode": "guided"},
            }
        )
    pd.DataFrame(records).to_parquet(out)
    print(f"actor guided RL: {len(records)} -> {out}")


def build_hybrid_actor_parquet(
    solo_rows: list[dict],
    guided_rows: list[dict],
    out: Path,
) -> None:
    solo_path = out.with_name(out.stem + "_solo.parquet")
    guided_path = out.with_name(out.stem + "_guided.parquet")
    build_actor_solo_parquet(solo_rows, solo_path)
    build_actor_guided_parquet(guided_rows, guided_path)
    df = pd.concat([pd.read_parquet(solo_path), pd.read_parquet(guided_path)], ignore_index=True)
    df.to_parquet(out)
    print(f"actor hybrid RL: {len(df)} -> {out}")


def build_coach_dpo_parquet(rows: list[dict], out: Path) -> None:
    """Parquet for garl.verl.coach_dpo_dataset.CoachDPODataset."""
    records = []
    for i, row in enumerate(rows):
        q = row.get("question", "").strip()
        initial = row.get("initial_solution", "").strip()
        chosen = row.get("chosen", "").strip()
        rejected = row.get("rejected", "").strip()
        if not all([q, initial, chosen, rejected]):
            continue
        user = COACH_GENERATION_PROMPT.format(question=q, solution=initial)
        records.append(
            {
                "prompt_messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                "chosen": chosen,
                "rejected": rejected,
                "extra_info": {"index": i},
            }
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(out)
    print(f"coach DPO: {len(records)} -> {out}")


def main() -> None:
    p = argparse.ArgumentParser(description="Build verl Parquet files for GARL")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument(
        "--task",
        choices=["coach_sft", "actor_solo", "actor_guided", "actor_hybrid", "coach_dpo"],
        required=True,
    )
    p.add_argument("--guided-input", type=Path, default=None, help="For actor_hybrid guided rows")
    args = p.parse_args()
    rows = _read_jsonl(args.input)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.task == "coach_sft":
        build_coach_sft_parquet(rows, args.out_dir / "coach_sft.parquet")
    elif args.task == "actor_solo":
        build_actor_solo_parquet(rows, args.out_dir / "actor_solo.parquet")
    elif args.task == "actor_guided":
        build_actor_guided_parquet(rows, args.out_dir / "actor_guided.parquet")
    elif args.task == "actor_hybrid":
        guided = _read_jsonl(args.guided_input) if args.guided_input else rows
        build_hybrid_actor_parquet(rows, guided, args.out_dir / "actor_hybrid.parquet")
    elif args.task == "coach_dpo":
        build_coach_dpo_parquet(rows, args.out_dir / "coach_dpo.parquet")


if __name__ == "__main__":
    main()
