"""Benchmark evaluation (pass@1 on refined solution)."""

from __future__ import annotations

import json
from pathlib import Path

from tqdm import tqdm

from garl.data.io import read_jsonl, write_jsonl
from garl.data.schema import row_get
from garl.generation import GenConfig
from garl.inference import DualAgentPipeline
from garl.math_grade import is_solution_correct
from garl.training.load import load_causal_lm, load_tokenizer


def evaluate_file(
    actor_path: str,
    coach_path: str,
    data_path: str,
    output_path: str,
    max_samples: int = -1,
) -> dict:
    tokenizer = load_tokenizer(actor_path)
    actor = load_causal_lm(actor_path)
    coach = load_causal_lm(coach_path)
    pipe = DualAgentPipeline(actor, coach, tokenizer, GenConfig(temperature=0.6, top_p=0.95))

    rows = list(read_jsonl(data_path))
    if max_samples > 0:
        rows = rows[:max_samples]

    results = []
    correct = 0
    for row in tqdm(rows, desc="eval"):
        q = row_get(row, "question", "problem")
        gt = row_get(row, "answer", "ground_truth")
        out = pipe.run(q, ground_truth=gt)
        ok = bool(out.refined_correct)
        correct += int(ok)
        results.append(
            {
                "question": q,
                "ground_truth": gt,
                "initial_solution": out.initial_solution,
                "guidance": out.guidance,
                "refined_solution": out.refined_solution,
                "initial_correct": out.initial_correct,
                "refined_correct": out.refined_correct,
            }
        )

    acc = 100.0 * correct / max(len(rows), 1)
    metrics = {"num_samples": len(rows), "pass_at_1_refined": acc}
    write_jsonl(output_path, results)
    Path(output_path).with_suffix(".metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return metrics
