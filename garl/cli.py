"""CLI helpers; training/eval on verl is via examples/garl_trainer/."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

from garl.eval.run import evaluate_file
from garl.generation import GenConfig
from garl.inference import DualAgentPipeline
from garl.training.load import load_causal_lm, load_tokenizer


def main_train() -> None:
    print(
        "GARL training runs on verl. Examples:\n"
        "  bash examples/garl_trainer/run_coach_sft.sh\n"
        "  bash examples/garl_trainer/run_actor_grpo.sh\n"
        "  bash examples/garl_trainer/run_coach_dpo.sh\n"
        "  bash examples/garl_trainer/run_garl_alternate.sh\n"
        "Data: python -m garl.verl.preprocess --help",
        file=sys.stderr,
    )
    raise SystemExit(0)


def main_eval() -> None:
    p = argparse.ArgumentParser(description="Evaluate dual-agent pass@1")
    p.add_argument("--actor", required=True)
    p.add_argument("--coach", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-samples", type=int, default=-1)
    args = p.parse_args()
    metrics = evaluate_file(args.actor, args.coach, args.data, args.output, args.max_samples)
    print(json.dumps(metrics, indent=2))


def main_infer() -> None:
    p = argparse.ArgumentParser(description="Dual-agent inference")
    p.add_argument("--actor", required=True)
    p.add_argument("--coach", required=True)
    p.add_argument("--question", required=True)
    args = p.parse_args()
    tokenizer = load_tokenizer(args.actor)
    actor = load_causal_lm(args.actor)
    coach = load_causal_lm(args.coach)
    pipe = DualAgentPipeline(actor, coach, tokenizer, GenConfig())
    result = pipe.run(args.question)
    print(json.dumps(result.__dict__, indent=2, ensure_ascii=False))


def main_preprocess() -> None:
    subprocess.check_call([sys.executable, "-m", "garl.verl.preprocess", *sys.argv[1:]])
