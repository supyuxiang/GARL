"""On-policy Coach preference pairs for DPO."""

from __future__ import annotations

import random
from dataclasses import dataclass

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from garl.chat import actor_guided_messages, apply_chat_template, coach_messages
from garl.generation import GenConfig, count_tokens, generate_text
from garl.guidance_score import build_preference_pair
from garl.data.schema import CoachPreferenceRow, row_get


@dataclass
class PreferenceConfig:
    alpha: float = 0.3
    gen: GenConfig | None = None


def sample_coach_guidance(
    coach: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    question: str,
    initial_solution: str,
    cfg: PreferenceConfig,
) -> str:
    gen = cfg.gen or GenConfig(temperature=1.0, top_p=1.0)
    prompt = apply_chat_template(
        tokenizer,
        coach_messages(question, initial_solution),
        add_generation_prompt=True,
    )
    return generate_text(coach, tokenizer, prompt, gen)


def build_preference_row(
    actor: PreTrainedModel,
    coach: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    question: str,
    ground_truth: str,
    initial_solution: str,
    cfg: PreferenceConfig,
) -> CoachPreferenceRow | None:
    gen = cfg.gen or GenConfig(temperature=1.0, top_p=1.0)
    candidates = [
        sample_coach_guidance(coach, tokenizer, question, initial_solution, cfg),
        sample_coach_guidance(coach, tokenizer, question, initial_solution, cfg),
    ]
    refined: list[str] = []
    lengths: list[int] = []
    for c in candidates:
        prompt = apply_chat_template(
            tokenizer,
            actor_guided_messages(question, initial_solution, c),
            add_generation_prompt=True,
        )
        sol = generate_text(actor, tokenizer, prompt, gen)
        refined.append(sol)
        lengths.append(count_tokens(tokenizer, sol))

    pair = build_preference_pair(
        question=question,
        ground_truth=ground_truth,
        initial_solution=initial_solution,
        guidance_candidates=candidates,
        refined_solutions=refined,
        token_lengths=lengths,
        alpha=cfg.alpha,
    )
    if pair is None:
        return None
    chosen, rejected = pair
    return {
        "question": question,
        "initial_solution": initial_solution,
        "chosen": chosen,
        "rejected": rejected,
    }


def build_preference_dataset(
    rows: list[dict],
    actor: PreTrainedModel,
    coach: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    cfg: PreferenceConfig | None = None,
    max_rows: int | None = None,
    seed: int = 42,
) -> list[CoachPreferenceRow]:
    cfg = cfg or PreferenceConfig()
    rng = random.Random(seed)
    pool = list(rows)
    rng.shuffle(pool)
    if max_rows:
        pool = pool[:max_rows]

    out: list[CoachPreferenceRow] = []
    for row in pool:
        question = row_get(row, "question", "problem")
        gt = row_get(row, "answer", "ground_truth")
        initial = row_get(row, "initial_solution")
        if not initial:
            from garl.chat import actor_messages

            ap = apply_chat_template(tokenizer, actor_messages(question), add_generation_prompt=True)
            initial = generate_text(actor, tokenizer, ap, cfg.gen or GenConfig())
        pref = build_preference_row(
            actor, coach, tokenizer, question, gt, initial, cfg
        )
        if pref is not None:
            out.append(pref)
    return out
