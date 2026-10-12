"""Build Coach cold-start dataset D_coach (rejection sampling + optional LLM judge)."""

from __future__ import annotations

import random
from dataclasses import dataclass

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from garl.chat import actor_messages, apply_chat_template
from garl.format import compose_coach_response, parse_coach_response
from garl.generation import GenConfig, generate_text
from garl.prompts import COMPOSITION_PROMPT, PLAN_JUDGE_PROMPT, PLAN_PROMPT, REFLECTION_JUDGE_PROMPT, REFLECTION_PROMPT
from garl.data.schema import CoachColdStartRow, row_get


@dataclass
class ColdStartConfig:
    reflection_samples: int = 4
    plan_samples: int = 4
    gen: GenConfig | None = None


def _judge_accept(judge_output: str) -> bool:
    return judge_output.strip().lower().startswith("accept")


def rejection_sample_reflection(
    coach: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    question: str,
    solution: str,
    judge_fn,
    cfg: ColdStartConfig,
) -> str | None:
    gen = cfg.gen or GenConfig(temperature=1.0, top_p=1.0)
    user = REFLECTION_PROMPT.format(question=question, solution=solution)
    messages = [{"role": "user", "content": user}]
    prompt = apply_chat_template(tokenizer, messages, add_generation_prompt=True)
    for _ in range(cfg.reflection_samples):
        text = generate_text(coach, tokenizer, prompt, gen)
        parsed = parse_coach_response(text) or parse_coach_response(f"<reflection>{text}</reflection>")
        reflection = parsed.reflection if parsed else text
        judge_prompt = REFLECTION_JUDGE_PROMPT.format(
            question=question, solution=solution, reflection=reflection
        )
        if _judge_accept(judge_fn(judge_prompt)):
            return reflection
    return None


def rejection_sample_plan(
    coach: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    question: str,
    solution: str,
    reflection: str,
    judge_fn,
    cfg: ColdStartConfig,
) -> str | None:
    gen = cfg.gen or GenConfig(temperature=1.0, top_p=1.0)
    user = PLAN_PROMPT.format(question=question, solution=solution, reflection=reflection)
    messages = [{"role": "user", "content": user}]
    prompt = apply_chat_template(tokenizer, messages, add_generation_prompt=True)
    for _ in range(cfg.plan_samples):
        text = generate_text(coach, tokenizer, prompt, gen)
        parsed = parse_coach_response(text) or parse_coach_response(f"<plan>{text}</plan>")
        plan = parsed.plan if parsed else text
        judge_prompt = PLAN_JUDGE_PROMPT.format(
            question=question,
            solution=solution,
            reflection=reflection,
            plan=plan,
        )
        if _judge_accept(judge_fn(judge_prompt)):
            return plan
    return None


def compose_with_teacher(reflection: str, plan: str, compose_fn) -> str:
    prompt = COMPOSITION_PROMPT.format(reflection=reflection, plan=plan)
    out = compose_fn(prompt).strip()
    parsed = parse_coach_response(out)
    if parsed:
        return compose_coach_response(parsed.reflection, parsed.plan)
    return compose_coach_response(reflection, plan)


def build_cold_start_row(
    actor: PreTrainedModel,
    coach: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    row: dict,
    judge_fn,
    compose_fn,
    cfg: ColdStartConfig,
) -> CoachColdStartRow | None:
    question = row_get(row, "question", "problem")

    ap = apply_chat_template(tokenizer, actor_messages(question), add_generation_prompt=True)
    initial_solution = generate_text(actor, tokenizer, ap, cfg.gen or GenConfig())

    reflection = rejection_sample_reflection(
        coach, tokenizer, question, initial_solution, judge_fn, cfg
    )
    if reflection is None:
        return None
    plan = rejection_sample_plan(
        coach, tokenizer, question, initial_solution, reflection, judge_fn, cfg
    )
    if plan is None:
        return None
    guidance = compose_with_teacher(reflection, plan, compose_fn)
    return {
        "question": question,
        "initial_solution": initial_solution,
        "guidance": guidance,
    }


def build_cold_start_dataset(
    rows: list[dict],
    actor: PreTrainedModel,
    coach: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    judge_fn,
    compose_fn,
    cfg: ColdStartConfig | None = None,
    max_rows: int | None = None,
    seed: int = 42,
) -> list[CoachColdStartRow]:
    cfg = cfg or ColdStartConfig()
    rng = random.Random(seed)
    pool = list(rows)
    rng.shuffle(pool)
    if max_rows is not None:
        pool = pool[:max_rows]

    out: list[CoachColdStartRow] = []
    for row in pool:
        item = build_cold_start_row(actor, coach, tokenizer, row, judge_fn, compose_fn, cfg)
        if item is not None:
            out.append(item)
    return out
