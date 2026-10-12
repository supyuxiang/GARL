"""Chat-template helpers for Actor and Coach."""

from __future__ import annotations

from garl import prompts


def actor_messages(question: str, system: str = prompts.SYSTEM_PROMPT) -> list[dict]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": prompts.ACTOR_USER_PROMPT.format(question=question)},
    ]


def actor_guided_messages(
    question: str,
    initial_solution: str,
    guidance: str,
) -> list[dict]:
    user = prompts.ACTOR_REGENERATION_PROMPT.format(
        question=question,
        solution=initial_solution,
        guidance=guidance,
    )
    return [
        {"role": "system", "content": prompts.SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def coach_messages(question: str, solution: str) -> list[dict]:
    user = prompts.COACH_GENERATION_PROMPT.format(question=question, solution=solution)
    return [
        {"role": "system", "content": prompts.SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def apply_chat_template(tokenizer, messages: list[dict], add_generation_prompt: bool = True) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    parts = []
    for m in messages:
        parts.append(f"{m['role'].upper()}: {m['content']}")
    if add_generation_prompt:
        parts.append("ASSISTANT:")
    return "\n\n".join(parts)
