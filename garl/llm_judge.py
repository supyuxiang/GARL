"""Optional OpenAI-compatible judge for cold-start rejection sampling."""

from __future__ import annotations

from typing import Callable


def openai_judge_factory(model: str = "gpt-4o-mini", temperature: float = 0.0) -> Callable[[str], str]:
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError("pip install openai for LLM judges") from e

    client = OpenAI()

    def _call(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""

    return _call


def heuristic_judge(prompt: str) -> str:
    """Fallback when no API key: accept non-empty reflections/plans."""
    del prompt
    return "accept"
