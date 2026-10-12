"""Structured output parsing and format checks for Actor and Coach."""

from __future__ import annotations

import re
from dataclasses import dataclass

REFLECTION_RE = re.compile(r"<reflection>(.*?)</reflection>", re.DOTALL | re.IGNORECASE)
PLAN_RE = re.compile(r"<plan>(.*?)</plan>", re.DOTALL | re.IGNORECASE)
THINKING_RE = re.compile(
    r"(?:<(?:redacted_)?think(?:ing)?>)(.*?)(?:</(?:redacted_)?think(?:ing)?>)",
    re.DOTALL | re.IGNORECASE,
)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")


@dataclass(frozen=True)
class ParsedCoachResponse:
    reflection: str
    plan: str
    raw: str


@dataclass(frozen=True)
class ParsedActorResponse:
    thinking: str
    answer: str
    raw: str


def parse_coach_response(text: str) -> ParsedCoachResponse | None:
    if not text:
        return None
    m_ref = REFLECTION_RE.search(text)
    m_plan = PLAN_RE.search(text)
    if not m_ref or not m_plan:
        return None
    return ParsedCoachResponse(
        reflection=m_ref.group(1).strip(),
        plan=m_plan.group(1).strip(),
        raw=text,
    )


def compose_coach_response(reflection: str, plan: str) -> str:
    reflection = reflection.strip()
    plan = plan.strip()
    if not reflection.startswith("<reflection>"):
        reflection = f"<reflection>{reflection}</reflection>"
    if not plan.startswith("<plan>"):
        plan = f"<plan>{plan}</plan>"
    return f"{reflection}\n{plan}"


def parse_actor_response(text: str) -> ParsedActorResponse | None:
    if not text:
        return None
    m_think = THINKING_RE.search(text)
    m_ans = ANSWER_RE.search(text)
    if not m_think or not m_ans:
        return None
    return ParsedActorResponse(
        thinking=m_think.group(1).strip(),
        answer=m_ans.group(1).strip(),
        raw=text,
    )


def actor_follows_format(text: str) -> bool:
    return parse_actor_response(text) is not None


def coach_follows_format(text: str) -> bool:
    return parse_coach_response(text) is not None


def extract_boxed(text: str) -> str | None:
    if not text:
        return None
    matches = BOXED_RE.findall(text)
    if matches:
        return matches[-1].strip()
    m = ANSWER_RE.search(text)
    if m:
        inner = m.group(1)
        boxed = BOXED_RE.findall(inner)
        if boxed:
            return boxed[-1].strip()
        return inner.strip()
    return None
