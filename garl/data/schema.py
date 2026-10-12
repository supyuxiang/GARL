"""JSONL schemas for cold-start SFT and Coach DPO preferences."""

from __future__ import annotations

from typing import Any, TypedDict


class CoachColdStartRow(TypedDict):
    question: str
    initial_solution: str
    guidance: str


class CoachPreferenceRow(TypedDict):
    question: str
    initial_solution: str
    chosen: str
    rejected: str


class RLRow(TypedDict):
    question: str
    answer: str
    guidance: str | None


def row_get(row: dict[str, Any], *keys: str, default: str = "") -> str:
    for k in keys:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return default
