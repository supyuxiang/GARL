"""Actor reward functions (Eq. actor_reward in the GARL paper)."""

from __future__ import annotations

from garl.format import actor_follows_format
from garl.math_grade import is_solution_correct


def outcome_reward(question: str, solution: str, ground_truth: str) -> float:
    return 1.0 if is_solution_correct(question, solution, ground_truth) else 0.0


def format_reward(solution: str) -> float:
    return 0.0 if actor_follows_format(solution) else -1.0


def actor_reward(question: str, solution: str, ground_truth: str) -> float:
    return outcome_reward(question, solution, ground_truth) + format_reward(solution)
