"""Custom reward for verl GRPO (Actor training).

Configure in Hydra:
  reward.custom_reward_function.path=/path/to/GARL/garl/verl/reward.py
  reward.custom_reward_function.name=compute_score
"""

from __future__ import annotations

from garl.format import actor_follows_format
from garl.math_grade import is_solution_correct


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    **kwargs,
):
    """Returns dict so verl can log outcome / format separately (GDPO-compatible)."""
    del data_source, extra_info, kwargs
    question = ""
    outcome = 1.0 if is_solution_correct(question, solution_str, str(ground_truth)) else 0.0
    fmt = 0.0 if actor_follows_format(solution_str) else -1.0
    return outcome + fmt
