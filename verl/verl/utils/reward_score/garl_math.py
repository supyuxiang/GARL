# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""GARL Actor rule-based reward (outcome + format)."""

from garl.format import actor_follows_format
from garl.math_grade import is_solution_correct


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    del data_source, extra_info, kwargs
    outcome = 1.0 if is_solution_correct("", solution_str, str(ground_truth)) else 0.0
    fmt = 0.0 if actor_follows_format(solution_str) else -1.0
    return outcome + fmt


__all__ = ["compute_score"]
