"""Coach preference scores for DPO pair construction (Sec. coach DPO)."""

from __future__ import annotations

from garl.math_grade import is_solution_correct


def guidance_outcome_score(
    initial_correct: bool,
    refined_correct: bool,
) -> float:
    if not initial_correct and refined_correct:
        return 1.0
    if initial_correct and not refined_correct:
        return 0.0
    return 0.1


def guidance_length_score_from_flags(
    both_refined_correct: bool,
    is_shorter_among_correct_pair: bool,
) -> float:
    if both_refined_correct and is_shorter_among_correct_pair:
        return 1.0
    return 0.0


def guidance_final_score(outcome: float, length: float, alpha: float = 0.3) -> float:
    return outcome + alpha * length


def build_preference_pair(
    question: str,
    ground_truth: str,
    initial_solution: str,
    guidance_candidates: list[str],
    refined_solutions: list[str],
    token_lengths: list[int],
    alpha: float = 0.3,
) -> tuple[str, str] | None:
    """Return (c_plus, c_minus) or None if scores tie."""
    if len(guidance_candidates) != 2 or len(refined_solutions) != 2:
        raise ValueError("expect two guidance candidates and two refined solutions")

    s_correct = is_solution_correct(question, initial_solution, ground_truth)
    r_correct = [
        is_solution_correct(question, refined_solutions[0], ground_truth),
        is_solution_correct(question, refined_solutions[1], ground_truth),
    ]
    both_correct = r_correct[0] and r_correct[1]

    scores: list[float] = []
    for j in range(2):
        sc_out = guidance_outcome_score(s_correct, r_correct[j])
        shorter_idx = min(range(2), key=lambda k: token_lengths[k])
        sc_len = guidance_length_score_from_flags(
            both_refined_correct=both_correct,
            is_shorter_among_correct_pair=(j == shorter_idx),
        )
        scores.append(guidance_final_score(sc_out, sc_len, alpha=alpha))

    if scores[0] == scores[1]:
        return None
    if scores[0] > scores[1]:
        return guidance_candidates[0], guidance_candidates[1]
    return guidance_candidates[1], guidance_candidates[0]
