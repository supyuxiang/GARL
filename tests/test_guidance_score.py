from garl.guidance_score import build_preference_pair, guidance_outcome_score


def test_outcome_score_fix_incorrect():
    assert guidance_outcome_score(False, True) == 1.0
    assert guidance_outcome_score(True, False) == 0.0
    assert guidance_outcome_score(True, True) == 0.1


def test_preference_pair_prefers_fixing_answer():
    pair = build_preference_pair(
        question="Q",
        ground_truth="42",
        initial_solution="wrong \\boxed{1}",
        guidance_candidates=["good guide", "bad guide"],
        refined_solutions=[
            "reasoning \\boxed{42}",
            "reasoning \\boxed{0}",
        ],
        token_lengths=[100, 50],
        alpha=0.3,
    )
    assert pair is not None
    assert pair[0] == "good guide"


def test_preference_pair_length_tiebreak():
    pair = build_preference_pair(
        question="Q",
        ground_truth="2",
        initial_solution="\\boxed{1}",
        guidance_candidates=["long path", "short path"],
        refined_solutions=[
            "x" * 10 + " \\boxed{2}",
            " \\boxed{2}",
        ],
        token_lengths=[200, 20],
        alpha=0.3,
    )
    assert pair is not None
    assert pair[0] == "short path"
