from garl.format import actor_follows_format, coach_follows_format, extract_boxed, parse_coach_response


def test_coach_parse():
    text = "<reflection>err</reflection>\n<plan>step 1</plan>"
    parsed = parse_coach_response(text)
    assert parsed is not None
    assert parsed.reflection == "err"
    assert coach_follows_format(text)


def test_actor_format_and_boxed():
    assert not actor_follows_format("only<answer>\\boxed{12}</answer>")
    text = "step<answer>Final: \\boxed{12}</answer>"
    assert actor_follows_format(text)
    assert extract_boxed(text) == "12"
