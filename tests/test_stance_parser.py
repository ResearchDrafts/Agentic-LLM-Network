import pytest

from sandbox.stance_parser import StanceParseFailure, clamp_and_validate_scale, parse_stance


def test_well_formed_stance_line_parses_value_and_strips_line_from_reason():
    value, reason = parse_stance("STANCE: 5\nBecause the evidence is compelling.")

    assert value == 5.0
    assert "STANCE" not in reason
    assert reason == "Because the evidence is compelling."


def test_missing_stance_line_raises_parse_failure():
    with pytest.raises(StanceParseFailure):
        parse_stance("I think this is a reasonable position overall.")


def test_non_numeric_stance_value_raises_parse_failure():
    with pytest.raises(StanceParseFailure):
        parse_stance("STANCE: high\nBecause I feel strongly.")


def test_negative_and_decimal_values_parse_correctly():
    value, _ = parse_stance("STANCE: -2.5\nSome reasoning.")
    assert value == -2.5


def test_stance_pattern_is_case_insensitive():
    value, _ = parse_stance("stance: 3\nlowercase label still matches.")
    assert value == 3.0


def test_multiple_stance_like_substrings_first_match_wins_all_stripped_from_reason():
    raw = "STANCE: 4\nI'm undecided. STANCE: 7 was my earlier draft answer."
    value, reason = parse_stance(raw)

    # First (leftmost) match's value is the one returned.
    assert value == 4.0
    # re.sub with no count replaces every match, not just the first --
    # both STANCE occurrences are stripped from reason_text.
    assert "STANCE" not in reason
    assert reason == "I'm undecided.  was my earlier draft answer."


def test_clamp_and_validate_scale_value_inside_range_passes_through_unchanged():
    assert clamp_and_validate_scale(4.0, [1, 2, 3, 4, 5, 6, 7]) == 4.0


def test_clamp_and_validate_scale_value_below_range_raises():
    with pytest.raises(StanceParseFailure):
        clamp_and_validate_scale(0.0, [1, 2, 3, 4, 5, 6, 7])


def test_clamp_and_validate_scale_value_above_range_raises():
    with pytest.raises(StanceParseFailure):
        clamp_and_validate_scale(8.0, [1, 2, 3, 4, 5, 6, 7])
