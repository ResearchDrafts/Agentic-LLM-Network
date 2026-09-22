"""Tests for the shared test helpers themselves.

A broken factory produces confusing failures in every module that uses it, so
the contracts Phase 4's orchestrator tests will lean on are pinned here.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sandbox.stance_parser import clamp_and_validate_scale, parse_stance
from tests.factories import (
    STANCE_SCALE,
    make_agent,
    make_backend_response,
    make_interaction,
    make_run,
    make_stance_text,
)


def test_make_run_defaults_are_valid_and_overridable():
    assert make_run().M == 10
    assert make_run(M=3, N=1).M == 3
    assert make_run(M=3, N=1).N == 1


def test_make_run_overrides_still_go_through_validation():
    """Overrides must not bypass the model's own validators: N >= M is
    rejected by ExperimentRun.check_n_less_than_m."""
    with pytest.raises(ValidationError):
        make_run(M=3, N=5)


def test_make_agent_positional_signature_matches_population_builders():
    agent = make_agent("agent_0007", 6.0)
    assert agent.agent_id == "agent_0007"
    assert agent.initial_stance == 6.0


def test_make_interaction_defaults_to_a_valid_generated_text_row():
    interaction = make_interaction("agent_0007", turn=2)
    assert interaction.speaker_agent_id == "agent_0007"
    assert interaction.content_type == "generated_text"
    assert interaction.meme_id is None
    assert interaction.is_valid_for_analysis


def test_make_interaction_can_build_a_meme_row():
    """Phase 4 is the first code to construct one, so the
    check_meme_id_consistency branch is exercised here first."""
    interaction = make_interaction(content_type="meme", meme_id="meme_test_001")
    assert interaction.content_type == "meme"
    assert interaction.meme_id == "meme_test_001"


def test_make_interaction_meme_row_without_meme_id_is_rejected():
    with pytest.raises(ValidationError):
        make_interaction(content_type="meme")


def test_make_interaction_can_build_a_failed_logged_null_row():
    """The soft-failure status the orchestrator writes when a stance parse or
    a fatal gateway error exhausts its budget. Never constructed by any
    pre-Phase-4 test."""
    interaction = make_interaction(
        api_call_status="failed_logged_null",
        reason_text="",
        prompt_token_count=0,
        completion_token_count=0,
        latency_ms=0,
    )
    assert not interaction.is_valid_for_analysis


# --- the contract that matters: replies the real parser accepts ----------


@pytest.mark.parametrize("stance", [1.0, 4.0, 6.5, 7.0])
def test_make_stance_text_round_trips_through_the_real_parser(stance):
    value, reason = parse_stance(make_stance_text(stance, reason="a reason"))
    assert value == stance
    assert reason == "a reason"
    assert clamp_and_validate_scale(value, STANCE_SCALE) == stance


def test_make_backend_response_body_is_parseable():
    """Every pre-Phase-4 mock returned "ok", which parse_stance rejects, so
    nothing in the suite produced a reply an orchestrator could consume."""
    value, _ = parse_stance(make_backend_response(stance=3.0).text)
    assert value == 3.0


def test_make_backend_response_text_override_drives_the_failure_path():
    with pytest.raises(Exception):
        parse_stance(make_backend_response(text="no stance line here").text)


# --- the mock_gateway fixture --------------------------------------------


async def test_mock_gateway_returns_a_parseable_reply(mock_gateway):
    response = await mock_gateway.generate("prompt")
    value, _ = parse_stance(response.text)
    assert clamp_and_validate_scale(value, STANCE_SCALE) == value


async def test_mock_gateway_rejects_calls_that_break_the_real_signature(mock_gateway):
    """autospec is the point of the fixture: a call the real ModelGateway
    would reject must fail here too, rather than silently passing.

    This regressed once already. Reassigning generate to a bare AsyncMock
    (rather than just setting its return_value) silently discards the
    autospec signature, and every other test keeps passing.
    """
    with pytest.raises(TypeError):
        await mock_gateway.generate()  # prompt_text is required


def test_text_only_gateway_reports_no_vision_support(text_only_gateway):
    assert text_only_gateway.supports_vision is False
