"""Shared test data builders.

Before this module, `_make_run` was copy-pasted verbatim into four test
files, `_make_agent` into three, and `_make_interaction` into two. Phase 4
adds three more test modules, so the duplication was about to get worse.

Plain functions rather than pytest fixtures, deliberately: fixtures cannot be
referenced inside @pytest.mark.parametrize, and these are pure data builders
with no setup or teardown. Stateful test doubles (mock_gateway) live in
conftest.py as fixtures, where they do belong.

Every builder takes **overrides passed straight to the model constructor, so
a test that needs one unusual field does not need a new builder.
"""

from __future__ import annotations

from sandbox.model_gateway import BackendResponse
from sandbox.models import Agent, ExperimentRun, Interaction, StanceRecord

# The 7-point Likert scale every existing test uses. Kept here so a test that
# needs a different scale states that intent explicitly via an override.
STANCE_SCALE: list[float] = [1, 2, 3, 4, 5, 6, 7]


def make_run(**overrides) -> ExperimentRun:
    """A valid ExperimentRun. Defaults to M=10/N=3/K=2 against the 4-entry
    test persona pool, with meme injection disabled."""
    base = dict(
        run_id="test_run",
        rq_target="RQ1_RQ2",
        topic="t",
        alpha=0.5,
        M=10,
        N=3,
        K=2,
        trial_number=1,
        language_condition="english",
        model_backend_id="gpt-4o",
        stance_scale=STANCE_SCALE,
        persona_pool_id="test_pool",
        seed=1,
        temperature=0.7,
    )
    base.update(overrides)
    return ExperimentRun(**base)


def make_agent(
    agent_id: str = "agent_0000",
    initial_stance: float = 4.0,
    stance_history: list[StanceRecord] | None = None,
    **overrides,
) -> Agent:
    """agent_id and initial_stance are positional-friendly because
    test_interaction_engine builds populations as make_agent(id, stance)."""
    base = dict(
        agent_id=agent_id,
        run_id="test_run",
        persona="a persona",
        language_condition="english",
        model_backend_id="gpt-4o",
        initial_stance=initial_stance,
        stance_history=stance_history or [],
    )
    base.update(overrides)
    return Agent(**base)


def make_interaction(
    speaker_agent_id: str = "agent_0000",
    turn: int = 1,
    stance_after: float = 5.0,
    interaction_id: str = "int_1",
    reason_text: str = "because",
    **overrides,
) -> Interaction:
    """A successful generated-text interaction.

    For a meme row pass content_type="meme" AND meme_id=..., since
    Interaction.check_meme_id_consistency requires them to agree.
    """
    base = dict(
        interaction_id=interaction_id,
        run_id="test_run",
        turn=turn,
        speaker_agent_id=speaker_agent_id,
        neighbor_agent_ids=["agent_0001"],
        stance_before=4.0,
        stance_after=stance_after,
        reason_text=reason_text,
        content_type="generated_text",
        prompt_token_count=10,
        completion_token_count=10,
        model_backend_id="gpt-4o",
        latency_ms=100,
        api_call_status="success",
        timestamp_utc="2026-01-01T00:00:00Z",
    )
    base.update(overrides)
    return Interaction(**base)


def make_stance_text(stance: float = 5.0, reason: str = "my neighbours shifted me") -> str:
    """A model reply in the exact shape stance_parser.parse_stance expects.

    Every pre-Phase-4 mock returned "ok" or "hello", which parse_stance
    rejects outright, so nothing in the suite produced a reply the orchestrator
    could actually consume. Note the STANCE line comes first and appears
    exactly once: parse_stance strips EVERY occurrence from reason_text, so a
    restated stance would silently vanish from the field RQ2 analyses.
    """
    return f"STANCE: {stance}\n{reason}"


def make_backend_response(
    stance: float = 5.0,
    reason: str = "my neighbours shifted me",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    latency_ms: int = 100,
    text: str | None = None,
) -> BackendResponse:
    """What a mocked ModelGateway.generate() should return.

    Pass text= to override the body entirely, e.g. text="no stance here" to
    drive the StanceParseFailure retry path.
    """
    return BackendResponse(
        text=make_stance_text(stance, reason) if text is None else text,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=latency_ms,
        raw_provider_response={"mocked": True},
    )
