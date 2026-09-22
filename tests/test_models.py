"""Tests for sandbox/models.py's own validation rules.

Phase 4 starter. full_design_doc.md Sec 4's constraints were previously
covered only incidentally, through the modules that happen to construct
these models; the cross-field validators and the Fix A memory cap had no
direct coverage at all. phase4.md Section 3 tracks the remaining gaps
(Interaction.check_meme_id_consistency's meme branch,
is_valid_for_analysis, MemeInjectionConfig.check_enabled_requirements).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sandbox.models import Agent


def _make_agent(memory_window: list[str] | None = None) -> Agent:
    return Agent(
        agent_id="agent_0000",
        run_id="test_run",
        persona="a persona",
        language_condition="english",
        model_backend_id="gpt-4o",
        initial_stance=4.0,
        memory_window=memory_window if memory_window is not None else [],
    )


# --- Fix A: the 5-turn memory window (phase4.md D7) ---------------------


def test_construction_rejects_memory_window_over_cap():
    with pytest.raises(ValidationError):
        _make_agent([f"i{i}" for i in range(6)])


def test_assignment_rejects_memory_window_over_cap():
    """The D7 regression.

    Without model_config = ConfigDict(validate_assignment=True), Pydantic v2
    validates only at construction, so this assignment silently succeeded and
    Fix A's cap was enforced solely by agent_manager.py's [-5:] slice -- a
    convention, not the schema-level guarantee full_design_doc.md Sec 4.1
    claims it is.
    """
    agent = _make_agent(["i1", "i2"])
    with pytest.raises(ValidationError):
        agent.memory_window = [f"i{i}" for i in range(8)]


def test_assignment_accepts_memory_window_at_cap():
    """Guards against over-correcting: exactly 5 must remain valid, since
    that is what agent_manager.py assigns on every apply_interaction()."""
    agent = _make_agent()
    agent.memory_window = [f"i{i}" for i in range(5)]
    assert len(agent.memory_window) == 5


def test_inplace_append_bypasses_the_cap():
    """Documents a known limit rather than asserting desired behaviour.

    Pydantic cannot observe in-place list mutation, so validate_assignment
    does not close this path. Fix A is therefore re-enforced at the point of
    use: prompt_builder.py slices [-5:] when rendering memory into a prompt.
    If a future Pydantic version starts catching this, update the test -- do
    not treat the change as a regression.
    """
    agent = _make_agent([f"i{i}" for i in range(5)])
    agent.memory_window.append("overflow")
    assert len(agent.memory_window) == 6


def test_assignment_still_validates_other_fields():
    """validate_assignment applies to every field, not just memory_window."""
    agent = _make_agent()
    with pytest.raises(ValidationError):
        agent.language_condition = "french"


def test_current_stance_falls_back_to_initial_when_history_empty():
    assert _make_agent().current_stance() == 4.0


# --- phase4 audit D10: unknown config keys must not vanish ---------------


def _run_kwargs(**overrides):
    base = dict(
        run_id="r", rq_target="RQ1_RQ2", topic="t", alpha=0.5, M=10, N=3, K=2,
        trial_number=1, language_condition="english", model_backend_id="gpt-4o",
        stance_scale=[1, 2, 3, 4, 5, 6, 7], persona_pool_id="test_pool",
        seed=1, temperature=0.7,
    )
    base.update(overrides)
    return base


def test_unknown_experiment_run_key_is_rejected():
    """The dangerous case: `meme_injections` (plural) used to load fine and
    leave meme injection off, so RQ3's meme-present arm silently ran as a
    second meme-absent arm and returned a clean null result."""
    from sandbox.models import ExperimentRun

    with pytest.raises(ValidationError):
        ExperimentRun(**_run_kwargs(meme_injections={"enabled": True}))


def test_unknown_meme_injection_key_is_rejected():
    from sandbox.models import MemeInjectionConfig

    with pytest.raises(ValidationError):
        MemeInjectionConfig(enabled=False, injection_rat=0.5)


def test_known_optional_keys_still_accepted():
    """Guards against over-correcting: extra="forbid" must not reject the
    legitimate optional fields."""
    from sandbox.models import ExperimentRun

    run = ExperimentRun(**_run_kwargs(
        max_cost_usd=5.0, rate_limits={"groq": 30}, checkpoint_every_n_turns=2,
        stance_low_label="opposed", stance_high_label="in favour",
        api_base="http://localhost:8000/v1",
    ))
    assert run.api_base == "http://localhost:8000/v1"
    assert run.max_cost_usd == 5.0


# --- phase4 audit D11: stance_scale needs distinct values ----------------


def test_stance_scale_with_duplicate_values_is_rejected():
    """[3, 3] passed Field(min_length=2), then min == max so only 3.0 ever
    validated and the entire population ended up failed_logged_null."""
    from sandbox.models import ExperimentRun

    with pytest.raises(ValidationError, match="distinct"):
        ExperimentRun(**_run_kwargs(stance_scale=[3.0, 3.0]))


def test_two_distinct_values_is_enough():
    from sandbox.models import ExperimentRun

    assert ExperimentRun(**_run_kwargs(stance_scale=[1.0, 7.0])).stance_scale == [1.0, 7.0]
