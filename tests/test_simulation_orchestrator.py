"""Tests for simulation_orchestrator.py, organised around phase4.md Section 2.

The gateway is always mocked. Nothing here makes a network call.
"""

from __future__ import annotations

import json

import pytest

from sandbox.cost_tracker import CostCeilingExceeded
from sandbox.model_gateway import FatalGatewayError
from sandbox.models import ExperimentRun, Interaction, MemeInjectionConfig
from sandbox.simulation_orchestrator import (
    build_orchestrator,
    _current_stance_for_logging,
)
from tests.factories import make_agent, make_backend_response, make_run


def _run(**overrides) -> ExperimentRun:
    base = dict(M=5, N=2, K=2, run_id="orch_run", persona_pool_id="test_pool")
    base.update(overrides)
    return make_run(**base)


def _meme_config(**overrides) -> MemeInjectionConfig:
    base = dict(enabled=True, meme_pool_id="test_pool", injection_rate=1.0)
    base.update(overrides)
    return MemeInjectionConfig(**base)


def _interactions(tmp_path, run_id="orch_run") -> list[Interaction]:
    path = tmp_path / run_id / "interactions.jsonl"
    return [
        Interaction.model_validate_json(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def _run_config(tmp_path, run_id="orch_run") -> dict:
    return json.loads((tmp_path / run_id / "run_config.json").read_text())


# --- the happy path -------------------------------------------------------


async def test_full_run_writes_one_interaction_per_agent_turn(tmp_path, mock_gateway):
    run = _run()
    await build_orchestrator(run, mock_gateway, runs_dir=tmp_path).run()

    rows = _interactions(tmp_path)
    assert len(rows) == run.M * run.K


async def test_every_row_has_exactly_n_neighbours_and_excludes_the_speaker(
    tmp_path, mock_gateway
):
    run = _run()
    await build_orchestrator(run, mock_gateway, runs_dir=tmp_path).run()

    for row in _interactions(tmp_path):
        assert len(row.neighbor_agent_ids) == run.N
        assert row.speaker_agent_id not in row.neighbor_agent_ids


async def test_agents_final_and_checkpoint_are_written(tmp_path, mock_gateway):
    run = _run()
    orchestrator = build_orchestrator(run, mock_gateway, runs_dir=tmp_path)
    await orchestrator.run()

    assert (tmp_path / run.run_id / "agents_final.jsonl").exists()
    assert (tmp_path / run.run_id / "checkpoints" / "latest.json").exists()


async def test_orchestrator_creates_the_run_directory(tmp_path, mock_gateway):
    """LoggingWriter deliberately does not mkdir (phase3.md:166 resolved in
    favour of the caller), so the orchestrator must, or write_run_config
    raises FileNotFoundError on a fresh run."""
    assert not (tmp_path / "orch_run").exists()
    await build_orchestrator(_run(), mock_gateway, runs_dir=tmp_path).run()
    assert (tmp_path / "orch_run" / "run_config.json").exists()


# --- Q4.1 / Fix F: neighbour posts come from the frozen snapshot ---------


async def test_turn_one_sends_no_neighbour_posts(tmp_path, mock_gateway):
    """Nobody has spoken at turn 1, so no prompt may contain a neighbour
    block. Guards against reading turn-t output during turn t."""
    await build_orchestrator(_run(K=1), mock_gateway, runs_dir=tmp_path).run()

    for call in mock_gateway.generate.await_args_list:
        assert "What others in your feed" not in call.args[0]


async def test_later_turns_do_send_neighbour_posts(tmp_path, mock_gateway):
    await build_orchestrator(_run(K=2), mock_gateway, runs_dir=tmp_path).run()

    prompts = [c.args[0] for c in mock_gateway.generate.await_args_list]
    assert any("What others in your feed" in p for p in prompts)


async def test_identical_seeds_produce_identical_runs(tmp_path, mock_gateway):
    """Reproducibility is the whole point of the seeded-RNG discipline, and
    nothing else in the suite exercises SeedManager against its real
    consumers."""
    first = tmp_path / "a"
    second = tmp_path / "b"
    await build_orchestrator(_run(seed=99), mock_gateway, runs_dir=first).run()
    await build_orchestrator(_run(seed=99), mock_gateway, runs_dir=second).run()

    def neighbours(base):
        return [
            (r.turn, r.speaker_agent_id, tuple(r.neighbor_agent_ids))
            for r in _interactions(base)
        ]

    assert neighbours(first) == neighbours(second)


async def test_different_seeds_produce_different_neighbour_draws(tmp_path, mock_gateway):
    await build_orchestrator(_run(seed=1), mock_gateway, runs_dir=tmp_path / "a").run()
    await build_orchestrator(_run(seed=2), mock_gateway, runs_dir=tmp_path / "b").run()

    def neighbours(base):
        return [tuple(r.neighbor_agent_ids) for r in _interactions(base)]

    assert neighbours(tmp_path / "a") != neighbours(tmp_path / "b")


# --- Q4.8 / 2g: the vision path is actually wired -----------------------


async def test_meme_turns_cost_no_llm_call_and_take_the_dataset_stance(
    tmp_path, mock_gateway
):
    run = _run(K=1, meme_injection=_meme_config())
    await build_orchestrator(run, mock_gateway, runs_dir=tmp_path).run()

    rows = _interactions(tmp_path)
    assert all(r.content_type == "meme" for r in rows)
    assert all(r.meme_id is not None for r in rows)
    assert all(r.prompt_token_count == 0 and r.completion_token_count == 0 for r in rows)
    assert mock_gateway.generate.await_count == 0


async def test_a_neighbours_meme_reaches_a_vision_backend_as_an_image(
    tmp_path, mock_gateway
):
    """Everyone posts a meme on turn 1, so every turn-2 prompt carries one."""
    run = _run(K=2, meme_injection=_meme_config(injection_schedule="fixed_turn", fixed_turns={1}))
    await build_orchestrator(run, mock_gateway, runs_dir=tmp_path).run()

    images = [c.kwargs.get("image") for c in mock_gateway.generate.await_args_list]
    assert any(img is not None for img in images)


async def test_text_only_backend_records_used_vision_fallback(
    tmp_path, text_only_gateway
):
    """Without this wiring used_vision_fallback stays False in every run ever
    produced, and vision_fallback.py remains unreachable dead code."""
    run = _run(K=2, meme_injection=_meme_config(injection_schedule="fixed_turn", fixed_turns={1}))
    await build_orchestrator(run, text_only_gateway, runs_dir=tmp_path).run()

    turn_two = [r for r in _interactions(tmp_path) if r.turn == 2]
    assert any(r.used_vision_fallback for r in turn_two)
    assert all(c.kwargs.get("image") is None for c in text_only_gateway.generate.await_args_list)


# --- 2b: the exception taxonomy ------------------------------------------


async def test_unparseable_reply_retries_then_logs_failed_logged_null(
    tmp_path, mock_gateway
):
    mock_gateway.generate.return_value = make_backend_response(text="no stance here")
    await build_orchestrator(_run(K=1), mock_gateway, runs_dir=tmp_path).run()

    rows = _interactions(tmp_path)
    assert all(r.api_call_status == "failed_logged_null" for r in rows)
    assert all(not r.is_valid_for_analysis for r in rows)
    # 3 attempts per agent, per Fix I.
    assert mock_gateway.generate.await_count == 5 * 3


async def test_a_recovered_parse_failure_is_marked_retried_success(
    tmp_path, mock_gateway
):
    mock_gateway.generate.side_effect = [
        make_backend_response(text="junk"),
        make_backend_response(stance=6.0),
    ] + [make_backend_response(stance=5.0)] * 20

    await build_orchestrator(_run(M=2, N=1, K=1), mock_gateway, runs_dir=tmp_path).run()

    statuses = {r.api_call_status for r in _interactions(tmp_path)}
    assert "retried_success" in statuses


async def test_failed_agent_keeps_its_previous_stance(tmp_path, mock_gateway):
    """Fix I: never fabricate a placeholder stance. stance_after repeats
    stance_before, and no StanceRecord is appended."""
    mock_gateway.generate.return_value = make_backend_response(text="no stance here")
    await build_orchestrator(_run(K=1), mock_gateway, runs_dir=tmp_path).run()

    for row in _interactions(tmp_path):
        assert row.stance_after == row.stance_before


async def test_fatal_gateway_error_is_absorbed_per_agent_not_fatal_to_the_run(
    tmp_path, mock_gateway
):
    mock_gateway.generate.side_effect = FatalGatewayError("bad key")
    await build_orchestrator(_run(K=1), mock_gateway, runs_dir=tmp_path).run()

    rows = _interactions(tmp_path)
    assert len(rows) == 5
    assert all(r.api_call_status == "failed_logged_null" for r in rows)


async def test_fatal_gateway_error_is_not_retried(tmp_path, mock_gateway):
    mock_gateway.generate.side_effect = FatalGatewayError("bad key")
    await build_orchestrator(_run(K=1), mock_gateway, runs_dir=tmp_path).run()
    assert mock_gateway.generate.await_count == 5  # one attempt each, not three


async def test_cost_ceiling_aborts_the_whole_run(tmp_path, mock_gateway):
    """The contradiction phase4.md 2b resolves. Sec 3.9's sample `continue`s
    past every gathered exception, which would silently swallow this and let
    the run continue spending."""
    mock_gateway.generate.side_effect = CostCeilingExceeded(10.0, 5.0)

    with pytest.raises(CostCeilingExceeded):
        await build_orchestrator(_run(K=2), mock_gateway, runs_dir=tmp_path).run()


async def test_an_unexpected_exception_aborts_the_whole_run(tmp_path, mock_gateway):
    """Only the two designed failure modes get soft handling; anything else
    is a bug and must surface rather than becoming a failed_logged_null row."""
    mock_gateway.generate.side_effect = RuntimeError("something nobody planned for")

    with pytest.raises(RuntimeError, match="nobody planned for"):
        await build_orchestrator(_run(K=2), mock_gateway, runs_dir=tmp_path).run()


async def test_a_fatal_abort_records_failed_partial_status(tmp_path, mock_gateway):
    """A run that dies must not be left looking like it is still running."""
    mock_gateway.generate.side_effect = [make_backend_response(stance=5.0)] * 5 + [
        RuntimeError("boom")
    ] * 20

    with pytest.raises(RuntimeError):
        await build_orchestrator(_run(K=3), mock_gateway, runs_dir=tmp_path).run()

    assert _run_config(tmp_path)["status"] == "failed_partial"


# --- 2c: the run-lifecycle fields nothing used to read -------------------


async def test_completed_run_persists_its_lifecycle_fields(tmp_path, mock_gateway):
    await build_orchestrator(_run(), mock_gateway, runs_dir=tmp_path).run()

    config = _run_config(tmp_path)
    assert config["status"] == "completed"
    assert config["started_at_utc"]
    assert config["completed_at_utc"]


async def test_checkpoint_every_n_turns_is_honoured(tmp_path, mock_gateway):
    """Sec 3.9's sample checkpoints unconditionally, ignoring the field."""
    from sandbox.checkpoint_manager import CheckpointManager

    run = _run(K=4, checkpoint_every_n_turns=2)
    await build_orchestrator(run, mock_gateway, runs_dir=tmp_path).run()

    assert CheckpointManager(tmp_path).load(run.run_id).last_completed_turn == 4


async def test_final_turn_is_checkpointed_even_when_the_interval_skips_it(
    tmp_path, mock_gateway
):
    """K=3 with checkpoint_every_n_turns=2 checkpoints turn 2 and would
    otherwise never checkpoint turn 3, so a completed run would resume from
    the wrong place."""
    from sandbox.checkpoint_manager import CheckpointManager

    run = _run(K=3, checkpoint_every_n_turns=2)
    await build_orchestrator(run, mock_gateway, runs_dir=tmp_path).run()

    assert CheckpointManager(tmp_path).load(run.run_id).last_completed_turn == 3


# --- resume ---------------------------------------------------------------


async def test_rerunning_a_completed_run_is_a_no_op(tmp_path, mock_gateway):
    run = _run()
    await build_orchestrator(run, mock_gateway, runs_dir=tmp_path).run()
    rows_before = len(_interactions(tmp_path))
    calls_before = mock_gateway.generate.await_count

    await build_orchestrator(_run(), mock_gateway, runs_dir=tmp_path).run()

    assert len(_interactions(tmp_path)) == rows_before
    assert mock_gateway.generate.await_count == calls_before


async def test_resume_continues_from_the_checkpoint(tmp_path, mock_gateway):
    """Crash after turn 1, then resume and finish."""
    mock_gateway.generate.side_effect = [make_backend_response(stance=5.0)] * 5 + [
        RuntimeError("boom")
    ] * 50

    with pytest.raises(RuntimeError):
        await build_orchestrator(_run(K=3), mock_gateway, runs_dir=tmp_path).run()
    assert {r.turn for r in _interactions(tmp_path)} == {1}

    mock_gateway.generate.side_effect = None
    mock_gateway.generate.return_value = make_backend_response(stance=5.0)
    await build_orchestrator(_run(K=3), mock_gateway, runs_dir=tmp_path).run()

    assert {r.turn for r in _interactions(tmp_path)} == {1, 2, 3}


async def test_resume_rebuilds_neighbour_posts_including_meme_content(
    tmp_path, mock_gateway
):
    """CheckpointState stores only agent_snapshot, and StanceRecord has no
    content_type, so without rebuilding from interactions.jsonl a resumed run
    would render every past meme as ordinary text and silently change the
    stimulus mid-run.
    """
    memes = _meme_config(injection_schedule="fixed_turn", fixed_turns={1})

    # Turn 1 is all memes, then the run dies before turn 2.
    mock_gateway.generate.side_effect = RuntimeError("boom")
    with pytest.raises(RuntimeError):
        await build_orchestrator(
            _run(K=2, meme_injection=memes), mock_gateway, runs_dir=tmp_path
        ).run()
    assert all(r.content_type == "meme" for r in _interactions(tmp_path))

    # Resuming must recover those meme posts, not just agent stances.
    resumed = build_orchestrator(
        _run(K=2, meme_injection=memes), mock_gateway, runs_dir=tmp_path
    )
    rebuilt = resumed._rebuild_last_posts()
    assert rebuilt
    assert all(post.content_type == "meme" for post in rebuilt.values())
    assert all(post.meme_id is not None for post in rebuilt.values())

    # And the resumed turn-2 prompts must actually carry the meme framing.
    mock_gateway.generate.side_effect = None
    mock_gateway.generate.return_value = make_backend_response(stance=5.0)
    await resumed.run()

    turn_two_prompts = [c.args[0] for c in mock_gateway.generate.await_args_list]
    assert any("Caption:" in p for p in turn_two_prompts)


# --- 2f: stance_before must match what neighbour sampling used -----------


def test_current_stance_for_logging_matches_interaction_engine():
    """If these drift, stance_before in the log is not the value sampling
    actually used, and every regression fitted on that column is wrong."""
    from sandbox.interaction_engine import _current_stance
    from sandbox.models import StanceRecord

    history = [
        StanceRecord(turn=1, stance_value=6.0, reason_text="r", interaction_id="i1")
    ]
    for agent in (make_agent(initial_stance=4.0), make_agent(initial_stance=4.0, stance_history=history)):
        for turn in (1, 2, 3):
            assert _current_stance_for_logging(agent, turn) == _current_stance(agent, turn)


async def test_stance_before_tracks_the_previous_turns_stance_after(
    tmp_path, mock_gateway
):
    mock_gateway.generate.return_value = make_backend_response(stance=6.0)
    await build_orchestrator(_run(K=2), mock_gateway, runs_dir=tmp_path).run()

    rows = _interactions(tmp_path)
    by_agent: dict[str, list[Interaction]] = {}
    for row in rows:
        by_agent.setdefault(row.speaker_agent_id, []).append(row)

    for turns in by_agent.values():
        turns.sort(key=lambda r: r.turn)
        assert turns[1].stance_before == turns[0].stance_after


async def test_no_agent_sees_another_agents_same_turn_output(tmp_path, mock_gateway):
    """Fix F, tested against what it actually guarantees.

    Every agent's turn-t prompt must show neighbours as of the end of turn
    t-1. If step 6 were interleaved into dispatch rather than running after
    asyncio.gather, agents dispatched later in a turn would see earlier
    agents' turn-t stances, and the run would stop being reproducible.

    Each call returns a distinct stance, so turn-1 posts carry stances 1-5
    and turn-2 posts carry 6-10. A turn-2 prompt may therefore mention
    positions 1-5 only; a 6 or higher means someone read same-turn output.
    """
    counter = iter(range(1, 200))
    mock_gateway.generate.side_effect = lambda *a, **k: make_backend_response(
        stance=5.0, reason=f"turnmarker-{next(counter)}"
    )

    await build_orchestrator(_run(M=5, N=2, K=3), mock_gateway, runs_dir=tmp_path).run()

    rows = _interactions(tmp_path)
    markers_by_turn = {
        turn: {r.reason_text for r in rows if r.turn == turn and r.reason_text}
        for turn in (1, 2, 3)
    }

    prompts_by_call = [c.args[0] for c in mock_gateway.generate.await_args_list]
    # Turn t prompts are the t-th block of M calls (no retries occur here).
    for turn in (2, 3):
        block = prompts_by_call[(turn - 1) * 5 : turn * 5]
        for prompt in block:
            for later_marker in markers_by_turn[turn]:
                assert later_marker not in prompt, (
                    f"turn {turn} prompt contained same-turn output {later_marker!r}"
                )
