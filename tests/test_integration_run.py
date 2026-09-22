"""End-to-end integration test (full_design_doc.md Sec 7.4).

The first test in the repo that exercises more than one production module
together. Everything before Phase 4 imported exactly one module plus
sandbox.models, so no seam between components had any coverage at all.

Covers the whole path a real run takes: a YAML config on disk, through
config_loader, into a fully wired orchestrator, out to interactions.jsonl,
agents_final.jsonl, run_config.json and a checkpoint. The only thing faked is
the LLM call itself.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from sandbox.agent_manager import AgentManager
from sandbox.checkpoint_manager import CheckpointManager
from sandbox.config_loader import load_run_config
from sandbox.models import Agent, ExperimentRun, Interaction
from sandbox.simulation_orchestrator import build_orchestrator

EXAMPLE_CONFIG = Path("configs/example_run.yaml")

# Sec 7.4's stated shape: small enough to assert exact counts on.
SMALL_RUN = dict(M=5, N=2, K=2)


@pytest.fixture
def config_path(tmp_path):
    """The shipped example config, shrunk to Sec 7.4's size and pointed at
    the test fixtures. Starting from the real file rather than a handwritten
    dict means this test fails if the example config ever stops loading."""

    def _write(**overrides) -> Path:
        raw = yaml.safe_load(EXAMPLE_CONFIG.read_text())
        raw.update(SMALL_RUN)
        raw["run_id"] = "integration_run"
        raw["persona_pool_id"] = "test_pool"
        raw.pop("max_cost_usd", None)
        raw.update(overrides)
        path = tmp_path / "run.yaml"
        path.write_text(yaml.safe_dump(raw))
        return path

    return _write


def _rows(runs_dir, run_id="integration_run") -> list[Interaction]:
    path = runs_dir / run_id / "interactions.jsonl"
    return [
        Interaction.model_validate_json(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


# --- the shipped example config ------------------------------------------


def test_the_shipped_example_config_actually_loads():
    """configs/example_run.yaml is the only worked example a researcher has.
    If it drifts out of sync with the schema it is worse than no example."""
    run = load_run_config(EXAMPLE_CONFIG)
    assert run.M == 100 and run.N == 5 and run.K == 10  # Ohagi's baseline
    assert run.git_commit_hash  # injected at load, never read from the YAML


def test_the_example_configs_commented_meme_block_is_valid_when_enabled(tmp_path):
    """The meme fields ship commented out, so nothing validates them. Enabling
    them exactly as documented must produce a loadable config."""
    raw = yaml.safe_load(EXAMPLE_CONFIG.read_text())
    raw["meme_injection"] = {
        "enabled": True,
        "meme_pool_id": "test_pool",
        "injection_rate": 0.2,
        "injection_schedule": "random",
    }
    path = tmp_path / "memes.yaml"
    path.write_text(yaml.safe_dump(raw))

    run = load_run_config(path)
    assert run.meme_injection.enabled
    assert run.meme_injection.injection_rate == 0.2


# --- Sec 7.4's integration assertions ------------------------------------


async def test_full_run_from_yaml_to_disk(tmp_path, config_path, mock_gateway):
    run = load_run_config(config_path())
    await build_orchestrator(run, mock_gateway, runs_dir=tmp_path).run()
    run_dir = tmp_path / run.run_id

    # Exactly M * K records, meme turns included.
    rows = _rows(tmp_path)
    assert len(rows) == run.M * run.K == 10

    # Every neighbour list is exactly N and never contains the speaker.
    for row in rows:
        assert len(row.neighbor_agent_ids) == run.N
        assert row.speaker_agent_id not in row.neighbor_agent_ids

    # Every record round-trips, and the file is one JSON object per line.
    lines = (run_dir / "interactions.jsonl").read_text().splitlines()
    assert len(lines) == 10
    assert all(json.loads(line) for line in lines)

    # agents_final.jsonl holds one Agent per line.
    finals = [
        Agent.model_validate_json(line)
        for line in (run_dir / "agents_final.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(finals) == run.M

    # run_config.json round-trips and carries terminal lifecycle fields.
    persisted = ExperimentRun.model_validate_json(
        (run_dir / "run_config.json").read_text()
    )
    assert persisted.status == "completed"
    assert persisted.started_at_utc and persisted.completed_at_utc

    # A checkpoint exists at the final turn.
    assert CheckpointManager(tmp_path).load(run.run_id).last_completed_turn == run.K


async def test_meme_enabled_run_produces_valid_meme_rows(
    tmp_path, config_path, mock_gateway
):
    run = load_run_config(
        config_path(
            meme_injection={
                "enabled": True,
                "meme_pool_id": "test_pool",
                "injection_rate": 1.0,
                "injection_schedule": "fixed_turn",
                "fixed_turns": [1],
            }
        )
    )
    await build_orchestrator(run, mock_gateway, runs_dir=tmp_path).run()

    rows = _rows(tmp_path)
    assert len(rows) == 10

    memes = [r for r in rows if r.content_type == "meme"]
    assert len(memes) == 5  # every agent, turn 1 only
    for row in memes:
        assert row.meme_id is not None
        assert row.prompt_token_count == 0
        assert row.completion_token_count == 0
        assert row.latency_ms == 0

    # Turn 2 is ordinary generated text, and those agents were shown the
    # turn-1 memes as neighbour content.
    assert all(r.content_type == "generated_text" for r in rows if r.turn == 2)
    assert any("Caption:" in c.args[0] for c in mock_gateway.generate.await_args_list)


async def test_rerunning_a_completed_run_is_a_no_op(tmp_path, config_path, mock_gateway):
    """Sec 7.4: resume against a finished run_id must not error and must not
    duplicate work, since start_turn > K leaves the loop empty."""
    path = config_path()
    await build_orchestrator(
        load_run_config(path), mock_gateway, runs_dir=tmp_path
    ).run()
    rows_before = len(_rows(tmp_path))
    calls_before = mock_gateway.generate.await_count

    await build_orchestrator(
        load_run_config(path), mock_gateway, runs_dir=tmp_path
    ).run()

    assert len(_rows(tmp_path)) == rows_before
    assert mock_gateway.generate.await_count == calls_before


async def test_the_same_seed_reproduces_the_run_exactly(
    tmp_path, config_path, mock_gateway
):
    """The property every RQ in this project rests on. Fix M also depends on
    it: reusing a seed across language arms is what holds neighbour sampling
    and meme injection constant while language varies."""
    first, second = tmp_path / "a", tmp_path / "b"
    await build_orchestrator(
        load_run_config(config_path(seed=7)), mock_gateway, runs_dir=first
    ).run()
    await build_orchestrator(
        load_run_config(config_path(seed=7)), mock_gateway, runs_dir=second
    ).run()

    def signature(base):
        return [
            (r.turn, r.speaker_agent_id, tuple(r.neighbor_agent_ids), r.stance_after)
            for r in _rows(base)
        ]

    assert signature(first) == signature(second)


async def test_agent_state_and_the_interaction_log_agree(
    tmp_path, config_path, mock_gateway
):
    """agents_final.jsonl and interactions.jsonl are written by different
    paths from different objects. Post-hoc analysis reads the log; anything
    loading Agent objects reads the other. They must tell the same story."""
    run = load_run_config(config_path())
    await build_orchestrator(run, mock_gateway, runs_dir=tmp_path).run()

    rows = _rows(tmp_path)
    finals = {
        a.agent_id: a
        for a in (
            Agent.model_validate_json(line)
            for line in (tmp_path / run.run_id / "agents_final.jsonl")
            .read_text()
            .splitlines()
            if line.strip()
        )
    }

    for agent_id, agent in finals.items():
        logged = sorted(
            (r for r in rows if r.speaker_agent_id == agent_id), key=lambda r: r.turn
        )
        assert [s.turn for s in agent.stance_history] == [r.turn for r in logged]
        assert agent.current_stance() == logged[-1].stance_after
        # Fix A: the window never exceeds 5, whatever K is.
        assert len(agent.memory_window) <= 5


async def test_a_resumed_run_matches_an_uninterrupted_one(
    tmp_path, config_path, mock_gateway
):
    """Checkpoint and resume exist so a failed run is not restarted from
    scratch, which would re-spend the budget. That is only worth anything if
    the resumed output is the output the run would have produced anyway."""
    from tests.factories import make_backend_response

    clean = tmp_path / "clean"
    await build_orchestrator(
        load_run_config(config_path(seed=11)), mock_gateway, runs_dir=clean
    ).run()

    crashed = tmp_path / "crashed"
    mock_gateway.generate.side_effect = [make_backend_response(stance=5.0)] * 5 + [
        RuntimeError("boom")
    ] * 50
    with pytest.raises(RuntimeError):
        await build_orchestrator(
            load_run_config(config_path(seed=11)), mock_gateway, runs_dir=crashed
        ).run()

    mock_gateway.generate.side_effect = None
    mock_gateway.generate.return_value = make_backend_response(stance=5.0)
    await build_orchestrator(
        load_run_config(config_path(seed=11)), mock_gateway, runs_dir=crashed
    ).run()

    def signature(base):
        return [
            (r.turn, r.speaker_agent_id, tuple(r.neighbor_agent_ids), r.stance_after)
            for r in _rows(base)
        ]

    assert signature(crashed) == signature(clean)


# --- the shape post-hoc analysis will consume ----------------------------


async def test_the_log_is_ready_for_the_phase_5_analysis_modules(
    tmp_path, config_path, mock_gateway
):
    """Sec 7.4 wants the post-hoc path validated end to end, but none of those
    modules exist yet (Phase 5). What can be checked now is that
    interactions.jsonl carries the columns they are specified to read, with
    usable dtypes, since that schema is what unblocks building them.
    """
    pd = pytest.importorskip("pandas")

    run = load_run_config(config_path())
    await build_orchestrator(run, mock_gateway, runs_dir=tmp_path).run()

    df = pd.read_json(tmp_path / run.run_id / "interactions.jsonl", lines=True)

    # stance_regression.py fits stance_after ~ stance_before + neighbour mean.
    for column in ("stance_before", "stance_after", "prompt_token_count", "turn"):
        assert pd.api.types.is_numeric_dtype(df[column]), column

    # coherence_scorer.py and sentiment_proxy.py must filter on this first
    # (Fix N), so it has to be present and exactly two-valued.
    assert set(df["content_type"]) <= {"generated_text", "meme"}

    # Every analysis drops failed rows via this column.
    assert set(df["api_call_status"]) <= {
        "success",
        "retried_success",
        "failed_logged_null",
    }

    # neighbour ids survive as real lists, not stringified ones: the
    # regression joins on them to compute each row's neighbour mean.
    assert all(isinstance(v, list) for v in df["neighbor_agent_ids"])
    assert df["neighbor_agent_ids"].map(len).eq(run.N).all()


async def test_prompt_token_count_is_populated_for_the_fix_a_diagnostic(
    tmp_path, config_path, mock_gateway
):
    """prompt_token_count is the field that turns "memory was capped at 5
    turns" into a measured result. If it were ever left at 0 for generated
    text, the RQ1 truncation-symmetry check could not be reported at all."""
    run = load_run_config(config_path())
    await build_orchestrator(run, mock_gateway, runs_dir=tmp_path).run()

    generated = [r for r in _rows(tmp_path) if r.content_type == "generated_text"]
    assert generated
    assert all(r.prompt_token_count > 0 for r in generated)
