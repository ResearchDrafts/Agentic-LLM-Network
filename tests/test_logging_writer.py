import asyncio

import pytest

from sandbox.logging_writer import LoggingWriter
from sandbox.models import Agent, ExperimentRun, Interaction


def _make_interaction(turn: int, interaction_id: str) -> Interaction:
    return Interaction(
        interaction_id=interaction_id,
        run_id="test_run",
        turn=turn,
        speaker_agent_id="agent_0000",
        neighbor_agent_ids=["agent_0001"],
        stance_before=4.0,
        stance_after=5.0,
        reason_text="because",
        content_type="generated_text",
        prompt_token_count=10,
        completion_token_count=10,
        model_backend_id="gpt-4o",
        latency_ms=100,
        api_call_status="success",
        timestamp_utc="2026-01-01T00:00:00Z",
    )


def _make_run(**overrides) -> ExperimentRun:
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
        stance_scale=[1, 2, 3, 4, 5, 6, 7],
        persona_pool_id="test_pool",
        seed=1,
        temperature=0.7,
    )
    base.update(overrides)
    return ExperimentRun(**base)


def _make_agent(agent_id: str = "agent_0000") -> Agent:
    return Agent(
        agent_id=agent_id,
        run_id="test_run",
        persona="a persona",
        language_condition="english",
        model_backend_id="gpt-4o",
        initial_stance=4.0,
    )


async def test_concurrent_write_interaction_produces_n_valid_untorn_lines(tmp_path):
    writer = LoggingWriter(tmp_path)
    n = 50
    interactions = [_make_interaction(turn=1, interaction_id=f"int_{i}") for i in range(n)]

    await asyncio.gather(*(writer.write_interaction(i) for i in interactions))

    lines = (tmp_path / "interactions.jsonl").read_text().splitlines()
    assert len(lines) == n
    parsed = [Interaction.model_validate_json(line) for line in lines]
    assert {p.interaction_id for p in parsed} == {i.interaction_id for i in interactions}


async def test_write_run_config_round_trips_via_experiment_run_model(tmp_path):
    writer = LoggingWriter(tmp_path)
    run = _make_run()

    writer.write_run_config(run)

    loaded = ExperimentRun.model_validate_json((tmp_path / "run_config.json").read_text())
    assert loaded == run


def test_write_agents_final_writes_n_lines_round_tripping_via_agent_model(tmp_path):
    writer = LoggingWriter(tmp_path)
    agents = [_make_agent(f"agent_{i:04d}") for i in range(3)]

    writer.write_agents_final(agents)

    lines = (tmp_path / "agents_final.jsonl").read_text().splitlines()
    assert len(lines) == 3
    parsed = [Agent.model_validate_json(line) for line in lines]
    assert [a.agent_id for a in parsed] == [a.agent_id for a in agents]


def test_write_agents_final_overwrites_not_appends_on_second_call(tmp_path):
    writer = LoggingWriter(tmp_path)
    writer.write_agents_final([_make_agent("agent_0000"), _make_agent("agent_0001")])

    writer.write_agents_final([_make_agent("agent_0002")])

    lines = (tmp_path / "agents_final.jsonl").read_text().splitlines()
    assert len(lines) == 1
    assert Agent.model_validate_json(lines[0]).agent_id == "agent_0002"


def test_write_run_config_against_nonexistent_run_dir_raises_file_not_found(tmp_path):
    writer = LoggingWriter(tmp_path / "does_not_exist")

    with pytest.raises(FileNotFoundError):
        writer.write_run_config(_make_run())
