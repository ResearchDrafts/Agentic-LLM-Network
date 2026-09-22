import asyncio

import pytest

from sandbox.logging_writer import LoggingWriter
from sandbox.models import Agent, ExperimentRun, Interaction
from tests.factories import make_agent as _make_agent, make_interaction as _make_interaction, make_run as _make_run


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
