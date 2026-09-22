import json
import random

import pytest
from pydantic import ValidationError

from sandbox.agent_manager import AgentManager
from sandbox.checkpoint_manager import CheckpointManager
from sandbox.models import ExperimentRun


def _make_run(**overrides) -> ExperimentRun:
    base = dict(
        run_id="test_run",
        rq_target="RQ1_RQ2",
        topic="t",
        alpha=0.5,
        M=3,
        N=1,
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


def _make_agent_snapshot():
    run = _make_run()
    manager = AgentManager(run, random.Random(run.seed))
    manager.initialize_population()
    return manager.snapshot()


def test_load_with_no_checkpoint_directory_returns_none(tmp_path):
    manager = CheckpointManager(runs_dir=tmp_path)
    assert manager.load("never_saved_run") is None


def test_save_then_load_round_trips_run_id_turn_and_agent_snapshot(tmp_path):
    manager = CheckpointManager(runs_dir=tmp_path)
    snapshot = _make_agent_snapshot()

    manager.save("test_run", turn=3, agent_snapshot=snapshot)
    state = manager.load("test_run")

    assert state is not None
    assert state.run_id == "test_run"
    assert state.last_completed_turn == 3
    assert [a.model_dump() for a in state.agent_snapshot] == [a.model_dump() for a in snapshot]


def test_two_sequential_saves_load_returns_only_the_latest(tmp_path):
    manager = CheckpointManager(runs_dir=tmp_path)
    snapshot = _make_agent_snapshot()

    manager.save("test_run", turn=1, agent_snapshot=snapshot)
    manager.save("test_run", turn=2, agent_snapshot=snapshot)
    state = manager.load("test_run")

    assert state.last_completed_turn == 2


def test_corrupted_checkpoint_file_raises_uncaught_on_load(tmp_path):
    checkpoint_dir = tmp_path / "test_run" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "latest.json").write_text("{not valid json")

    manager = CheckpointManager(runs_dir=tmp_path)
    with pytest.raises(ValidationError):
        manager.load("test_run")


def test_save_creates_checkpoints_directory_if_missing(tmp_path):
    manager = CheckpointManager(runs_dir=tmp_path)
    assert not (tmp_path / "test_run").exists()

    manager.save("test_run", turn=1, agent_snapshot=_make_agent_snapshot())

    assert (tmp_path / "test_run" / "checkpoints" / "latest.json").exists()


def test_save_leaves_no_tmp_file_behind_after_success(tmp_path):
    manager = CheckpointManager(runs_dir=tmp_path)
    manager.save("test_run", turn=1, agent_snapshot=_make_agent_snapshot())

    checkpoint_dir = tmp_path / "test_run" / "checkpoints"
    assert (checkpoint_dir / "latest.json").exists()
    assert not (checkpoint_dir / "latest.json.tmp").exists()


def test_saved_file_is_valid_json(tmp_path):
    manager = CheckpointManager(runs_dir=tmp_path)
    manager.save("test_run", turn=1, agent_snapshot=_make_agent_snapshot())

    raw = (tmp_path / "test_run" / "checkpoints" / "latest.json").read_text()
    json.loads(raw)  # must not raise
