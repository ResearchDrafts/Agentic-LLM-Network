import json
import random

import pytest
from pydantic import ValidationError

from sandbox.agent_manager import AgentManager
from sandbox.checkpoint_manager import CheckpointManager
from tests.factories import make_run as _make_run


def _make_agent_snapshot():
    # M=3/N=1 explicitly: this file's own _make_run used to default to a
    # deliberately tiny population, and the shared factory defaults to M=10.
    # Round-trip assertions hold at either size, so the drift would have been
    # silent; kept small so a failing snapshot diff stays readable.
    run = _make_run(M=3, N=1)
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
