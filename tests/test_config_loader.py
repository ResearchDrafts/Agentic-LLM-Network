from pathlib import Path

import pytest

from sandbox.config_loader import ConfigLoadError, load_run_config


def test_valid_config_produces_experiment_run_with_git_hash(valid_config_dict, write_config):
    path = write_config(valid_config_dict)
    run = load_run_config(path)
    assert run.run_id == "test_run_0001"
    assert run.git_commit_hash  # populated automatically, never from YAML
    assert len(run.git_commit_hash) == 40  # full sha


def test_git_commit_hash_in_yaml_is_ignored(valid_config_dict, write_config):
    valid_config_dict["git_commit_hash"] = "deadbeef" * 5
    path = write_config(valid_config_dict)
    run = load_run_config(path)
    assert run.git_commit_hash != "deadbeef" * 5


def test_missing_file_raises_config_load_error(tmp_path):
    with pytest.raises(ConfigLoadError):
        load_run_config(tmp_path / "does_not_exist.yaml")


def test_malformed_yaml_raises_config_load_error(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("topic: [unclosed")
    with pytest.raises(ConfigLoadError):
        load_run_config(path)


def test_missing_required_field_names_it(valid_config_dict, write_config):
    del valid_config_dict["topic"]
    path = write_config(valid_config_dict)
    with pytest.raises(ConfigLoadError) as exc_info:
        load_run_config(path)
    assert any("topic" in e for e in exc_info.value.errors)


def test_n_greater_or_equal_m_rejected(valid_config_dict, write_config):
    valid_config_dict["M"] = 5
    valid_config_dict["N"] = 5
    path = write_config(valid_config_dict)
    with pytest.raises(ConfigLoadError):
        load_run_config(path)


def test_meme_enabled_without_pool_id_rejected_before_filesystem_check(valid_config_dict, write_config):
    valid_config_dict["meme_injection"] = {"enabled": True}
    path = write_config(valid_config_dict)
    with pytest.raises(ConfigLoadError):
        load_run_config(path)


def test_meme_enabled_with_nonexistent_pool_rejected_naming_path(valid_config_dict, write_config):
    valid_config_dict["meme_injection"] = {"enabled": True, "meme_pool_id": "does_not_exist_pool"}
    path = write_config(valid_config_dict)
    with pytest.raises(ConfigLoadError) as exc_info:
        load_run_config(path)
    assert any("data/memes" in e for e in exc_info.value.errors)


def test_meme_enabled_with_real_pool_succeeds(valid_config_dict, write_config):
    valid_config_dict["meme_injection"] = {"enabled": True, "meme_pool_id": "test_pool"}
    path = write_config(valid_config_dict)
    run = load_run_config(path)
    assert run.meme_injection.enabled is True


def test_two_independent_errors_both_surface(valid_config_dict, write_config):
    valid_config_dict["alpha"] = 1.5
    valid_config_dict["persona_pool_id"] = "no_such_pool"
    path = write_config(valid_config_dict)
    with pytest.raises(ConfigLoadError) as exc_info:
        load_run_config(path)
    # alpha is a Pydantic-level failure (surfaces first, blocking the
    # filesystem check per the fail-before-filesystem-check ordering);
    # this asserts the error message names alpha specifically.
    assert any("alpha" in e for e in exc_info.value.errors)


def test_persona_pool_missing_alone_is_reported(valid_config_dict, write_config):
    valid_config_dict["persona_pool_id"] = "no_such_pool"
    path = write_config(valid_config_dict)
    with pytest.raises(ConfigLoadError) as exc_info:
        load_run_config(path)
    assert any("no_such_pool" in e for e in exc_info.value.errors)


def test_mode_other_than_multi_turn_rejected(valid_config_dict, write_config):
    valid_config_dict["mode"] = "single_exposure"
    path = write_config(valid_config_dict)
    with pytest.raises(ConfigLoadError):
        load_run_config(path)
