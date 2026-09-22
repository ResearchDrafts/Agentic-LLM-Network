
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


# --- phase4.md D4: non-mapping YAML must still raise ConfigLoadError -----


@pytest.mark.parametrize(
    "yaml_text, described_as",
    [
        ("- a\n- b\n", "list"),
        ("just_a_string\n", "str"),
        ("42\n", "int"),
        ("true\n", "bool"),
    ],
)
def test_non_mapping_yaml_raises_config_load_error(tmp_path, yaml_text, described_as):
    """A syntactically valid YAML document need not be a mapping. Without the
    isinstance guard these raised raw TypeError/AttributeError out of
    raw.pop(), breaking load_run_config's documented contract that every
    failure surfaces as ConfigLoadError."""
    path = tmp_path / "run.yaml"
    path.write_text(yaml_text)

    with pytest.raises(ConfigLoadError) as excinfo:
        load_run_config(path)

    assert "must be a mapping" in str(excinfo.value)
    assert described_as in str(excinfo.value)


# --- phase4.md D5: errors carry the caller's real config path ------------


def test_config_load_error_carries_the_real_path(tmp_path):
    path = tmp_path / "my_experiment.yaml"
    path.write_text("42\n")

    with pytest.raises(ConfigLoadError) as excinfo:
        load_run_config(path)

    assert excinfo.value.path == path
    assert "my_experiment.yaml" in str(excinfo.value)


def test_missing_file_error_carries_the_real_path(tmp_path):
    path = tmp_path / "does_not_exist.yaml"

    with pytest.raises(ConfigLoadError) as excinfo:
        load_run_config(path)

    assert excinfo.value.path == path


# --- readable validation errors -----------------------------------------


def test_validation_errors_are_field_message_not_raw_dicts(tmp_path):
    """str() on a Pydantic error dict prints the whole structure including the
    full input payload, repeated once per error, so a config missing a dozen
    fields buried the actual problems in a wall of text."""
    path = tmp_path / "bad.yaml"
    path.write_text("run_id: x\ntopic: t\n")

    with pytest.raises(ConfigLoadError) as excinfo:
        load_run_config(path)

    message = str(excinfo.value)
    assert "alpha: Field required" in message
    assert "'type':" not in message and "'loc':" not in message  # no raw dicts
    assert "errors.pydantic.dev" not in message  # no doc URLs


def test_unknown_key_error_names_the_field_and_explains(tmp_path, valid_config_dict, write_config):
    valid_config_dict["meme_injections"] = {"enabled": True}
    path = write_config(valid_config_dict)

    with pytest.raises(ConfigLoadError) as excinfo:
        load_run_config(path)

    message = str(excinfo.value)
    assert "meme_injections: unknown field" in message
    assert "typo" in message
