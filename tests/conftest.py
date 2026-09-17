import copy

import pytest
import yaml


def _base_config_dict() -> dict:
    return {
        "run_id": "test_run_0001",
        "rq_target": "RQ1_RQ2",
        "topic": "a neutral test topic",
        "alpha": 0.5,
        "M": 10,
        "N": 3,
        "K": 2,
        "trial_number": 1,
        "language_condition": "english",
        "model_backend_id": "gpt-4o",
        "stance_scale": [1, 2, 3, 4, 5, 6, 7],
        "persona_pool_id": "test_pool",
        "seed": 42,
        "temperature": 0.7,
    }


@pytest.fixture
def valid_config_dict() -> dict:
    return _base_config_dict()


@pytest.fixture
def write_config(tmp_path):
    """Writes a dict as a YAML file under tmp_path and returns its Path."""

    def _write(config: dict, filename: str = "run.yaml"):
        path = tmp_path / filename
        path.write_text(yaml.safe_dump(config))
        return path

    return _write
