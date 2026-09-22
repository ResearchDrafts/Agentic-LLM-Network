import random

import pytest

from sandbox.meme_pool_manager import MemePoolManager
from sandbox.models import Agent, MemeInjectionConfig


def _make_agent(agent_id: str = "agent_0000") -> Agent:
    return Agent(
        agent_id=agent_id,
        run_id="test_run",
        persona="a persona",
        language_condition="english",
        model_backend_id="gpt-4o",
        initial_stance=4.0,
    )


def _config(**overrides) -> MemeInjectionConfig:
    base = dict(enabled=True, meme_pool_id="test_pool", injection_rate=0.5, injection_schedule="random")
    base.update(overrides)
    return MemeInjectionConfig(**base)


def test_disabled_yields_none_for_every_speaker_regardless_of_rate():
    config = MemeInjectionConfig(enabled=False)
    manager = MemePoolManager(config, random.Random(1))
    speakers = [_make_agent(f"agent_{i:04d}") for i in range(5)]

    result = manager.resolve_injections_for_turn(speakers, turn=1)

    assert all(v is None for v in result.values())


def test_injection_rate_zero_never_injects_across_many_calls():
    config = _config(injection_rate=0.0)
    manager = MemePoolManager(config, random.Random(1))
    speakers = [_make_agent(f"agent_{i:04d}") for i in range(20)]

    for turn in range(1, 20):
        result = manager.resolve_injections_for_turn(speakers, turn=turn)
        assert all(v is None for v in result.values())


def test_injection_rate_one_random_schedule_always_injects():
    config = _config(injection_rate=1.0, injection_schedule="random")
    manager = MemePoolManager(config, random.Random(1))
    speakers = [_make_agent(f"agent_{i:04d}") for i in range(20)]

    result = manager.resolve_injections_for_turn(speakers, turn=1)

    assert all(v is not None for v in result.values())


def test_fixed_turn_schedule_never_injects_except_on_configured_turn():
    config = _config(injection_rate=1.0, injection_schedule="fixed_turn", fixed_turns={5})
    manager = MemePoolManager(config, random.Random(1))
    speakers = [_make_agent()]

    for turn in range(1, 8):
        result = manager.resolve_injections_for_turn(speakers, turn=turn)
        if turn == 5:
            assert result["agent_0000"] is not None
        else:
            assert result["agent_0000"] is None


def test_identically_seeded_rng_produces_identical_injection_decisions():
    speakers = [_make_agent(f"agent_{i:04d}") for i in range(10)]
    config = _config(injection_rate=0.5)
    manager1 = MemePoolManager(config, random.Random(42))
    manager2 = MemePoolManager(config, random.Random(42))

    for turn in range(1, 6):
        result1 = manager1.resolve_injections_for_turn(speakers, turn)
        result2 = manager2.resolve_injections_for_turn(speakers, turn)
        ids1 = {agent_id: (meme.meme_id if meme else None) for agent_id, meme in result1.items()}
        ids2 = {agent_id: (meme.meme_id if meme else None) for agent_id, meme in result2.items()}
        assert ids1 == ids2


def test_empty_pool_file_raises_value_error_at_construction():
    config = _config(meme_pool_id="empty_pool")

    with pytest.raises(ValueError):
        MemePoolManager(config, random.Random(1))


def test_missing_pool_file_raises_file_not_found_at_construction():
    config = _config(meme_pool_id="does_not_exist")

    with pytest.raises(FileNotFoundError):
        MemePoolManager(config, random.Random(1))


# --- phase4.md D3: every pooled meme's image must exist on disk ----------


def test_every_image_path_in_the_real_pool_resolves():
    """Fixture-drift guard. data/memes/test_pool.jsonl referenced three
    images while only test_001.jpg existed, so a meme-enabled run against a
    vision-capable backend raised FileNotFoundError out of
    vision_fallback._load_image for roughly two sampled memes in three,
    non-deterministically depending on the RNG draw. No test caught it
    because test_vision_fallback.py hand-builds its MemeContent and this
    module's tests never read image_path."""
    from pathlib import Path

    config = MemeInjectionConfig(
        enabled=True, meme_pool_id="test_pool", injection_rate=1.0
    )
    manager = MemePoolManager(config, random.Random(0))

    missing = [
        meme.image_path
        for meme in manager._pool
        if not Path(meme.image_path).exists()
    ]
    assert not missing, f"meme pool references images that do not exist: {missing}"
