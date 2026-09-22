import pytest

from sandbox.seed_manager import SeedManager


def test_same_seed_produces_identical_neighbor_sampling_draws():
    a = SeedManager(seed=7)
    b = SeedManager(seed=7)
    draws_a = [a.neighbor_sampling_rng.random() for _ in range(20)]
    draws_b = [b.neighbor_sampling_rng.random() for _ in range(20)]
    assert draws_a == draws_b


def test_same_seed_produces_identical_meme_injection_draws():
    a = SeedManager(seed=99)
    b = SeedManager(seed=99)
    draws_a = [a.meme_injection_rng.random() for _ in range(20)]
    draws_b = [b.meme_injection_rng.random() for _ in range(20)]
    assert draws_a == draws_b


def test_three_streams_never_correlate():
    m = SeedManager(seed=1)
    neighbor_draws = [m.neighbor_sampling_rng.random() for _ in range(50)]
    meme_draws = [m.meme_injection_rng.random() for _ in range(50)]
    persona_draws = [m.persona_assignment_rng.random() for _ in range(50)]
    assert neighbor_draws != meme_draws
    assert neighbor_draws != persona_draws
    assert meme_draws != persona_draws


def test_property_returns_same_instance_by_identity():
    m = SeedManager(seed=5)
    assert m.neighbor_sampling_rng is m.neighbor_sampling_rng
    assert m.meme_injection_rng is m.meme_injection_rng
    assert m.persona_assignment_rng is m.persona_assignment_rng


def test_repeated_property_access_advances_stream_not_resets_it():
    m = SeedManager(seed=3)
    first = m.neighbor_sampling_rng.random()
    second = m.neighbor_sampling_rng.random()
    assert first != second


# --- phase4.md D8: per-turn streams survive a resume ---------------------


def test_turn_rng_is_a_pure_function_of_seed_purpose_and_turn():
    """The property that makes resume correct: turn t's draws do not depend
    on whether turns 1..t-1 ran in this process."""
    fresh = SeedManager(42).turn_rng("neighbor", 5)

    consumed = SeedManager(42)
    for turn in range(1, 5):  # burn turns 1-4 as a real run would
        consumed.turn_rng("neighbor", turn).random()
    resumed = consumed.turn_rng("neighbor", 5)

    assert [fresh.random() for _ in range(10)] == [resumed.random() for _ in range(10)]


def test_the_long_lived_streams_do_not_survive_a_resume():
    """Documents why turn_rng exists.

    The long-lived properties advance as turns consume them, and that
    position is not checkpointed, so a resumed run replays earlier draws.
    This asserts the broken behaviour deliberately: if a future change made
    the long-lived streams resume-safe, this test should be revisited rather
    than treated as a regression.
    """
    uninterrupted = SeedManager(42).neighbor_sampling_rng
    turn_one = [uninterrupted.random() for _ in range(5)]
    turn_two = [uninterrupted.random() for _ in range(5)]

    after_resume = SeedManager(42).neighbor_sampling_rng
    replayed = [after_resume.random() for _ in range(5)]

    assert replayed == turn_one       # it replays turn 1...
    assert replayed != turn_two       # ...instead of continuing into turn 2


def test_turn_streams_differ_across_turns_and_purposes():
    seeds = SeedManager(42)
    draws = {
        (purpose, turn): seeds.turn_rng(purpose, turn).random()
        for purpose in ("neighbor", "meme")
        for turn in (1, 2, 3)
    }
    assert len(set(draws.values())) == len(draws)


def test_turn_rng_preserves_fix_m_across_two_runs_sharing_a_seed():
    """Fix M: reusing a seed across language arms must hold neighbour
    sampling and meme injection constant while language varies."""
    english = SeedManager(2026)
    hinglish = SeedManager(2026)
    for turn in (1, 2, 3):
        for purpose in ("neighbor", "meme"):
            a = english.turn_rng(purpose, turn)
            b = hinglish.turn_rng(purpose, turn)
            assert [a.random() for _ in range(5)] == [b.random() for _ in range(5)]


def test_turn_rng_rejects_unknown_purpose_and_invalid_turn():
    seeds = SeedManager(1)
    with pytest.raises(ValueError, match="unknown rng purpose"):
        seeds.turn_rng("persona", 1)
    with pytest.raises(ValueError, match="turn must be"):
        seeds.turn_rng("neighbor", 0)
