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
