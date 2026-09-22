import random
from collections import Counter

import pytest

from sandbox.interaction_engine import AlphaSampling
from sandbox.models import Agent, StanceRecord


def _make_agent(agent_id: str, initial_stance: float, stance_history: list[StanceRecord] | None = None) -> Agent:
    return Agent(
        agent_id=agent_id,
        run_id="test_run",
        persona="a persona",
        language_condition="english",
        model_backend_id="gpt-4o",
        initial_stance=initial_stance,
        stance_history=stance_history or [],
    )


def _population(n: int, stance_fn) -> list[Agent]:
    return [_make_agent(f"agent_{i:04d}", stance_fn(i)) for i in range(n)]


def test_construction_rejects_alpha_outside_zero_one():
    with pytest.raises(ValueError):
        AlphaSampling(alpha=1.5, N=3)
    with pytest.raises(ValueError):
        AlphaSampling(alpha=-0.1, N=3)


def test_construction_rejects_n_less_than_one():
    with pytest.raises(ValueError):
        AlphaSampling(alpha=0.5, N=0)


def test_alpha_one_deterministically_returns_closest_stance_candidates():
    # This is a *weighted* sample, not a strict top-k-by-weight pick, so
    # "deterministic" here means: with stances separated enough that the
    # far candidates' weights are negligible next to the close candidates',
    # the closest two win for any rng draw, not just a cherry-picked seed
    # (verified empirically across many seeds during test development).
    strategy = AlphaSampling(alpha=1.0, N=2)
    speaker = _make_agent("speaker", initial_stance=4.0)
    others = [
        _make_agent("far_1", initial_stance=-996.0),
        _make_agent("far_2", initial_stance=1004.0),
        _make_agent("close_1", initial_stance=4.1),
        _make_agent("close_2", initial_stance=3.9),
    ]
    all_agents = [speaker, *others]

    for seed in range(10):
        result = strategy.select_neighbors(speaker, all_agents, turn=1, rng=random.Random(seed))
        assert {a.agent_id for a in result} == {"close_1", "close_2"}


def test_alpha_zero_selection_frequency_is_roughly_uniform_across_seeds():
    strategy = AlphaSampling(alpha=0.0, N=1)
    speaker = _make_agent("speaker", initial_stance=4.0)
    candidates = _population(10, lambda i: float(1 + i))  # varied stances, irrelevant at alpha=0
    candidates = [c for c in candidates if c.agent_id != "speaker"]
    all_agents = [speaker, *candidates]

    counts = Counter()
    trials = 2000
    for seed in range(trials):
        chosen = strategy.select_neighbors(speaker, all_agents, turn=1, rng=random.Random(seed))
        counts[chosen[0].agent_id] += 1

    expected = trials / len(candidates)
    for c in candidates:
        # wide tolerance: uniform-ish, not exact, over 2000 trials across 10 candidates
        assert 0.5 * expected < counts[c.agent_id] < 1.5 * expected


def test_never_returns_speaker_and_always_returns_n_distinct_agents():
    strategy = AlphaSampling(alpha=0.5, N=3)
    speaker = _make_agent("speaker", initial_stance=4.0)
    others = _population(6, lambda i: float(i))
    all_agents = [speaker, *others]

    result = strategy.select_neighbors(speaker, all_agents, turn=1, rng=random.Random(7))

    assert speaker.agent_id not in {a.agent_id for a in result}
    assert len(result) == 3
    assert len({a.agent_id for a in result}) == 3


def test_population_too_small_raises_value_error():
    strategy = AlphaSampling(alpha=0.5, N=5)
    speaker = _make_agent("speaker", initial_stance=4.0)
    others = _population(3, lambda i: float(i))
    all_agents = [speaker, *others]

    with pytest.raises(ValueError):
        strategy.select_neighbors(speaker, all_agents, turn=1, rng=random.Random(0))


def test_identically_seeded_rng_produces_identical_neighbor_sets():
    strategy = AlphaSampling(alpha=0.5, N=3)
    speaker = _make_agent("speaker", initial_stance=4.0)
    others = _population(8, lambda i: float(i))
    all_agents = [speaker, *others]

    result1 = strategy.select_neighbors(speaker, all_agents, turn=1, rng=random.Random(42))
    result2 = strategy.select_neighbors(speaker, all_agents, turn=1, rng=random.Random(42))

    assert [a.agent_id for a in result1] == [a.agent_id for a in result2]


def test_turn_one_always_reads_initial_stance_even_with_stance_history_present():
    strategy = AlphaSampling(alpha=1.0, N=1)
    # speaker's initial_stance (4.0) is close to "matches_initial"; its
    # stance_history (if wrongly read at turn=1) would instead be close to
    # "matches_history" -- this distinguishes which one select_neighbors used.
    speaker = _make_agent(
        "speaker",
        initial_stance=4.0,
        stance_history=[StanceRecord(turn=1, stance_value=1.0, reason_text="r", interaction_id="i1")],
    )
    matches_initial = _make_agent("matches_initial", initial_stance=4.0)
    matches_history = _make_agent("matches_history", initial_stance=1.0)
    all_agents = [speaker, matches_initial, matches_history]

    result = strategy.select_neighbors(speaker, all_agents, turn=1, rng=random.Random(0))

    assert result[0].agent_id == "matches_initial"
