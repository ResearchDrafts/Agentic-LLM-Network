import random
from pathlib import Path

import pytest

from sandbox.agent_manager import AgentManager
from sandbox.models import ExperimentRun, Interaction


def _make_run(**overrides) -> ExperimentRun:
    base = dict(
        run_id="test_run",
        rq_target="RQ1_RQ2",
        topic="t",
        alpha=0.5,
        M=10,
        N=3,
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


def _make_interaction(
    agent_id: str,
    turn: int = 1,
    stance_after: float = 5.0,
    interaction_id: str = "int_1",
    reason_text: str = "because",
) -> Interaction:
    return Interaction(
        interaction_id=interaction_id,
        run_id="test_run",
        turn=turn,
        speaker_agent_id=agent_id,
        neighbor_agent_ids=["agent_0001"],
        stance_before=4.0,
        stance_after=stance_after,
        reason_text=reason_text,
        content_type="generated_text",
        prompt_token_count=10,
        completion_token_count=10,
        model_backend_id="gpt-4o",
        latency_ms=100,
        api_call_status="success",
        timestamp_utc="2026-01-01T00:00:00Z",
    )


def _by_id(agents: list, agent_id: str):
    return next(a for a in agents if a.agent_id == agent_id)


def test_initialize_population_cycles_persona_pool():
    run = _make_run(M=100, persona_pool_id="pool_20")
    manager = AgentManager(run, random.Random(run.seed))
    agents = manager.initialize_population()

    assert len(agents) == 100
    pool = [line.strip() for line in Path("data/personas/pool_20.jsonl").read_text().splitlines() if line.strip()]
    assert len(pool) == 20
    for i, agent in enumerate(agents):
        assert agent.persona == pool[i % 20]


def test_identically_seeded_rng_produces_identical_initial_stance_sequences():
    run = _make_run()
    agents1 = AgentManager(run, random.Random(run.seed)).initialize_population()
    agents2 = AgentManager(run, random.Random(run.seed)).initialize_population()

    assert [a.initial_stance for a in agents1] == [a.initial_stance for a in agents2]


def test_get_agent_unknown_id_raises_key_error_uncaught():
    run = _make_run(M=100)
    manager = AgentManager(run, random.Random(run.seed))
    manager.initialize_population()

    with pytest.raises(KeyError):
        manager.get_agent("agent_9999")


def test_missing_persona_pool_raises_file_not_found_at_initialize_time():
    run = _make_run(persona_pool_id="does_not_exist")
    manager = AgentManager(run, random.Random(run.seed))

    with pytest.raises(FileNotFoundError):
        manager.initialize_population()


def test_apply_interaction_appends_history_and_caps_memory_window_at_five():
    run = _make_run(M=2, N=1)
    manager = AgentManager(run, random.Random(run.seed))
    manager.initialize_population()
    agent_id = "agent_0000"

    for turn in range(1, 8):  # 7 turns, more than the cap of 5
        manager.apply_interaction(
            agent_id,
            _make_interaction(agent_id, turn=turn, stance_after=float(turn), interaction_id=f"int_{turn}"),
        )

    agent = manager.get_agent(agent_id)
    assert len(agent.stance_history) == 7
    assert len(agent.memory_window) == 5
    assert agent.memory_window == [f"int_{t}" for t in range(3, 8)]
    assert agent.current_stance() == 7.0


def test_snapshot_is_unaffected_by_later_mutation():
    run = _make_run(M=2, N=1)
    manager = AgentManager(run, random.Random(run.seed))
    manager.initialize_population()

    snap = manager.snapshot()
    manager.apply_interaction("agent_0000", _make_interaction("agent_0000"))

    assert _by_id(snap, "agent_0000").stance_history == []
    assert len(manager.get_agent("agent_0000").stance_history) == 1


def test_restore_from_snapshot_reproduces_agent_state_exactly():
    run = _make_run(M=2, N=1)
    manager = AgentManager(run, random.Random(run.seed))
    manager.initialize_population()

    manager.apply_interaction("agent_0000", _make_interaction("agent_0000", turn=1, interaction_id="int_1"))
    snap = manager.snapshot()

    manager.apply_interaction("agent_0000", _make_interaction("agent_0000", turn=2, interaction_id="int_2"))
    assert len(manager.get_agent("agent_0000").stance_history) == 2

    manager.restore_from_snapshot(snap)
    restored = manager.get_agent("agent_0000")
    assert len(restored.stance_history) == 1
    assert restored.stance_history[0].interaction_id == "int_1"

    # mutating the manager after restore must not affect the snapshot list passed in
    manager.apply_interaction("agent_0000", _make_interaction("agent_0000", turn=3, interaction_id="int_3"))
    assert len(_by_id(snap, "agent_0000").stance_history) == 1
