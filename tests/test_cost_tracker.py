import asyncio
from pathlib import Path

import pytest

from sandbox.cost_tracker import CostCeilingExceeded, CostTracker
from sandbox.models import ExperimentRun

PRICING_TABLE = Path("pricing_table.yaml")


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


def test_record_computes_expected_cost_and_accumulates():
    tracker = CostTracker(_make_run(), PRICING_TABLE)
    tracker.record("openai", "gpt-4o", "text", prompt_tokens=1000, completion_tokens=1000)
    assert tracker.total_usd == pytest.approx(0.0050 + 0.0150)


def test_unknown_combination_raises_value_error_not_silent_zero():
    tracker = CostTracker(_make_run(), PRICING_TABLE)
    with pytest.raises(ValueError):
        tracker.record("openai", "no-such-model", "text", prompt_tokens=100, completion_tokens=100)
    assert tracker.total_usd == 0.0


def test_ceiling_exceeded_after_total_is_updated():
    run = _make_run(max_cost_usd=0.01)
    tracker = CostTracker(run, PRICING_TABLE)
    with pytest.raises(CostCeilingExceeded):
        tracker.record("openai", "gpt-4o", "text", prompt_tokens=1000, completion_tokens=1000)
    # the triggering call's cost is still counted
    assert tracker.total_usd == pytest.approx(0.0050 + 0.0150)


def test_none_ceiling_never_raises_regardless_of_total():
    run = _make_run(max_cost_usd=None)
    tracker = CostTracker(run, PRICING_TABLE)
    for _ in range(50):
        tracker.record("openai", "gpt-4o", "text", prompt_tokens=1000, completion_tokens=1000)
    assert tracker.total_usd > 1.0


def test_vision_and_text_modality_tracked_independently():
    tracker = CostTracker(_make_run(), PRICING_TABLE)
    tracker.record("google", "gemini-2.0-flash", "text", prompt_tokens=1000, completion_tokens=1000)
    text_only_total = tracker.total_usd
    tracker.record("google", "gemini-2.0-flash", "vision", prompt_tokens=1000, completion_tokens=1000)
    assert tracker.total_usd > text_only_total


def test_concurrent_record_calls_lose_no_updates():
    tracker = CostTracker(_make_run(), PRICING_TABLE)

    async def _run():
        await asyncio.gather(
            *[
                asyncio.to_thread(
                    tracker.record, "local", "llama-3.1-70b", "text", 1000, 1000
                )
                for _ in range(1)
            ],
            *[
                asyncio.to_thread(
                    tracker.record, "openai", "gpt-4o", "text", 1000, 1000
                )
                for _ in range(20)
            ],
        )

    asyncio.run(_run())
    expected = 20 * (0.0050 + 0.0150)
    assert tracker.total_usd == pytest.approx(expected)


def test_missing_pricing_table_raises_at_construction(tmp_path):
    with pytest.raises(FileNotFoundError):
        CostTracker(_make_run(), tmp_path / "no_such_table.yaml")
