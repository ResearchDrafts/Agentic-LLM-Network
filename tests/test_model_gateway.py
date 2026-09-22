from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, create_autospec, patch

import litellm
import pytest

from sandbox.cost_tracker import CostTracker
from sandbox.model_gateway import FatalGatewayError, ModelGateway
from sandbox.rate_limiter import RateLimiter
from tests.factories import make_run


@pytest.fixture(autouse=True)
def no_real_sleep():
    """Tenacity's wait_exponential backoff (min=2s) would otherwise make
    every retry test take several real seconds; patching asyncio.sleep
    keeps retry tests fast and deterministic without touching the
    production retry configuration."""
    with patch("asyncio.sleep", new_callable=AsyncMock):
        yield


@pytest.fixture
def rate_limiter():
    limiter = create_autospec(RateLimiter, instance=True)
    limiter.acquire = AsyncMock()
    return limiter


@pytest.fixture
def cost_tracker():
    return create_autospec(CostTracker, instance=True)


# Local, not in factories.py: this is a raw litellm-shaped response, and it
# deliberately DOES define response_ms even though real ModelResponse objects
# do not (phase4.md D2). Keeping it lets the D2 tests below prove production
# code ignores the field rather than merely tolerating its absence.
def _make_raw_response(text="ok", prompt_tokens=10, completion_tokens=5, response_ms=123):
    raw = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
        response_ms=response_ms,
    )
    raw.model_dump = lambda: {"text": text}
    return raw


async def test_transient_error_then_success_retries_and_succeeds(rate_limiter, cost_tracker):
    gateway = ModelGateway("gpt-4o", rate_limiter, cost_tracker)
    success = _make_raw_response(text="hello")
    mock_completion = AsyncMock(
        side_effect=[
            litellm.exceptions.RateLimitError(message="rate limited", llm_provider="openai", model="gpt-4o"),
            success,
        ]
    )

    with patch("litellm.acompletion", mock_completion):
        response = await gateway.generate("prompt")

    assert response.text == "hello"
    assert mock_completion.call_count == 2


async def test_authentication_error_fails_on_first_attempt_no_retry(rate_limiter, cost_tracker):
    gateway = ModelGateway("gpt-4o", rate_limiter, cost_tracker)
    mock_completion = AsyncMock(
        side_effect=litellm.exceptions.AuthenticationError(
            message="bad key", llm_provider="openai", model="gpt-4o"
        )
    )

    with patch("litellm.acompletion", mock_completion):
        with pytest.raises(FatalGatewayError):
            await gateway.generate("prompt")

    assert mock_completion.call_count == 1


def test_unrecognized_model_id_raises_value_error_at_construction(rate_limiter, cost_tracker):
    with pytest.raises(ValueError):
        ModelGateway("totally-made-up-model-id", rate_limiter, cost_tracker)


async def test_text_call_records_cost_with_text_modality(rate_limiter, cost_tracker):
    gateway = ModelGateway("gpt-4o", rate_limiter, cost_tracker)
    with patch("litellm.acompletion", AsyncMock(return_value=_make_raw_response())):
        await gateway.generate("prompt", image=None)

    cost_tracker.record.assert_called_once_with(
        provider="openai",
        model_id="gpt-4o",
        modality="text",
        prompt_tokens=10,
        completion_tokens=5,
    )


async def test_vision_call_records_cost_with_vision_modality(rate_limiter, cost_tracker):
    gateway = ModelGateway("gpt-4o", rate_limiter, cost_tracker)
    with patch("litellm.acompletion", AsyncMock(return_value=_make_raw_response())):
        await gateway.generate("prompt", image=b"fake-image-bytes")

    cost_tracker.record.assert_called_once_with(
        provider="openai",
        model_id="gpt-4o",
        modality="vision",
        prompt_tokens=10,
        completion_tokens=5,
    )


async def test_rate_limiter_release_called_once_on_success(rate_limiter, cost_tracker):
    gateway = ModelGateway("gpt-4o", rate_limiter, cost_tracker)
    with patch("litellm.acompletion", AsyncMock(return_value=_make_raw_response())):
        await gateway.generate("prompt")

    rate_limiter.release.assert_called_once_with("openai")


async def test_rate_limiter_release_called_once_even_when_call_raises(rate_limiter, cost_tracker):
    gateway = ModelGateway("gpt-4o", rate_limiter, cost_tracker)
    mock_completion = AsyncMock(
        side_effect=litellm.exceptions.AuthenticationError(
            message="bad key", llm_provider="openai", model="gpt-4o"
        )
    )

    with patch("litellm.acompletion", mock_completion):
        with pytest.raises(FatalGatewayError):
            await gateway.generate("prompt")

    rate_limiter.release.assert_called_once_with("openai")


# --- phase4.md D1: pricing lookup must use the BARE model id -------------


@pytest.mark.parametrize(
    "model_backend_id, expected_provider, expected_model_id",
    [
        ("gpt-4o", "openai", "gpt-4o"),
        ("gpt-4o-mini", "openai", "gpt-4o-mini"),
        ("gpt-3.5-turbo", "openai", "gpt-3.5-turbo"),
        ("anthropic/claude-sonnet-4-6", "anthropic", "claude-sonnet-4-6"),
        ("google/gemini-2.0-flash", "google", "gemini-2.0-flash"),
        ("local/llama-3.1-70b", "local", "llama-3.1-70b"),
        ("local/llama-3.1-8b", "local", "llama-3.1-8b"),
    ],
)
async def test_cost_record_uses_bare_model_id(
    rate_limiter, cost_tracker, model_backend_id, expected_provider, expected_model_id
):
    """Previously generate() passed the full prefixed model_backend_id as
    model_id while provider was already stripped, so the pricing lookup became
    pricing["anthropic"]["anthropic/claude-sonnet-4-6"] and raised for every
    model except a bare OpenAI one."""
    gateway = ModelGateway(model_backend_id, rate_limiter, cost_tracker)
    with patch("litellm.acompletion", AsyncMock(return_value=_make_raw_response())):
        await gateway.generate("prompt")

    kwargs = cost_tracker.record.call_args.kwargs
    assert kwargs["provider"] == expected_provider
    assert kwargs["model_id"] == expected_model_id


@pytest.mark.parametrize(
    "model_backend_id",
    [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-3.5-turbo",
        "anthropic/claude-sonnet-4-6",
        "google/gemini-2.0-flash",
        "local/llama-3.1-70b",
        "local/llama-3.1-8b",
    ],
)
async def test_every_priced_model_resolves_against_real_cost_tracker(
    rate_limiter, model_backend_id
):
    """End-to-end against the real pricing_table.yaml and a real CostTracker,
    not an autospec mock. This is the seam that was completely untested and
    that D1 broke: 6 of 7 priced models raised ValueError at record() time,
    after the API call had already been paid for."""
    run = make_run(model_backend_id=model_backend_id)
    tracker = CostTracker(run, Path("pricing_table.yaml"))
    gateway = ModelGateway(model_backend_id, rate_limiter, tracker)

    with patch("litellm.acompletion", AsyncMock(return_value=_make_raw_response())):
        await gateway.generate("prompt")  # must not raise

    assert tracker.total_usd >= 0.0


# --- phase4.md D2: latency is measured, not read off the response --------


async def test_latency_is_measured_not_read_from_provider_response(
    rate_limiter, cost_tracker
):
    """full_design_doc.md Sec 3.5 reads raw.response_ms, which does not exist
    on litellm's ModelResponse. The mock below supplies response_ms=999 to
    prove production code no longer reads it."""
    gateway = ModelGateway("gpt-4o", rate_limiter, cost_tracker)
    with patch(
        "litellm.acompletion",
        AsyncMock(return_value=_make_raw_response(response_ms=999)),
    ):
        response = await gateway.generate("prompt")

    assert response.latency_ms != 999
    assert isinstance(response.latency_ms, int)
    assert response.latency_ms >= 0


async def test_latency_survives_a_response_object_without_response_ms(
    rate_limiter, cost_tracker
):
    """The real regression: a response lacking response_ms entirely, as every
    real litellm ModelResponse does. This raised AttributeError, which is
    neither gateway error type and so aborted the whole run."""
    raw = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )
    raw.model_dump = lambda: {"text": "ok"}

    gateway = ModelGateway("gpt-4o", rate_limiter, cost_tracker)
    with patch("litellm.acompletion", AsyncMock(return_value=raw)):
        response = await gateway.generate("prompt")

    assert isinstance(response.latency_ms, int)


# --- audit B1/B3: reaching a self-hosted server -------------------------


@pytest.mark.parametrize(
    "model_backend_id, expect_vision",
    [
        ("hosted_vllm/Qwen/Qwen2.5-7B-Instruct", False),
        ("hosted_vllm/Qwen/Qwen2.5-VL-7B-Instruct", True),
        ("hosted_vllm/Qwen/Qwen2.5-VL-3B-Instruct", True),
        ("groq/openai/gpt-oss-120b", False),
        ("cerebras/glm-4.7", False),
    ],
)
def test_self_hosted_and_free_tier_models_construct(
    rate_limiter, cost_tracker, model_backend_id, expect_vision
):
    """_resolve_vision_support raises on an unrecognized id at construction,
    so an unlisted model cannot start a run at all."""
    gateway = ModelGateway(model_backend_id, rate_limiter, cost_tracker)
    assert gateway.supports_vision is expect_vision


async def test_api_base_is_forwarded_to_litellm(rate_limiter, cost_tracker):
    """Without this there is no way to reach a local vLLM server: litellm
    routes on the model prefix alone and would call a hosted provider."""
    gateway = ModelGateway(
        "hosted_vllm/Qwen/Qwen2.5-7B-Instruct", rate_limiter, cost_tracker,
        api_base="http://localhost:8000/v1",
    )
    mock_completion = AsyncMock(return_value=_make_raw_response())
    with patch("litellm.acompletion", mock_completion):
        await gateway.generate("prompt")

    assert mock_completion.call_args.kwargs["api_base"] == "http://localhost:8000/v1"


async def test_api_base_defaults_to_none_for_hosted_providers(rate_limiter, cost_tracker):
    gateway = ModelGateway("gpt-4o", rate_limiter, cost_tracker)
    mock_completion = AsyncMock(return_value=_make_raw_response())
    with patch("litellm.acompletion", mock_completion):
        await gateway.generate("prompt")

    assert mock_completion.call_args.kwargs["api_base"] is None


@pytest.mark.parametrize(
    "model_backend_id, bare, modality",
    [
        ("hosted_vllm/Qwen/Qwen2.5-7B-Instruct", "Qwen2.5-7B-Instruct", "text"),
        ("hosted_vllm/Qwen/Qwen2.5-VL-7B-Instruct", "Qwen2.5-VL-7B-Instruct", "vision"),
        ("groq/openai/gpt-oss-120b", "gpt-oss-120b", "text"),
        ("cerebras/gpt-oss-120b", "gpt-oss-120b", "text"),
    ],
)
async def test_every_configured_backend_has_a_pricing_entry(
    rate_limiter, model_backend_id, bare, modality
):
    """CostTracker raises on a missing entry AFTER the call completes, so a
    gap here kills a run having already spent the compute."""
    run = make_run(model_backend_id=model_backend_id)
    tracker = CostTracker(run, Path("pricing_table.yaml"))
    gateway = ModelGateway(model_backend_id, rate_limiter, tracker)

    image = b"fake" if modality == "vision" else None
    with patch("litellm.acompletion", AsyncMock(return_value=_make_raw_response())):
        await gateway.generate("prompt", image=image)  # must not raise

    assert tracker.total_usd == 0.0  # genuinely free, not merely unpriced


# --- concurrency cap: bounding in-flight requests ------------------------
#
# These use asyncio.Barrier rather than sleeps to force genuine overlap: the
# module's autouse no_real_sleep fixture patches asyncio.sleep, so a mock that
# awaits it never actually yields and tasks cannot interleave.


async def _peak_concurrency(gateway, n_requests, barrier_size):
    """Runs n_requests through the gateway, returning peak simultaneous calls.

    Each mocked call waits on a barrier of barrier_size, so it cannot return
    until that many are genuinely in flight at once.
    """
    import asyncio

    barrier = asyncio.Barrier(barrier_size)
    in_flight = peak = 0

    async def tracked(*a, **k):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        try:
            await asyncio.wait_for(barrier.wait(), timeout=5)
        except (TimeoutError, asyncio.BrokenBarrierError):
            pass
        in_flight -= 1
        return _make_raw_response()

    with patch("litellm.acompletion", AsyncMock(side_effect=tracked)):
        await asyncio.gather(*(gateway.generate("p") for _ in range(n_requests)))
    return peak


async def test_max_concurrency_bounds_simultaneous_requests(rate_limiter, cost_tracker):
    """The Orchestrator dispatches all M agents at once. Against a local
    server batching 32 at a time, M=100 leaves 68 connections open and idle
    each turn; they accumulate until the server closes them and the run dies
    with "Server disconnected" while the process is still healthy."""
    gateway = ModelGateway("gpt-4o", rate_limiter, cost_tracker, max_concurrency=4)
    peak = await _peak_concurrency(gateway, n_requests=40, barrier_size=4)
    assert peak == 4, f"expected exactly the cap, got {peak}"


async def test_unbounded_by_default(rate_limiter, cost_tracker):
    """Hosted providers pool connections themselves, so omitting the cap
    preserves the previous behaviour."""
    gateway = ModelGateway("gpt-4o", rate_limiter, cost_tracker)
    peak = await _peak_concurrency(gateway, n_requests=20, barrier_size=20)
    assert peak == 20, f"expected all 20 in flight, got {peak}"


async def test_all_requests_still_complete_under_the_cap(rate_limiter, cost_tracker):
    """Bounding is not dropping: every request runs, just not all at once."""
    import asyncio

    gateway = ModelGateway("gpt-4o", rate_limiter, cost_tracker, max_concurrency=4)
    with patch("litellm.acompletion", AsyncMock(return_value=_make_raw_response())):
        results = await asyncio.gather(*(gateway.generate("p") for _ in range(40)))
    assert len(results) == 40
    assert all(r.text == "ok" for r in results)


async def test_cap_holds_across_retries(rate_limiter, cost_tracker):
    """The slot is held for the whole call, retries included, so a retrying
    agent never adds a connection beyond the cap."""
    import asyncio

    gateway = ModelGateway("gpt-4o", rate_limiter, cost_tracker, max_concurrency=2)
    in_flight = peak = 0
    calls = {"n": 0}

    async def flaky(*a, **k):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        calls["n"] += 1
        should_fail = calls["n"] % 3 == 1
        in_flight -= 1
        if should_fail:
            raise litellm.exceptions.RateLimitError(
                message="429", llm_provider="openai", model="gpt-4o")
        return _make_raw_response()

    with patch("litellm.acompletion", AsyncMock(side_effect=flaky)):
        await asyncio.gather(*(gateway.generate("p") for _ in range(12)))

    assert peak <= 2, f"cap of 2 exceeded during retries: {peak}"
    assert calls["n"] > 12, "expected retries to have occurred"
