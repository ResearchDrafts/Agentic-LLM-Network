from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, create_autospec, patch

import litellm
import pytest

from sandbox.cost_tracker import CostTracker
from sandbox.model_gateway import (
    FatalGatewayError,
    ModelGateway,
    TransientGatewayError,
)
from sandbox.rate_limiter import RateLimiter


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
    from sandbox.cost_tracker import CostTracker
    from sandbox.models import ExperimentRun

    run = ExperimentRun(
        run_id="r", rq_target="RQ1_RQ2", topic="t", alpha=0.5, M=10, N=3, K=2,
        trial_number=1, language_condition="english",
        model_backend_id=model_backend_id, stance_scale=[1, 2, 3, 4, 5, 6, 7],
        persona_pool_id="test_pool", seed=1, temperature=0.7,
    )
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
