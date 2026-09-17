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
