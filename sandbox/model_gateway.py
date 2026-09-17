"""Feature 7: Model Gateway.

The single normalized async interface to every LLM provider, backed by
LiteLLM. Direct port of full_design_doc.md Sec 3.5.

Per Phase 1's summary, CostTracker.record() is synchronous (not
`async def`), so generate() calls it as a plain call, never awaited.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass

import litellm
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from sandbox.cost_tracker import CostTracker
from sandbox.rate_limiter import RateLimiter

# Matches full_design_doc.md Sec 3.5's _resolve_vision_support table
# exactly. This is a strict superset of pricing_table.yaml's current
# entries -- llava and qwen-vl are recognized here as vision-capable per
# the HLD's mention of local vision models, even though pricing_table.yaml
# has no pricing entry for them yet. Using either of those two in a real
# run would surface loudly via CostTracker's "no pricing entry" ValueError
# at record() time, consistent with this system's fail-before-spending
# philosophy, rather than silently misclassifying them as unrecognized.
VISION_CAPABLE = {"gpt-4o", "gpt-4o-mini", "claude-sonnet-4-6", "gemini-2.0-flash", "llava", "qwen-vl"}
TEXT_ONLY = {"gpt-3.5-turbo", "llama-3.1-8b", "llama-3.1-70b"}


class TransientGatewayError(Exception):
    """Wraps any provider error judged retryable (timeouts, 429s, 5xxs)."""


class FatalGatewayError(Exception):
    """Wraps any provider error judged non-retryable (auth failure,
    invalid model id, content policy rejection) -- never retried."""


@dataclass
class BackendResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    raw_provider_response: dict


class ModelGateway:
    def __init__(self, model_backend_id: str, rate_limiter: RateLimiter, cost_tracker: CostTracker):
        self.model_backend_id = model_backend_id
        self.supports_vision = _resolve_vision_support(model_backend_id)
        self._rate_limiter = rate_limiter
        self._cost_tracker = cost_tracker

    async def generate(
        self,
        prompt_text: str,
        image: bytes | None = None,
        temperature: float = 0.7,
    ) -> BackendResponse:
        """
        Normalizes a call to self.model_backend_id via LiteLLM. If image
        is provided but self.supports_vision is False, the CALLER is
        responsible for having already substituted a caption-only prompt
        (via vision_fallback.py) BEFORE calling this method -- this method
        does not itself perform the fallback substitution, it only
        reports supports_vision so callers can decide.
        """
        provider = self._provider_name()
        await self._rate_limiter.acquire(provider)
        try:
            response = await self._call_with_retry(prompt_text, image, temperature)
        finally:
            self._rate_limiter.release(provider)

        modality = "vision" if image is not None else "text"
        self._cost_tracker.record(
            provider=provider,
            model_id=self.model_backend_id,
            modality=modality,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
        )
        return response

    @retry(
        retry=retry_if_exception_type(TransientGatewayError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    async def _call_with_retry(
        self, prompt_text: str, image: bytes | None, temperature: float
    ) -> BackendResponse:
        try:
            raw = await litellm.acompletion(
                model=self.model_backend_id,
                messages=_build_messages(prompt_text, image),
                temperature=temperature,
            )
        except litellm.exceptions.RateLimitError as e:
            raise TransientGatewayError(str(e)) from e
        except litellm.exceptions.APIConnectionError as e:
            raise TransientGatewayError(str(e)) from e
        except litellm.exceptions.AuthenticationError as e:
            raise FatalGatewayError(str(e)) from e
        except litellm.exceptions.BadRequestError as e:
            raise FatalGatewayError(str(e)) from e

        return BackendResponse(
            text=raw.choices[0].message.content,
            prompt_tokens=raw.usage.prompt_tokens,
            completion_tokens=raw.usage.completion_tokens,
            latency_ms=raw.response_ms,
            raw_provider_response=raw.model_dump(),
        )

    def _provider_name(self) -> str:
        return self.model_backend_id.split("/")[0] if "/" in self.model_backend_id else "openai"


def _resolve_vision_support(model_backend_id: str) -> bool:
    """Static lookup table, not a runtime probe -- checked once at
    ModelGateway construction. Raises ValueError for an unrecognized
    model_backend_id rather than silently defaulting to False, since a
    silent False could cause a vision-capable model to be treated as
    text-only without anyone noticing."""
    base = model_backend_id.split("/")[-1]
    if base in VISION_CAPABLE:
        return True
    if base in TEXT_ONLY:
        return False
    raise ValueError(f"unrecognized model_backend_id, cannot determine vision support: {model_backend_id}")


def _build_messages(prompt_text: str, image: bytes | None) -> list[dict]:
    if image is None:
        return [{"role": "user", "content": prompt_text}]
    b64 = base64.b64encode(image).decode()
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }
    ]
