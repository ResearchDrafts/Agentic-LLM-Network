"""Feature 7: Model Gateway.

The single normalized async interface to every LLM provider, backed by
LiteLLM. Direct port of full_design_doc.md Sec 3.5.

Per Phase 1's summary, CostTracker.record() is synchronous (not
`async def`), so generate() calls it as a plain call, never awaited.
"""

from __future__ import annotations

import base64
import time
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
VISION_CAPABLE = {
    "gpt-4o", "gpt-4o-mini", "claude-sonnet-4-6", "gemini-2.0-flash", "llava", "qwen-vl",
    # Self-hosted VLMs (phase5,6.md A.4). Serving one of these for a whole RQ3
    # campaign, text turns included, keeps meme and non-meme turns on the same
    # model, so they differ in content rather than in which model produced them.
    "Qwen2.5-VL-7B-Instruct", "Qwen2.5-VL-3B-Instruct",
}
TEXT_ONLY = {
    "gpt-3.5-turbo", "llama-3.1-8b", "llama-3.1-70b",
    # Self-hosted text models.
    "Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct", "Qwen2.5-14B-Instruct",
    # Hosted free-tier fallback (phase5,6.md A.3). None of these accepts
    # images: putting one in VISION_CAPABLE would send image payloads a
    # text-only endpoint rejects, and a meme-enabled run would die mid-turn.
    "gpt-oss-120b", "gpt-oss-20b", "qwen3.8-27b", "glm-4.7",
    # Ollama tags, for local smoke tests on hardware vLLM cannot target
    # (Apple Silicon). Ollama keeps the ":tag" suffix in the model name, so
    # the bare id is "qwen2.5:7b", not "qwen2.5".
    "qwen2.5:3b", "qwen2.5:7b", "llama3.2:3b", "llama3.1:8b",
}


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
    def __init__(
        self,
        model_backend_id: str,
        rate_limiter: RateLimiter,
        cost_tracker: CostTracker,
        api_base: str | None = None,
    ):
        self.model_backend_id = model_backend_id
        self.supports_vision = _resolve_vision_support(model_backend_id)
        self._rate_limiter = rate_limiter
        self._cost_tracker = cost_tracker
        # Endpoint for a self-hosted backend, e.g. "http://localhost:8000/v1"
        # for vLLM. Without this there is no way to reach a local server at
        # all: litellm routes by the model prefix alone and would try to call
        # a hosted provider. None keeps hosted-provider behaviour unchanged.
        self._api_base = api_base

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
            # Bare model id, not the full model_backend_id. pricing_table.yaml
            # nests as provider -> model_id, so passing the prefixed form here
            # while provider is already stripped looks up
            # pricing["anthropic"]["anthropic/claude-sonnet-4-6"] and fails for
            # every model except a bare OpenAI one. Normalized the same way
            # _resolve_vision_support() does.
            model_id=_bare_model_id(self.model_backend_id),
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
        started = time.monotonic()
        try:
            raw = await litellm.acompletion(
                model=self.model_backend_id,
                messages=_build_messages(prompt_text, image),
                temperature=temperature,
                api_base=self._api_base,
            )
        except litellm.exceptions.RateLimitError as e:
            raise TransientGatewayError(str(e)) from e
        except litellm.exceptions.APIConnectionError as e:
            raise TransientGatewayError(str(e)) from e
        except litellm.exceptions.AuthenticationError as e:
            raise FatalGatewayError(str(e)) from e
        except litellm.exceptions.BadRequestError as e:
            raise FatalGatewayError(str(e)) from e

        # Measured here rather than read off the provider response.
        # full_design_doc.md Sec 3.5's sample reads raw.response_ms, which does
        # not exist on litellm's ModelResponse (neither response_ms nor
        # _response_ms is present). That raises AttributeError, which is
        # neither TransientGatewayError nor FatalGatewayError, so per the
        # orchestrator's error contract it would abort the whole run on the
        # first *successful* API response. int() because Interaction.latency_ms
        # is an int with Field(ge=0) and rejects a fractional float.
        latency_ms = int((time.monotonic() - started) * 1000)

        return BackendResponse(
            text=raw.choices[0].message.content,
            prompt_tokens=raw.usage.prompt_tokens,
            completion_tokens=raw.usage.completion_tokens,
            latency_ms=latency_ms,
            raw_provider_response=raw.model_dump(),
        )

    def _provider_name(self) -> str:
        return self.model_backend_id.split("/")[0] if "/" in self.model_backend_id else "openai"


def _bare_model_id(model_backend_id: str) -> str:
    """Strips any provider prefix: 'anthropic/claude-sonnet-4-6' ->
    'claude-sonnet-4-6'. Both the vision table and pricing_table.yaml are
    keyed by bare model ids, so every lookup must normalize through here."""
    return model_backend_id.split("/")[-1]


def _resolve_vision_support(model_backend_id: str) -> bool:
    """Static lookup table, not a runtime probe -- checked once at
    ModelGateway construction. Raises ValueError for an unrecognized
    model_backend_id rather than silently defaulting to False, since a
    silent False could cause a vision-capable model to be treated as
    text-only without anyone noticing."""
    base = _bare_model_id(model_backend_id)
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
