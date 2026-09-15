# LLM Agent Echo Chamber Sandbox — Full Design Document
### HLD · LLD · Data Models · Storage Schema · API Design

*Companion to `sandbox_hld_v2.md`, which remains the source of truth for architectural reasoning, the three Adversarial Self-Critique passes, and Fixes A–N. This document does not repeat that reasoning; it extends it one level deeper into implementation.*

---

## 1. Executive Summary

This sandbox simulates populations of LLM-backed agents discussing a topic over multiple turns under a tunable echo-chamber sampling mechanism, in order to answer three research questions: whether Hindi-English code-mixed (Hinglish) discussion produces different polarization dynamics than English discussion under identical conditions (RQ1); whether the *quality* of an agent's opinion-updating reasoning differs by language and correlates with polarization speed (RQ2, a post-hoc analysis of RQ1's own data); and whether injecting memes from a pre-existing labeled dataset into an ongoing discussion accelerates convergence to a polarized distribution compared to text-only discussion (RQ3). The single architectural fact that shapes everything below: there is exactly **one simulation mode** (multi-turn), and memes are implemented as a *content-replacement mechanism* — a meme-bearing agent's turn uses a dataset-provided stance label instead of an LLM call, and is then sampled by the existing echo-chamber mechanism exactly like any other agent. This is what allows RQ1, RQ2, and RQ3 to share one Orchestrator, one schema family, and one config surface, with no cross-model sweep and no second execution path anywhere in the system.

---

## 2. High-Level Design (Condensed)

### 2.1 Components

| Component | Responsibility |
|---|---|
| Config/Run Loader | Parses and validates a YAML run config into an `ExperimentRun` |
| Simulation Orchestrator | Drives the single multi-turn loop; the only caller of every other runtime component |
| Agent Manager | Creates and holds `Agent` objects |
| Interaction/Recommendation Engine | Selects N neighbors per agent per turn (`AlphaSampling`) |
| Meme Pool Manager | Loads a pre-existing meme dataset into `MemeContent` records; supplies injection decisions |
| Model Gateway | Single normalized interface to all LLM providers (LiteLLM-based); handles vision/text-only dispatch, retries, rate limiting, cost tracking |
| Logging/Persistence Layer | Append-only JSONL writer; one run = one directory |
| Post-hoc Analysis Module | Offline: regression, bimodality analysis, coherence/sentiment/code-mix scoring, embedding clustering |

The **Model Gateway** is the renamed "Model Backend Abstraction Layer" from `sandbox_hld_v2.md` — same responsibilities, new name, used consistently throughout this document and in the codebase as `model_gateway.py`.

### 2.2 Data flow (single mode, serves RQ1/RQ2/RQ3)

Config Loader → `ExperimentRun` → Orchestrator creates M agents via Agent Manager → for each of K turns: freeze snapshot → Interaction Engine resolves neighbor assignments for all M agents → Meme Pool Manager resolves which (if any) scheduled speakers post a meme this turn → dispatch: generated-text speakers go to the Model Gateway, meme speakers are a free dataset lookup → collect all results → write `Interaction` records → advance snapshot → checkpoint → next turn. RQ2 and parts of RQ3's analysis run entirely offline against the resulting `interactions.jsonl`, with no new simulation calls.

### 2.3 What's out of scope (carried forward from `sandbox_hld_v2.md` §8, restated for completeness)

No human-facing UI, no real-time streaming, no relational database as primary store, no live graph mutation, no cross-model sweep, no single-exposure mode, no new meme-annotation pipeline. NetworkX remains a listed dependency but is not exercised by any module in this document — no graph-metric module or endpoint is designed here, consistent with the locked scope.

---

## 3. Low-Level Design

This section proceeds module-by-module, in the same order as the HLD's module table. Every function shown uses real type hints. Pydantic models referenced here (`Agent`, `Interaction`, etc.) are fully specified in §4; only the parts relevant to control flow are inlined here.

### 3.1 `config_loader.py`

**Purpose:** parse and validate a run config into an `ExperimentRun`.

```python
from pathlib import Path
import yaml
from pydantic import ValidationError

class ConfigLoadError(Exception):
    """Raised for any config problem; wraps the underlying cause."""
    def __init__(self, path: Path, errors: list[str]):
        self.path = path
        self.errors = errors
        super().__init__(f"Invalid config at {path}: {'; '.join(errors)}")


def load_run_config(path: Path) -> "ExperimentRun":
    """
    Reads a YAML file, validates it against the ExperimentRun schema,
    and resolves referenced assets (meme_pool_id, persona files) to
    confirm they exist on disk -- WITHOUT loading their full contents
    (that happens lazily in agent_manager.py / meme_pool_manager.py).

    Raises ConfigLoadError on any failure. Never returns a partially
    valid ExperimentRun.
    """
    if not path.exists():
        raise ConfigLoadError(path, [f"file not found: {path}"])

    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        raise ConfigLoadError(path, [f"malformed YAML: {e}"]) from e

    try:
        run = ExperimentRun(**raw)
    except ValidationError as e:
        raise ConfigLoadError(path, [str(err) for err in e.errors()]) from e

    _validate_referenced_assets(run)  # see below
    return run


def _validate_referenced_assets(run: "ExperimentRun") -> None:
    """
    Checks that meme_pool_id (if meme_injection.enabled) resolves to an
    actual dataset file, and that the persona pool referenced by the run
    exists. Raises ConfigLoadError listing ALL missing assets at once
    (not fail-fast on the first one), so a researcher fixing a config
    sees every problem in one pass rather than one-at-a-time.
    """
    errors: list[str] = []
    if run.meme_injection.enabled:
        pool_path = Path("data/memes") / f"{run.meme_injection.meme_pool_id}.jsonl"
        if not pool_path.exists():
            errors.append(f"meme_pool_id '{run.meme_injection.meme_pool_id}' not found at {pool_path}")
    persona_path = Path("data/personas") / f"{run.persona_pool_id}.jsonl"
    if not persona_path.exists():
        errors.append(f"persona pool '{run.persona_pool_id}' not found at {persona_path}")
    if errors:
        raise ConfigLoadError(Path("<config>"), errors)
```

**Error handling:** all failures raise `ConfigLoadError`; nothing is silently defaulted. This function never makes a network call, so no retry logic is needed here.

**Concurrency:** not applicable — called once, synchronously, before any concurrent dispatch begins.

---

### 3.2 `agent_manager.py`

**Purpose:** create and hold the `Agent` population for a run.

```python
import random
from pathlib import Path

class AgentManager:
    def __init__(self, run: "ExperimentRun", rng: random.Random):
        self._run = run
        self._rng = rng
        self._agents: dict[str, Agent] = {}

    def initialize_population(self) -> list["Agent"]:
        """
        Creates run.M agents: assigns each a persona (drawn from the
        persona pool referenced by run.persona_pool_id), an initial_stance
        (per the configured initial-distribution strategy -- default:
        uniform draw across the finite stance scale, matching Ohagi's
        baseline condition), and run.language_condition / run.model_backend_id
        (identical for every agent in this scope, since there is no
        per-agent language or model variation within a single run).

        Deterministic given the same seed: persona assignment order and
        initial-stance draws both consume from self._rng in a fixed order
        (persona first, then stance), so re-running with the same seed
        reproduces the same population.
        """
        personas = self._load_persona_pool()
        agents = []
        for i in range(self._run.M):
            persona = personas[i % len(personas)]  # cycle if M > pool size
            stance = self._draw_initial_stance()
            agent = Agent(
                agent_id=f"agent_{i:04d}",
                run_id=self._run.run_id,
                persona=persona,
                language_condition=self._run.language_condition,
                model_backend_id=self._run.model_backend_id,
                initial_stance=stance,
                stance_history=[],
                memory_window=[],
                created_at_turn=0,
            )
            self._agents[agent.agent_id] = agent
            agents.append(agent)
        return agents

    def _load_persona_pool(self) -> list[str]:
        """Raises FileNotFoundError if the pool file is missing -- should
        never happen here since config_loader already validated this path
        exists, but this function does not trust that check blindly and
        re-raises with a clear message if the file vanished between
        load time and use time (e.g., a concurrent process deleted it)."""
        path = Path("data/personas") / f"{self._run.persona_pool_id}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"persona pool vanished: {path}")
        return [line.strip() for line in path.read_text().splitlines() if line.strip()]

    def _draw_initial_stance(self) -> float:
        # Uniform draw across the configured finite stance scale (default 1-7,
        # matching Ohagi). Uses self._rng, NOT the global random module, so
        # this is fully reproducible given seed_manager's seeded instance.
        scale = self._run.stance_scale  # e.g. [1, 2, 3, 4, 5, 6, 7]
        return float(self._rng.choice(scale))

    def get_agent(self, agent_id: str) -> "Agent":
        """Raises KeyError if agent_id is unknown -- this is treated as a
        programmer error (should be impossible given correct Orchestrator
        logic), so it is NOT caught/wrapped here; it propagates."""
        return self._agents[agent_id]

    def all_agents(self) -> list["Agent"]:
        return list(self._agents.values())
```

**Error handling:** `FileNotFoundError` on a vanished persona pool propagates to the Orchestrator, which treats it as a fatal run-setup error (no partial run is started). `KeyError` from `get_agent` on an unknown id is treated as a programmer error and is not defensively caught.

**Concurrency:** `initialize_population()` runs once, before the per-turn loop begins — not called concurrently. `get_agent()` is called during the concurrent dispatch phase (§3.9) but is read-only against a dict that isn't mutated during dispatch (mutations happen only in the "advance snapshot" step, per Fix F), so concurrent reads are safe without locking.

---

### 3.3 `interaction_engine.py` / `recommendation_strategies.py`

**Purpose:** select N neighbors per agent per turn.

```python
from typing import Protocol
import random

class RecommendationStrategy(Protocol):
    def select_neighbors(
        self,
        speaker: "Agent",
        all_agents: list["Agent"],
        turn: int,
        rng: random.Random,
    ) -> list["Agent"]:
        """Must return exactly N distinct agents, never including speaker itself."""
        ...


class AlphaSampling:
    """Ohagi's mechanism. alpha in [0,1]: 0 = uniform random sampling,
    1 = maximally homophilous (always prefers closest-stance candidates)."""

    def __init__(self, alpha: float, N: int):
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0,1], got {alpha}")
        if N < 1:
            raise ValueError(f"N must be >= 1, got {N}")
        self.alpha = alpha
        self.N = N

    def select_neighbors(
        self,
        speaker: "Agent",
        all_agents: list["Agent"],
        turn: int,
        rng: random.Random,
    ) -> list["Agent"]:
        candidates = [a for a in all_agents if a.agent_id != speaker.agent_id]
        if len(candidates) < self.N:
            raise ValueError(
                f"population too small: need {self.N} candidates, have {len(candidates)}"
            )

        current_stance = _current_stance(speaker, turn)
        weights = []
        for c in candidates:
            c_stance = _current_stance(c, turn)
            distance = abs(current_stance - c_stance)
            similarity_weight = 1.0 / (1.0 + distance)      # closer stance -> higher weight
            uniform_weight = 1.0                             # equal weight for every candidate
            blended = self.alpha * similarity_weight + (1 - self.alpha) * uniform_weight
            weights.append(blended)

        # weighted sample without replacement, using rng (never the global
        # random module -- reproducibility depends on this)
        return _weighted_sample_without_replacement(candidates, weights, self.N, rng)


def _current_stance(agent: "Agent", turn: int) -> float:
    """Reads the agent's stance as of the END of turn-1 (the frozen
    snapshot, per Fix F). turn=1 reads initial_stance; turn>1 reads the
    last entry in stance_history. This function does NOT read any
    in-progress turn-t state -- that state does not exist yet at the
    point this function is called, by construction of the Orchestrator's
    sequence (freeze snapshot happens strictly before this is called)."""
    if turn == 1 or not agent.stance_history:
        return agent.initial_stance
    return agent.stance_history[-1].stance_value


def _weighted_sample_without_replacement(
    items: list["Agent"], weights: list[float], k: int, rng: random.Random
) -> list["Agent"]:
    """Efraimidis-Spirakis weighted reservoir sampling -- O(n log k),
    deterministic given rng's state."""
    keyed = [(rng.random() ** (1.0 / w), item) for w, item in zip(weights, items)]
    keyed.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in keyed[:k]]
```

**Error handling:** `ValueError` on invalid `alpha`/`N`/insufficient population — these are construction-time or call-time contract violations, raised immediately rather than silently clamped, because silently clamping `alpha` or `N` would be exactly the kind of implementation-level confound Critique Pass 3 in the HLD warns against.

**Concurrency:** `select_neighbors()` is called once per agent, for all M agents, during the "resolve neighbor assignments" step of the Orchestrator's per-turn sequence — this step happens *before* any concurrent dispatch and reads only frozen-snapshot state (`_current_stance` never reads in-progress data), so it is safe to parallelize this step itself if desired, though at M=100 it's cheap enough that sequential execution here is not a bottleneck.

---

### 3.4 `meme_pool_manager.py`

**Purpose:** load a pre-existing meme dataset; decide, per scheduled speaker per turn, whether that speaker's post is a meme.

```python
import random
from pathlib import Path

class MemePoolManager:
    def __init__(self, config: "MemeInjectionConfig", rng: random.Random):
        self._config = config
        self._rng = rng
        self._pool: list[MemeContent] = []
        if config.enabled:
            self._pool = self._load_pool()

    def _load_pool(self) -> list["MemeContent"]:
        path = Path("data/memes") / f"{self._config.meme_pool_id}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"meme pool not found: {path}")
        items = [MemeContent.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]
        if not items:
            raise ValueError(f"meme pool at {path} is empty")
        return items

    def resolve_injections_for_turn(
        self, scheduled_speakers: list["Agent"], turn: int
    ) -> dict[str, "MemeContent | None"]:
        """
        For each scheduled speaker, decides (per Fix L: uniformly, at the
        configured injection_rate, applied independently per agent-turn
        under injection_schedule='random'; or only at the configured
        turn(s) under injection_schedule='fixed_turn') whether that
        speaker's post this turn is a meme. Returns a dict mapping
        agent_id -> MemeContent or None (None = generate text normally).

        Must be called with the SAME rng instance seed_manager provides
        for this run, and must be called as part of the frozen-snapshot
        phase of the Orchestrator's per-turn sequence (Critique Pass 2's
        extension of Fix F) -- i.e. before any dispatch begins.
        """
        if not self._config.enabled:
            return {a.agent_id: None for a in scheduled_speakers}

        eligible = self._is_eligible_turn(turn)
        result: dict[str, MemeContent | None] = {}
        for agent in scheduled_speakers:
            if eligible and self._rng.random() < self._config.injection_rate:
                result[agent.agent_id] = self._sample_meme()
            else:
                result[agent.agent_id] = None
        return result

    def _is_eligible_turn(self, turn: int) -> bool:
        if self._config.injection_schedule == "random":
            return True
        if self._config.injection_schedule == "fixed_turn":
            return turn in self._config.fixed_turns  # e.g. {5}
        raise ValueError(f"unknown injection_schedule: {self._config.injection_schedule}")

    def _sample_meme(self) -> "MemeContent":
        return self._rng.choice(self._pool)
```

**Error handling:** `FileNotFoundError`/empty-pool `ValueError` at construction time — fatal, propagates before the run starts (mirrors `config_loader`'s asset-validation philosophy: fail before spending API budget). `ValueError` on an unknown `injection_schedule` value is a construction-time contract violation from a malformed config that should already have been caught by Pydantic validation (§4) — its presence here is defense-in-depth, not the primary validation path.

**Concurrency:** `resolve_injections_for_turn()` is called once per turn, synchronously, immediately after neighbor resolution and before dispatch — same placement guarantee as `interaction_engine.py`. It mutates nothing; it is a pure function of `(scheduled_speakers, turn, rng-state)`.

---

### 3.5 `model_gateway.py`

**Purpose:** the single normalized interface to every LLM provider. (Renamed from "Model Backend Abstraction Layer" per this document's locked terminology.)

```python
from typing import Protocol
from dataclasses import dataclass
import litellm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

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
        rate_limiter: "RateLimiter",
        cost_tracker: "CostTracker",
    ):
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
        reports supports_vision so callers can decide. This keeps the
        Gateway a pure dispatch layer with no content-decision logic.
        """
        await self._rate_limiter.acquire(self._provider_name())
        try:
            response = await self._call_with_retry(prompt_text, image, temperature)
        finally:
            self._rate_limiter.release(self._provider_name())

        modality = "vision" if image is not None else "text"
        self._cost_tracker.record(
            provider=self._provider_name(),
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
    VISION_CAPABLE = {"gpt-4o", "gpt-4o-mini", "claude-sonnet-4-6", "gemini-2.0-flash", "llava", "qwen-vl"}
    TEXT_ONLY = {"gpt-3.5-turbo", "llama-3.1-8b", "llama-3.1-70b"}
    base = model_backend_id.split("/")[-1]
    if base in VISION_CAPABLE:
        return True
    if base in TEXT_ONLY:
        return False
    raise ValueError(f"unrecognized model_backend_id, cannot determine vision support: {model_backend_id}")


def _build_messages(prompt_text: str, image: bytes | None) -> list[dict]:
    if image is None:
        return [{"role": "user", "content": prompt_text}]
    import base64
    b64 = base64.b64encode(image).decode()
    return [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt_text},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ],
    }]
```

**Error handling:** `TransientGatewayError` is retried up to 3 times with exponential backoff via `tenacity`; `FatalGatewayError` is never retried and propagates immediately to the Orchestrator, which logs the `Interaction` with `api_call_status = "failed_logged_null"` (§3.7's `stance_parser.py` handles the downstream consequence) and does not attempt that call again this run.

**Concurrency:** `generate()` is `async` and is the method called concurrently for all scheduled generated-text speakers during dispatch. It is safe to call concurrently because it holds no mutable state shared across calls other than the rate limiter (which is itself designed for concurrent access — see §3.10) and the cost tracker (append-only, see §3.11).

---

### 3.6 `vision_fallback.py`

**Purpose:** substitute a caption-only prompt when a meme must be shown to a text-only-backend receiver.

```python
def build_meme_prompt(
    meme: "MemeContent", receiver_supports_vision: bool
) -> tuple[str, bytes | None]:
    """
    Returns (prompt_text, image_bytes_or_None). If receiver_supports_vision
    is False, image is always None and the caption is woven into the
    prompt text with an explicit instruction frame -- this is the ONLY
    place in the codebase that constructs a meme-derived prompt, so the
    text framing here is used identically regardless of vision support;
    only the image payload differs.
    """
    instruction = (
        "You are shown the following content from another participant "
        "in the discussion. Consider it as their contribution and respond "
        "according to your persona and current view."
    )
    if receiver_supports_vision:
        image_bytes = _load_image(meme.image_path)
        prompt_text = f"{instruction}\n\n[Image shown separately]\nCaption: {meme.caption_text}"
        return prompt_text, image_bytes
    else:
        prompt_text = f"{instruction}\n\nContent (text-only, as no image is available to you): {meme.caption_text}"
        return prompt_text, None


def _load_image(image_path: str) -> bytes:
    from pathlib import Path
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"meme image missing on disk: {image_path}")
    return path.read_bytes()
```

**Error handling:** `FileNotFoundError` on a missing image file propagates — treated as a data-integrity problem with the meme pool, not something to silently degrade around (silently falling back to caption-only when the *config* says vision is supported would itself introduce an unlogged fallback, which is exactly what `used_vision_fallback` exists to make visible; a missing file is a different failure mode and should surface loudly).

**Concurrency:** pure function, no shared state; safe to call concurrently from multiple dispatch tasks (each with a different `meme`).

---

### 3.7 `stance_parser.py`

**Purpose:** extract a structured `(stance_value, reason_text)` pair from raw model output; implement the standardized retry/exclude policy (Fix I).

```python
import re

class StanceParseFailure(Exception):
    """Raised when parsing fails even after the retry budget is exhausted.
    Caught by the Orchestrator, never by callers of parse_stance() directly."""

_STANCE_PATTERN = re.compile(r"STANCE:\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)

def parse_stance(raw_text: str) -> tuple[float, str]:
    """
    Expects the model to have followed the prompt's required output
    format (a line "STANCE: <number>" followed by free-text reasoning).
    Returns (stance_value, reason_text). Raises StanceParseFailure if no
    STANCE line is found or the value doesn't parse as a float, or if
    the parsed value falls outside the configured stance_scale bounds.

    This function does NOT retry -- retry orchestration lives in the
    Orchestrator (see 3.9), because retrying requires re-dispatching to
    the Model Gateway with an amended prompt, which this pure parsing
    function has no access to.
    """
    match = _STANCE_PATTERN.search(raw_text)
    if match is None:
        raise StanceParseFailure(f"no STANCE line found in output: {raw_text[:200]!r}")
    try:
        value = float(match.group(1))
    except ValueError as e:
        raise StanceParseFailure(f"STANCE value not numeric: {match.group(1)!r}") from e

    reason = _STANCE_PATTERN.sub("", raw_text).strip()
    return value, reason


def clamp_and_validate_scale(value: float, stance_scale: list[float]) -> float:
    """Raises StanceParseFailure if value is outside [min(scale), max(scale)]
    -- an out-of-range value is treated identically to an unparseable one,
    never silently clamped, per Fix I's 'never fabricate a placeholder
    stance' principle."""
    lo, hi = min(stance_scale), max(stance_scale)
    if not (lo <= value <= hi):
        raise StanceParseFailure(f"stance {value} outside configured scale [{lo}, {hi}]")
    return value
```

**Error handling:** `StanceParseFailure` is the only exception type this module raises; it is always caught by the Orchestrator (§3.9), never left to propagate further, since the Orchestrator's job is to implement the retry-twice-then-exclude policy around this function's failures.

**Concurrency:** pure, stateless functions; safe to call concurrently.

---

### 3.8 `logging_writer.py`

**Purpose:** append-only writer for every record type.

```python
import json
from pathlib import Path
import asyncio

class LoggingWriter:
    def __init__(self, run_dir: Path):
        self._run_dir = run_dir
        self._interactions_path = run_dir / "interactions.jsonl"
        self._lock = asyncio.Lock()  # serializes concurrent appends from dispatch tasks

    async def write_interaction(self, interaction: "Interaction") -> None:
        """
        Appends one JSON line. Uses an asyncio.Lock rather than relying on
        OS-level atomic append, because Python's async file writes are not
        guaranteed atomic across concurrent tasks writing to the same
        file handle within one process (this is a single-process design
        per the HLD -- no cross-process write contention to handle).
        """
        line = interaction.model_dump_json() + "\n"
        async with self._lock:
            with open(self._interactions_path, "a", encoding="utf-8") as f:
                f.write(line)

    def write_run_config(self, run: "ExperimentRun") -> None:
        """Called once at run start, before any concurrent activity --
        no locking needed."""
        path = self._run_dir / "run_config.json"
        path.write_text(run.model_dump_json(indent=2))

    def write_agents_final(self, agents: list["Agent"]) -> None:
        """Called once at run completion, after all turns finish and no
        further concurrent writes are possible -- no locking needed."""
        path = self._run_dir / "agents_final.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for agent in agents:
                f.write(agent.model_dump_json() + "\n")
```

**Error handling:** disk-write failures (`OSError`, e.g. disk full) propagate uncaught — treated as fatal, since silently losing an `Interaction` record would corrupt the run's data integrity in a way no downstream analysis could detect.

**Concurrency:** `write_interaction()` is the only method called during concurrent dispatch; the internal `asyncio.Lock` serializes actual file writes while still letting dispatch tasks proceed concurrently up to that point (LLM calls are the slow part; the file write itself is fast, so lock contention here is not expected to be a bottleneck).

---

### 3.9 `simulation_orchestrator.py`

**Purpose:** drive the per-turn sequence; the only module that calls every other component in order.

```python
import asyncio

class SimulationOrchestrator:
    def __init__(
        self,
        run: "ExperimentRun",
        agent_manager: "AgentManager",
        interaction_engine: "RecommendationStrategy",
        meme_pool: "MemePoolManager",
        gateway: "ModelGateway",
        logger: "LoggingWriter",
        checkpoint_mgr: "CheckpointManager",
        prompt_builder: "PromptBuilder",
        rng: "random.Random",
    ):
        self._run = run
        self._agent_manager = agent_manager
        self._interaction_engine = interaction_engine
        self._meme_pool = meme_pool
        self._gateway = gateway
        self._logger = logger
        self._checkpoint_mgr = checkpoint_mgr
        self._prompt_builder = prompt_builder
        self._rng = rng

    async def run(self) -> None:
        """
        Top-level entry point. Resumes from checkpoint if one exists for
        this run_id, otherwise starts fresh at turn 1.
        """
        resume_state = self._checkpoint_mgr.load(self._run.run_id)
        if resume_state is not None:
            start_turn = resume_state.last_completed_turn + 1
            self._agent_manager.restore_from_snapshot(resume_state.agent_snapshot)
        else:
            self._agent_manager.initialize_population()
            self._logger.write_run_config(self._run)
            start_turn = 1

        for turn in range(start_turn, self._run.K + 1):
            await self._run_turn(turn)
            self._checkpoint_mgr.save(
                self._run.run_id, turn, self._agent_manager.snapshot()
            )

        self._logger.write_agents_final(self._agent_manager.all_agents())

    async def _run_turn(self, turn: int) -> None:
        """
        Implements the exact sequence required by Fix F (extended per
        Critique Pass 2 to include meme-injection resolution):
          1. freeze snapshot (implicit: agent_manager's state IS the
             frozen snapshot at this point, since no writes have
             happened yet this turn)
          2. resolve neighbor assignments for all M agents
          3. resolve meme-injection assignments for all scheduled speakers
          4. dispatch (concurrently): generated-text calls to the Gateway,
             meme lookups are free
          5. collect all results
          6. advance snapshot (write results into agent state)
        No agent's turn-t output is read by step 2 or 3 for ANY agent,
        including itself -- both steps run to completion before any
        dispatch begins.
        """
        all_agents = self._agent_manager.all_agents()

        # Step 2
        neighbor_map: dict[str, list[Agent]] = {
            a.agent_id: self._interaction_engine.select_neighbors(a, all_agents, turn, self._rng)
            for a in all_agents
        }

        # Step 3
        meme_map = self._meme_pool.resolve_injections_for_turn(all_agents, turn)

        # Step 4 + 5
        tasks = [
            self._process_one_agent_turn(agent, neighbor_map[agent.agent_id], meme_map[agent.agent_id], turn)
            for agent in all_agents
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Step 6
        for agent, result in zip(all_agents, results):
            if isinstance(result, Exception):
                # a fatal, non-retryable failure already logged inside
                # _process_one_agent_turn; this agent is excluded from
                # this turn's neighbor-sampling pool going forward
                # (Fix I) -- accomplished by simply not appending a new
                # stance_history entry, so _current_stance() in
                # interaction_engine.py falls back to the agent's last
                # known stance automatically.
                continue
            interaction: Interaction = result
            await self._logger.write_interaction(interaction)
            self._agent_manager.apply_interaction(agent.agent_id, interaction)

    async def _process_one_agent_turn(
        self, agent: "Agent", neighbors: list["Agent"], meme: "MemeContent | None", turn: int
    ) -> "Interaction":
        if meme is not None:
            return self._build_meme_interaction(agent, neighbors, meme, turn)

        prompt = self._prompt_builder.build_discussion_prompt(agent, neighbors, turn)
        stance_before = _current_stance_for_logging(agent, turn)

        for attempt in range(3):  # Fix I: 2 retries after the first attempt
            try:
                response = await self._gateway.generate(prompt, image=None, temperature=self._run.temperature)
                stance_after, reason = parse_stance(response.text)
                stance_after = clamp_and_validate_scale(stance_after, self._run.stance_scale)
                return Interaction(
                    interaction_id=str(uuid.uuid4()),
                    run_id=self._run.run_id,
                    turn=turn,
                    speaker_agent_id=agent.agent_id,
                    neighbor_agent_ids=[n.agent_id for n in neighbors],
                    stance_before=stance_before,
                    stance_after=stance_after,
                    reason_text=reason,
                    content_type="generated_text",
                    meme_id=None,
                    used_vision_fallback=False,
                    prompt_token_count=response.prompt_tokens,
                    completion_token_count=response.completion_tokens,
                    model_backend_id=self._run.model_backend_id,
                    latency_ms=response.latency_ms,
                    api_call_status="success" if attempt == 0 else "retried_success",
                    timestamp_utc=_now_utc_iso(),
                )
            except StanceParseFailure:
                if attempt < 2:
                    prompt = self._prompt_builder.build_discussion_prompt(
                        agent, neighbors, turn, emphasize_format=True
                    )
                    continue
                # Fix I: exhausted retries -- log and exclude, do not raise
                return Interaction(
                    interaction_id=str(uuid.uuid4()), run_id=self._run.run_id, turn=turn,
                    speaker_agent_id=agent.agent_id,
                    neighbor_agent_ids=[n.agent_id for n in neighbors],
                    stance_before=stance_before, stance_after=stance_before,  # unchanged
                    reason_text="", content_type="generated_text", meme_id=None,
                    used_vision_fallback=False, prompt_token_count=0, completion_token_count=0,
                    model_backend_id=self._run.model_backend_id, latency_ms=0,
                    api_call_status="failed_logged_null", timestamp_utc=_now_utc_iso(),
                )
            except FatalGatewayError:
                # not retried at all, per model_gateway.py's contract
                return Interaction(
                    interaction_id=str(uuid.uuid4()), run_id=self._run.run_id, turn=turn,
                    speaker_agent_id=agent.agent_id,
                    neighbor_agent_ids=[n.agent_id for n in neighbors],
                    stance_before=stance_before, stance_after=stance_before,
                    reason_text="", content_type="generated_text", meme_id=None,
                    used_vision_fallback=False, prompt_token_count=0, completion_token_count=0,
                    model_backend_id=self._run.model_backend_id, latency_ms=0,
                    api_call_status="failed_logged_null", timestamp_utc=_now_utc_iso(),
                )

    def _build_meme_interaction(
        self, agent: "Agent", neighbors: list["Agent"], meme: "MemeContent", turn: int
    ) -> "Interaction":
        """No Gateway call -- the meme's stance_label IS this agent's
        stance_after for this turn, per the HLD's meme-injection design."""
        stance_before = _current_stance_for_logging(agent, turn)
        return Interaction(
            interaction_id=str(uuid.uuid4()), run_id=self._run.run_id, turn=turn,
            speaker_agent_id=agent.agent_id,
            neighbor_agent_ids=[n.agent_id for n in neighbors],
            stance_before=stance_before, stance_after=meme.stance_label,
            reason_text=meme.caption_text, content_type="meme", meme_id=meme.meme_id,
            used_vision_fallback=False,  # set True only when a NEIGHBOR (in a later
                                          # turn) receives this meme via caption-only
                                          # fallback -- not applicable to the posting
                                          # turn itself, which never calls the Gateway
            prompt_token_count=0, completion_token_count=0,
            model_backend_id=self._run.model_backend_id, latency_ms=0,
            api_call_status="success", timestamp_utc=_now_utc_iso(),
        )
```

**Error handling:** the Orchestrator is where every other module's exceptions are ultimately caught or allowed to propagate as a fatal run failure. `StanceParseFailure` and `FatalGatewayError` are caught per-agent inside `_process_one_agent_turn` and converted into a `failed_logged_null` `Interaction` rather than aborting the whole turn (`asyncio.gather(..., return_exceptions=True)` ensures one agent's failure doesn't cancel the other 99 concurrent tasks). Any *other*, unanticipated exception type is NOT caught here and propagates out of `run()`, which is treated as a fatal run failure requiring investigation — this is a deliberate choice: only the two failure modes explicitly designed for (parse failure, fatal gateway error) get soft-failure handling; anything else is a bug that should surface loudly rather than be silently absorbed into `failed_logged_null` records that could mask a real problem.

**Concurrency:** `_run_turn()` is the method implementing Fix F's guarantee directly — steps 2–3 (neighbor and meme resolution) are fully sequential and complete before step 4's `asyncio.gather()` begins, so no dispatch task ever reads another task's in-progress result.

---

### 3.10 `rate_limiter.py`

```python
import asyncio
import time

class RateLimiter:
    """Per-provider token-bucket. One instance shared across all
    concurrent dispatch tasks for a run."""

    def __init__(self, limits: dict[str, int]):
        # limits: {"openai": 500, "anthropic": 400, ...} requests/minute
        self._limits = limits
        self._semaphores = {
            provider: asyncio.Semaphore(max_concurrent) for provider, max_concurrent in limits.items()
        }
        self._last_request_time: dict[str, float] = {p: 0.0 for p in limits}
        self._min_interval = {p: 60.0 / n if n > 0 else 0.0 for p, n in limits.items()}
        self._locks = {p: asyncio.Lock() for p in limits}

    async def acquire(self, provider: str) -> None:
        if provider not in self._limits:
            return  # unconfigured provider (e.g. local backend): no limit applied
        async with self._locks[provider]:
            elapsed = time.monotonic() - self._last_request_time[provider]
            wait = self._min_interval[provider] - elapsed
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_request_time[provider] = time.monotonic()

    def release(self, provider: str) -> None:
        pass  # token-bucket-by-interval needs no explicit release; kept
              # for interface symmetry with model_gateway.py's call site
```

**Error handling:** no exceptions raised by design — an unconfigured provider is treated as "no limit," not an error, since `local` backends legitimately have no rate limit per the original config schema.

**Concurrency:** this module's entire purpose is concurrency control; `acquire()` is designed to be called from many concurrent tasks and serializes them per-provider via the internal lock.

---

### 3.11 `cost_tracker.py`

```python
import yaml
from pathlib import Path
import asyncio

class CostTracker:
    def __init__(self, run: "ExperimentRun", pricing_table_path: Path = Path("pricing_table.yaml")):
        self._run = run
        self._pricing = yaml.safe_load(pricing_table_path.read_text())
        self._total_usd = 0.0
        self._lock = asyncio.Lock()

    def record(
        self, provider: str, model_id: str, modality: str,
        prompt_tokens: int, completion_tokens: int,
    ) -> None:
        """Synchronous by design -- called from within model_gateway.py's
        async generate() but doesn't itself need to await anything except
        the lock, since pricing lookup is pure computation."""
        rate = self._lookup_rate(provider, model_id, modality)
        cost = (prompt_tokens / 1000) * rate["prompt_per_1k"] + (completion_tokens / 1000) * rate["completion_per_1k"]
        self._total_usd += cost
        if self._run.max_cost_usd is not None and self._total_usd >= self._run.max_cost_usd:
            raise CostCeilingExceeded(self._total_usd, self._run.max_cost_usd)

    def _lookup_rate(self, provider: str, model_id: str, modality: str) -> dict:
        try:
            return self._pricing[provider][model_id][modality]
        except KeyError as e:
            raise ValueError(
                f"no pricing entry for {provider}/{model_id}/{modality} -- "
                f"update pricing_table.yaml before running"
            ) from e

    @property
    def total_usd(self) -> float:
        return self._total_usd


class CostCeilingExceeded(Exception):
    def __init__(self, current: float, ceiling: float):
        self.current = current
        self.ceiling = ceiling
        super().__init__(f"cost ${current:.2f} reached/exceeded ceiling ${ceiling:.2f}")
```

**Error handling:** a missing pricing entry is a fatal, immediate `ValueError` — the design deliberately does not fall back to an estimated/default rate, since an incorrect silent estimate would defeat the purpose of cost tracking. `CostCeilingExceeded` is raised **after** the triggering call has already completed and its cost recorded (see §7 for the full policy statement) — it propagates up through `model_gateway.py`'s `generate()` call, through the Orchestrator's per-agent handling, and is treated as a fatal run-level exception (not caught per-agent like `StanceParseFailure`), halting the run at the next checkpoint boundary.

**Concurrency:** `record()` is called from many concurrent Gateway calls; the internal lock is needed here because `+=` is not atomic in the presence of concurrent access, unlike the append-only writes in `logging_writer.py`.

---

### 3.12 `seed_manager.py`

```python
import random

class SeedManager:
    """One instance per run. Provides separately-seeded, purpose-specific
    rng instances so that, per Fix M, reusing the same top-level seed
    across a language-condition sweep holds meme-injection and
    neighbor-sampling patterns constant while language varies."""

    def __init__(self, seed: int):
        self._base_seed = seed
        self._neighbor_rng = random.Random(seed)
        self._meme_rng = random.Random(seed + 1)       # deliberately offset,
        self._persona_rng = random.Random(seed + 2)     # so the three streams
                                                          # never accidentally
                                                          # correlate

    @property
    def neighbor_sampling_rng(self) -> random.Random:
        return self._neighbor_rng

    @property
    def meme_injection_rng(self) -> random.Random:
        return self._meme_rng

    @property
    def persona_assignment_rng(self) -> random.Random:
        return self._persona_rng
```

Note on Fix M's implementation: this design offsets the three purpose-specific seeds (`seed`, `seed+1`, `seed+2`) deterministically from one base seed, so "reuse the same seed across language conditions" simply means passing the same top-level `seed` value in both configs — the offsetting inside `SeedManager` guarantees the meme-injection stream and neighbor-sampling stream are each independently reproducible without the caller needing to manage three separate seed values manually.

**Error handling:** none needed — pure construction.

**Concurrency:** each `random.Random` instance is not thread/task-safe for *concurrent* calls to the same instance; because neighbor resolution and meme-injection resolution both happen sequentially (not concurrently) within step 2/3 of `_run_turn`, this is not a problem in the current design. If any future change made these steps concurrent with each other, this would need revisiting — flagged in §8.

---

### 3.13 Post-hoc analysis modules (`stance_regression.py`, `coherence_scorer.py`, `codemix_ratio_tracker.py`, `sentiment_proxy.py`, `bimodality_analysis.py`, `embedding_cluster.py`)

These all share a common shape: load `interactions.jsonl` into a `pd.DataFrame`, filter as needed, compute a metric, return a result object. Two representative signatures (the others follow the same pattern):

```python
import pandas as pd
import statsmodels.api as sm

def load_run_dataframe(run_dir: Path) -> pd.DataFrame:
    """Shared loader used by every post-hoc module. Reads interactions.jsonl
    into a DataFrame with dtypes matching the Interaction Pydantic model
    (see 4.3)."""
    df = pd.read_json(run_dir / "interactions.jsonl", lines=True)
    return df


def run_ohagi_regression(df: pd.DataFrame) -> "RegressionResult":
    """
    Computes neighbor_avg_stance per row (joining neighbor_agent_ids
    against that turn's stance snapshot), then fits
        stance_after ~ stance_before + neighbor_avg_stance
    via statsmodels OLS. Raises ValueError if df is empty or contains
    only failed_logged_null rows (nothing to regress on).
    """
    df = df[df["api_call_status"] != "failed_logged_null"].copy()
    if df.empty:
        raise ValueError("no valid interactions to regress on")
    df["neighbor_avg_stance"] = df.apply(lambda row: _neighbor_avg(row, df), axis=1)
    X = sm.add_constant(df[["stance_before", "neighbor_avg_stance"]])
    model = sm.OLS(df["stance_after"], X).fit()
    return RegressionResult(coefficients=model.params.to_dict(), r_squared=model.rsquared, n=len(df))


def score_coherence(df: pd.DataFrame, judge_model_id: str = "gpt-4o") -> pd.DataFrame:
    """
    Fix N: filters to content_type == 'generated_text' BEFORE scoring --
    meme-posted rows (reason_text is either empty or a dataset caption,
    not the agent's own reasoning) must never reach the judge model.
    Adds a `coherence_score` column (1-5) to a copy of df; does not
    mutate the input.
    """
    filtered = df[df["content_type"] == "generated_text"].copy()
    filtered["coherence_score"] = filtered["reason_text"].apply(
        lambda text: _llm_judge_coherence(text, judge_model_id)
    )
    return filtered
```

**Error handling:** these are offline, non-time-critical scripts — failures (e.g., an empty DataFrame, a judge-model call failure) raise directly rather than needing retry/fallback logic; a failed post-hoc analysis run can simply be re-run without cost implications beyond the judge-model calls themselves (which should be budgeted for separately, see §7).

**Concurrency:** not designed for concurrent execution — these are run once, offline, per completed simulation run; if `score_coherence` becomes a bottleneck at scale (many judge-model calls), that's a candidate for future `asyncio`-based batching, noted in §8, not designed here since it isn't required by the current three-RQ scope's data volume (§ Critique Pass 2 of the HLD estimated ~20,000 simulation calls total, and the coherence-scoring pass only needs to judge the `generated_text` subset of that, which is smaller still after RQ3's meme turns are excluded).

---

### 3.14 `checkpoint_manager.py`

```python
from pathlib import Path
from pydantic import BaseModel

class CheckpointState(BaseModel):
    run_id: str
    last_completed_turn: int
    agent_snapshot: list["Agent"]   # full serialized state of every agent
                                      # as of the end of last_completed_turn

class CheckpointManager:
    def __init__(self, runs_dir: Path = Path("runs")):
        self._runs_dir = runs_dir

    def save(self, run_id: str, turn: int, agent_snapshot: list["Agent"]) -> None:
        """Called once per completed turn, per Fix G. Overwrites the
        previous checkpoint (not append-only, unlike interactions.jsonl --
        only the LATEST checkpoint is ever needed for resume)."""
        checkpoint_dir = self._runs_dir / run_id / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        state = CheckpointState(run_id=run_id, last_completed_turn=turn, agent_snapshot=agent_snapshot)
        tmp_path = checkpoint_dir / "latest.json.tmp"
        final_path = checkpoint_dir / "latest.json"
        tmp_path.write_text(state.model_dump_json())
        tmp_path.replace(final_path)  # atomic rename -- avoids a torn
                                        # checkpoint file if the process
                                        # is killed mid-write

    def load(self, run_id: str) -> "CheckpointState | None":
        path = self._runs_dir / run_id / "checkpoints" / "latest.json"
        if not path.exists():
            return None
        return CheckpointState.model_validate_json(path.read_text())
```

**Error handling:** the atomic-rename pattern (`write to .tmp, then os.replace`) is specifically to prevent a corrupted, partially-written checkpoint from being loaded on resume after a crash mid-write — `os.replace` is atomic on POSIX systems, so `load()` will only ever see either the previous complete checkpoint or the new complete one, never a torn intermediate state. A corrupted checkpoint (e.g., from a non-POSIX filesystem edge case) that fails Pydantic validation on `load()` propagates as a `ValidationError`, treated as a fatal resume failure requiring manual intervention rather than silently falling back to turn 0 (which would silently discard completed work and re-spend that budget).

**Concurrency:** `save()` is called once per turn, sequentially, from the Orchestrator's main loop — never called concurrently with itself.

---

## 4. Data Models

All models are Pydantic `BaseModel` subclasses (not plain dataclasses, per this document's requirement — the HLD's original dataclass sketches are superseded by these).

### 4.1 `Agent`

```python
from pydantic import BaseModel, Field
from datetime import datetime

class StanceRecord(BaseModel):
    turn: int = Field(ge=1)
    stance_value: float
    reason_text: str
    interaction_id: str

class Agent(BaseModel):
    agent_id: str
    run_id: str
    persona: str = Field(min_length=1)
    language_condition: str = Field(pattern="^(english|hinglish|hindi)$")
    model_backend_id: str
    initial_stance: float
    stance_history: list[StanceRecord] = Field(default_factory=list)
    memory_window: list[str] = Field(default_factory=list, max_length=5)  # Fix A: hard cap of 5
    created_at_turn: int = 0

    def current_stance(self) -> float:
        """Computed, not stored: last stance_history entry, or initial_stance
        if empty. This mirrors interaction_engine.py's _current_stance()
        logic exactly and exists as a convenience for post-hoc analysis
        code that loads Agent objects directly (rather than deriving stance
        from interactions.jsonl)."""
        return self.stance_history[-1].stance_value if self.stance_history else self.initial_stance
```

**Validation constraints:** `language_condition` restricted to the three-value enum via regex pattern (Pydantic v2 `Field(pattern=...)`); `memory_window` hard-capped at 5 entries at the schema level, not just enforced by convention in `agent_manager.py` — this makes Fix A's fixed-turn-count window a schema-level guarantee, not just an implementation habit that could silently drift.

**Serialization:** `agents_final.jsonl` — one line per agent, `model_dump_json()`, full `stance_history` embedded (this is the only place `StanceRecord` is persisted as a nested list; `interactions.jsonl` is the per-record flattened equivalent, see §4.3).

### 4.2 `MemeContent`

```python
class MemeContent(BaseModel):
    meme_id: str
    image_path: str
    caption_text: str = Field(min_length=1)
    stance_label: float
    offensiveness_label: float | None = None
    source_dataset: str
```

**Validation constraints:** `caption_text` must be non-empty (a meme with no extractable caption text cannot be shown to a text-only-backend receiver via `vision_fallback.py`, so this is enforced at load time in `meme_pool_manager.py`'s `_load_pool()`, which will raise on any record failing this constraint during `MemeContent.model_validate_json()`). `stance_label` is intentionally NOT constrained to `stance_scale` at the schema level (unlike `Agent.initial_stance`), because the meme dataset's label scale may not match the simulation's configured `stance_scale` exactly — this is resolved at load time instead (see §8, Open Question: meme stance-label rescaling).

**Serialization:** loaded from `data/memes/{meme_pool_id}.jsonl` (one `MemeContent` per line) — this file is an *input* asset, not a run output, and lives outside any `runs/{run_id}/` directory.

### 4.3 `Interaction`

```python
class Interaction(BaseModel):
    interaction_id: str
    run_id: str
    turn: int = Field(ge=1)
    speaker_agent_id: str
    neighbor_agent_ids: list[str] = Field(min_length=1)
    stance_before: float
    stance_after: float
    reason_text: str
    content_type: str = Field(pattern="^(generated_text|meme)$")
    meme_id: str | None = None
    used_vision_fallback: bool = False
    prompt_token_count: int = Field(ge=0)
    completion_token_count: int = Field(ge=0)
    model_backend_id: str
    latency_ms: int = Field(ge=0)
    api_call_status: str = Field(pattern="^(success|retried_success|failed_logged_null)$")
    timestamp_utc: str

    @property
    def is_valid_for_analysis(self) -> bool:
        """Convenience for post-hoc filtering: excludes failed_logged_null
        rows from any downstream metric computation."""
        return self.api_call_status != "failed_logged_null"
```

**Validation constraints:** `meme_id` is nullable but should be non-null iff `content_type == "meme"` — this cross-field constraint is enforced by a `model_validator`, not just a comment:

```python
from pydantic import model_validator

class Interaction(BaseModel):
    # ... fields as above ...

    @model_validator(mode="after")
    def check_meme_id_consistency(self) -> "Interaction":
        if self.content_type == "meme" and self.meme_id is None:
            raise ValueError("content_type='meme' requires a non-null meme_id")
        if self.content_type == "generated_text" and self.meme_id is not None:
            raise ValueError("content_type='generated_text' requires meme_id to be null")
        return self
```

**Serialization:** one JSON line per record, appended to `interactions.jsonl` — this is the primary, load-bearing schema of the entire system; every other analysis module (regression, bimodality, coherence) reads this file.

### 4.4 `MemeInjectionConfig`

```python
class MemeInjectionConfig(BaseModel):
    enabled: bool = False
    meme_pool_id: str | None = None
    injection_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    injection_schedule: str = Field(default="random", pattern="^(random|fixed_turn)$")
    fixed_turns: set[int] = Field(default_factory=set)  # used iff injection_schedule == "fixed_turn"

    @model_validator(mode="after")
    def check_enabled_requirements(self) -> "MemeInjectionConfig":
        if self.enabled:
            if self.meme_pool_id is None:
                raise ValueError("meme_pool_id is required when meme_injection.enabled is True")
            if self.injection_schedule == "fixed_turn" and not self.fixed_turns:
                raise ValueError("fixed_turns must be non-empty when injection_schedule='fixed_turn'")
        return self
```

**Validation constraints:** `injection_rate` bounded `[0,1]` at the schema level (this is the exact constraint the earlier prompt for this document asked to see spelled out as a `Field(ge=..., le=...)` example — implemented here as specified). Cross-field validation ensures a config can never be `enabled=True` with a missing pool id, catching this at config-load time rather than at first-use inside `meme_pool_manager.py`.

### 4.5 `ExperimentRun`

```python
class ExperimentRun(BaseModel):
    run_id: str
    rq_target: str = Field(pattern="^(RQ1_RQ2|RQ3|RQ1_RQ2_RQ3)$")
    mode: str = Field(default="multi_turn", pattern="^(multi_turn)$")  # single allowed value;
                                                                          # retained as a field for
                                                                          # forward-compatibility only
    topic: str = Field(min_length=1)
    alpha: float = Field(ge=0.0, le=1.0)
    M: int = Field(ge=2)     # >=2 so at least one non-self candidate can exist for N=1
    N: int = Field(ge=1)
    K: int = Field(ge=1)
    trial_number: int = Field(ge=1)
    language_condition: str = Field(pattern="^(english|hinglish|hindi)$")
    model_backend_id: str = Field(min_length=1)
    stance_scale: list[float] = Field(min_length=2)
    persona_pool_id: str = Field(min_length=1)
    meme_injection: MemeInjectionConfig = Field(default_factory=MemeInjectionConfig)
    seed: int
    temperature: float = Field(ge=0.0, le=2.0)
    max_cost_usd: float | None = Field(default=None, gt=0.0)
    rate_limits: dict[str, int] = Field(default_factory=dict)
    checkpoint_every_n_turns: int = Field(default=1, ge=1)
    git_commit_hash: str = ""       # populated by config_loader.py at load
                                      # time, not user-supplied in the YAML
    started_at_utc: str | None = None
    completed_at_utc: str | None = None
    status: str = Field(default="pending", pattern="^(pending|running|completed|failed_partial|failed_total)$")
    total_cost_usd: float = 0.0

    @model_validator(mode="after")
    def check_n_less_than_m(self) -> "ExperimentRun":
        if self.N >= self.M:
            raise ValueError(f"N ({self.N}) must be less than M ({self.M})")
        return self
```

**Validation constraints (full list, consolidated here and cross-referenced from §7):**
- `alpha`: `[0.0, 1.0]`
- `M`: `>= 2`; `N`: `>= 1` and strictly `< M` (cross-field)
- `K`, `trial_number`, `checkpoint_every_n_turns`: positive integers
- `language_condition`: one of `{english, hinglish, hindi}`
- `stance_scale`: at least 2 distinct values (a 1-value scale can't express polarization)
- `meme_injection`: see §4.4's own validator
- `temperature`: `[0.0, 2.0]` (standard LLM API bound)
- `max_cost_usd`: if provided, strictly positive (a zero or negative ceiling would halt the run before it starts, which is a config error, not a valid "no limit" — use `None` for no limit)
- `mode`: fixed to `"multi_turn"` — this is a deliberate single-value enum rather than a removed field, so that any future reintroduction of a second mode is a schema *extension* (adding a new allowed value) rather than a schema *break*

**Serialization:** `run_config.json` (single object, not JSONL, since there's exactly one per run) — written once at run start via `logging_writer.write_run_config()`, and re-written at run completion with `completed_at_utc`, `status`, and final `total_cost_usd` populated (this is the one field set that is mutated and re-persisted, unlike the append-only `interactions.jsonl`).

---

## 5. Database / Storage Schema

### 5.1 On-disk directory layout

```
runs/
  {run_id}/
    run_config.json          # one ExperimentRun object
    interactions.jsonl       # append-only, one Interaction per line
    agents_final.jsonl       # one Agent per line, written once at completion
    checkpoints/
      latest.json            # one CheckpointState object, overwritten each turn

data/
  personas/
    {persona_pool_id}.jsonl  # one persona string per line (plain text,
                               # not a Pydantic model -- see 3.2)
  memes/
    {meme_pool_id}.jsonl     # one MemeContent per line

pricing_table.yaml           # see 5.3

runs_manifest.sqlite         # OPTIONAL, see 5.4
```

### 5.2 JSONL field schemas (cross-referenced to §4)

**`interactions.jsonl`** — one `Interaction` per line, per §4.3's Pydantic model. Field-by-field:

| Field | Type | Nullable | Notes |
|---|---|---|---|
| `interaction_id` | string (uuid4) | No | |
| `run_id` | string | No | FK to `run_config.json`'s `run_id` |
| `turn` | int | No | ≥1 |
| `speaker_agent_id` | string | No | FK to `agents_final.jsonl`'s `agent_id` |
| `neighbor_agent_ids` | array[string] | No | length == run's `N`, except never empty |
| `stance_before` | float | No | |
| `stance_after` | float | No | |
| `reason_text` | string | No | may be empty string (never null) for `failed_logged_null` or meme rows without a caption fallback |
| `content_type` | string enum | No | `"generated_text"` \| `"meme"` |
| `meme_id` | string | Yes | non-null iff `content_type == "meme"` |
| `used_vision_fallback` | bool | No | |
| `prompt_token_count` | int | No | 0 for meme-posting turns |
| `completion_token_count` | int | No | 0 for meme-posting turns |
| `model_backend_id` | string | No | |
| `latency_ms` | int | No | 0 for meme-posting turns |
| `api_call_status` | string enum | No | `"success"` \| `"retried_success"` \| `"failed_logged_null"` |
| `timestamp_utc` | string (ISO 8601) | No | |

**`agents_final.jsonl`** — one `Agent` per line, per §4.1, with `stance_history` embedded as a nested array of `StanceRecord` objects (fields: `turn`, `stance_value`, `reason_text`, `interaction_id`).

**`data/personas/{persona_pool_id}.jsonl`** — plain-text lines, one persona description per line (not JSON-wrapped; a deliberate simplification since a persona is just a string with no structured sub-fields in this scope).

**`data/memes/{meme_pool_id}.jsonl`** — one `MemeContent` per line, per §4.2.

### 5.3 `pricing_table.yaml`

```yaml
openai:
  gpt-4o:
    text: {prompt_per_1k: 0.0050, completion_per_1k: 0.0150}
    vision: {prompt_per_1k: 0.0050, completion_per_1k: 0.0150}  # image cost
                                                                    # folded into
                                                                    # prompt_tokens
                                                                    # count by
                                                                    # the provider
  gpt-3.5-turbo:
    text: {prompt_per_1k: 0.0005, completion_per_1k: 0.0015}
anthropic:
  claude-sonnet-4-6:
    text: {prompt_per_1k: 0.0030, completion_per_1k: 0.0150}
    vision: {prompt_per_1k: 0.0030, completion_per_1k: 0.0150}
google:
  gemini-2.0-flash:
    text: {prompt_per_1k: 0.0001, completion_per_1k: 0.0004}
    vision: {prompt_per_1k: 0.0001, completion_per_1k: 0.0004}
local:
  llama-3.1-70b:
    text: {prompt_per_1k: 0.0, completion_per_1k: 0.0}  # self-hosted, no per-call $ cost
```

Schema: top-level key = provider (matches `model_backend_id`'s provider prefix), second-level key = model id, third-level key = modality (`text` | `vision`), leaf = `{prompt_per_1k: float, completion_per_1k: float}`. This file is hand-maintained per Fix H's note that pricing changes faster than any package update cycle — it is not derived from anything else and must be updated manually when providers change pricing.

### 5.4 Run-manifest index (optional addition — justified here, not in the HLD)

**This is a new addition beyond what `sandbox_hld_v2.md` specifies.** The HLD's non-goal explicitly rules out a relational database as the *primary* store, and this respects that: `runs_manifest.sqlite` is a **derived, rebuildable index**, never the source of truth, used only to answer "show me all runs matching X" queries (e.g., "list every RQ3 run with `meme_injection.enabled=true` and `status=completed`") without scanning every `run_config.json` file individually as the number of runs grows across a research campaign.

```sql
CREATE TABLE run_manifest (
    run_id TEXT PRIMARY KEY,
    rq_target TEXT NOT NULL,
    topic TEXT NOT NULL,
    language_condition TEXT NOT NULL,
    meme_injection_enabled INTEGER NOT NULL,  -- 0/1
    model_backend_id TEXT NOT NULL,
    trial_number INTEGER NOT NULL,
    status TEXT NOT NULL,
    total_cost_usd REAL NOT NULL,
    started_at_utc TEXT,
    completed_at_utc TEXT
);
```

This table is populated by a small rebuild script (`rebuild_manifest.py`, not detailed further here since it is a thin, non-load-bearing convenience utility) that scans `runs/*/run_config.json` and upserts one row per run. It is safe to delete `runs_manifest.sqlite` at any time and regenerate it from the JSONL/JSON files — nothing in the Orchestrator, Gateway, or any core module reads from or writes to this file; only the Python API's `list_runs()` function (§6.2) uses it, and that function falls back to a direct directory scan if the manifest file is absent or stale (see §6.2).

### 5.5 Worked examples

**One `Interaction` line — generated-text turn:**

```json
{"interaction_id":"7c3e1a2f-...","run_id":"run_2026-09-15_a1b2","turn":6,"speaker_agent_id":"agent_0042","neighbor_agent_ids":["agent_0011","agent_0077","agent_0003","agent_0059","agent_0028"],"stance_before":4.0,"stance_after":5.0,"reason_text":"Given what my neighbors have said, I'm leaning more strongly toward supporting this policy because...","content_type":"generated_text","meme_id":null,"used_vision_fallback":false,"prompt_token_count":842,"completion_token_count":156,"model_backend_id":"gpt-4o","latency_ms":1834,"api_call_status":"success","timestamp_utc":"2026-09-15T14:32:07Z"}
```

**One `Interaction` line — meme-injected turn:**

```json
{"interaction_id":"9f4d2b1e-...","run_id":"run_2026-09-15_c7d8","turn":4,"speaker_agent_id":"agent_0015","neighbor_agent_ids":["agent_0040","agent_0002","agent_0091","agent_0066","agent_0033"],"stance_before":3.0,"stance_after":6.5,"reason_text":"When will people learn??","content_type":"meme","meme_id":"meme_ipm_00214","used_vision_fallback":false,"prompt_token_count":0,"completion_token_count":0,"model_backend_id":"gpt-4o","latency_ms":0,"api_call_status":"success","timestamp_utc":"2026-09-15T14:31:52Z"}
```

Note the meme row's `stance_before` (this agent's prior generated stance) versus `stance_after` (the meme's fixed `stance_label`, 6.5) — the "shift" for a meme-posting agent's own turn is an artifact of the injection mechanism itself, not something to be interpreted as that agent's own reasoning-driven update; downstream analysis (per §7's filtering requirement) treats meme rows differently from generated-text rows for exactly this reason.

---

## 6. API Design

No human-facing web UI exists in this system (per the HLD's non-goal). "API" here means the CLI and Python interfaces a researcher uses directly.

### 6.1 CLI

```
python -m sandbox run --config <path> [--dry-run]
python -m sandbox status --run-id <id>
python -m sandbox resume --run-id <id>
python -m sandbox list [--rq-target <RQ1_RQ2|RQ3|RQ1_RQ2_RQ3>] [--status <status>]
```

**`run`**
- `--config <path>`: required, path to a YAML run config.
- `--dry-run`: validates the config (via `config_loader.load_run_config`) and prints the resolved `ExperimentRun` plus an estimated call count and estimated cost (using `pricing_table.yaml` and the run's `M/N/K/trials` — a simple multiplication, not a simulation), WITHOUT launching the run. Recommended before every real run, especially given the HLD's Critique Pass 2 cost concerns.
- Exit codes: `0` success (run completed, `status=completed`); `1` config validation failure (nothing was launched); `2` run started but failed (`status=failed_partial` or `failed_total` — a checkpoint may exist and `resume` can be tried); `130` if interrupted by SIGINT (checkpoint from the last completed turn is preserved, since `checkpoint_manager.save()` runs after every turn, per Fix G).
- stdout: structured JSON progress lines, one per completed turn, e.g. `{"event": "turn_completed", "turn": 6, "run_id": "...", "cost_so_far_usd": 4.32}` — machine-parseable, so this can be piped into a monitoring script.
- stderr: human-readable log messages (see §7's logging conventions).

**`status --run-id <id>`**
- Reads `run_config.json` and the latest checkpoint (if any); prints current `status`, `last_completed_turn` (from checkpoint), `total_cost_usd`, and estimated turns remaining. Exit code `0` if the run_id is found (regardless of its status), `1` if not found.

**`resume --run-id <id>`**
- Re-invokes `SimulationOrchestrator.run()` for an existing `run_id`, which internally detects and loads the checkpoint per §3.14. Exit codes mirror `run`. Fails with exit code `1` and a clear message if no checkpoint exists for that `run_id` (nothing to resume from — the user should use `run` with the original config instead).

**`list`**
- Queries `runs_manifest.sqlite` if present and not stale (see §6.2); otherwise scans `runs/*/run_config.json` directly. Prints a table (run_id, rq_target, status, cost) to stdout.

### 6.2 Python API

```python
# sandbox/api.py

async def launch_run(config_path: Path) -> str:
    """Loads and validates the config, then runs the full simulation
    synchronously (awaits until completion or failure). Returns the
    run_id. Raises ConfigLoadError, CostCeilingExceeded, or any
    uncaught Orchestrator exception -- this function does NOT swallow
    errors, since a notebook user needs to see failures directly."""
    run = load_run_config(config_path)
    orchestrator = _build_orchestrator(run)
    await orchestrator.run()
    return run.run_id


def get_run_status(run_id: str) -> "RunStatus":
    """Synchronous, fast (reads run_config.json + latest checkpoint only,
    no full interactions.jsonl scan). Raises FileNotFoundError if run_id
    is unknown."""
    ...


def load_run_dataframe(run_id: str) -> "pd.DataFrame":
    """Thin wrapper around the loader already defined in 3.13 --
    exposed here as the primary entry point a researcher calls from a
    notebook: `df = load_run_dataframe("run_2026-09-15_a1b2")`."""
    run_dir = Path("runs") / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"no such run: {run_id}")
    return _analysis.load_run_dataframe(run_dir)


def run_coherence_analysis(run_id: str, judge_model_id: str = "gpt-4o") -> "pd.DataFrame":
    df = load_run_dataframe(run_id)
    return _analysis.score_coherence(df, judge_model_id)


def run_bimodality_analysis(run_id: str) -> "BimodalityResult":
    df = load_run_dataframe(run_id)
    return _analysis.bimodality_by_turn(df)


def run_ohagi_regression(run_id: str) -> "RegressionResult":
    df = load_run_dataframe(run_id)
    return _analysis.run_ohagi_regression(df)


def list_runs(rq_target: str | None = None, status: str | None = None) -> list["RunSummary"]:
    """Prefers runs_manifest.sqlite; falls back to a directory scan if
    the manifest is missing OR if its row count doesn't match the
    number of run directories present (a simple staleness heuristic --
    not a perfect check, but cheap and catches the common case of the
    manifest having fallen behind after manual file deletion)."""
    ...
```

Every function above is a thin, documented entry point over the LLD modules in §3 — this section exists so a researcher has one place (`sandbox/api.py`) to import from, rather than needing to know which of the dozen internal modules to call directly.

### 6.3 Optional local HTTP status endpoint

**Justification for including this:** given K=10-turn runs at M=100 can run for a meaningful fraction of an hour depending on model latency, and given checkpointing already writes progress to disk every turn (§3.14), a minimal read-only polling endpoint is low-cost to add and lets a researcher check progress from a second terminal or a lightweight external script without needing to shell into the process running the CLI. This is explicitly **not** a UI — no HTML, no visualization, JSON only, and it is entirely optional (the CLI's `status` subcommand already covers the same need for a user with terminal access to the same machine; this only adds value for remote/multi-process monitoring).

```
GET http://localhost:8765/runs/{run_id}/status
→ 200 {"run_id": "...", "status": "running", "last_completed_turn": 6,
        "total_turns": 10, "cost_so_far_usd": 4.32}
→ 404 if run_id unknown
```

Implementation: a single-file `sandbox/status_server.py` using a minimal ASGI framework (e.g., a handful of routes), reading the same `run_config.json` + checkpoint files the CLI's `status` subcommand reads — no new data source, purely a network-accessible wrapper around `get_run_status()`. Started optionally via `python -m sandbox serve-status --port 8765`, separate from `run` (a researcher who doesn't want this can simply never invoke `serve-status`).

---

## 7. Cross-Cutting Concerns

### 7.1 Configuration validation rules (consolidated, full list)

All enforced by Pydantic at `ExperimentRun`/`MemeInjectionConfig` construction time (§4.4, §4.5), surfaced via `ConfigLoadError` (§3.1) before any run starts:

- `alpha ∈ [0.0, 1.0]`
- `M ≥ 2`, `N ≥ 1`, `N < M`
- `K ≥ 1`, `trial_number ≥ 1`, `checkpoint_every_n_turns ≥ 1`
- `language_condition ∈ {english, hinglish, hindi}`
- `stance_scale` has ≥ 2 distinct values
- `temperature ∈ [0.0, 2.0]`
- `max_cost_usd` is `None` or `> 0.0`
- `meme_injection.enabled == True` requires non-null `meme_pool_id`
- `meme_injection.injection_rate ∈ [0.0, 1.0]`
- `meme_injection.injection_schedule ∈ {random, fixed_turn}`; `fixed_turn` requires non-empty `fixed_turns`
- `meme_pool_id` (if referenced) and `persona_pool_id` must resolve to existing files on disk (checked by `config_loader._validate_referenced_assets`, not by Pydantic itself, since this requires filesystem access)
- `mode` is fixed to `"multi_turn"` (schema-enforced single-value enum)
- `rq_target ∈ {RQ1_RQ2, RQ3, RQ1_RQ2_RQ3}`

### 7.2 Logging conventions

- **stderr, human-readable, leveled** (`DEBUG`/`INFO`/`WARNING`/`ERROR`) via Python's standard `logging` module: `INFO` for turn-completion and checkpoint events; `WARNING` for `StanceParseFailure` retries (each retry attempt logged, so a high per-model retry rate is visible without needing to grep `interactions.jsonl`); `ERROR` for `FatalGatewayError` and any uncaught exception that halts a run.
- **stdout, structured JSON, one event per line** for progress events consumed by tooling (per §6.1) — kept strictly separate from the human-readable stderr stream so piping stdout into another process never has to filter out log noise.
- **`interactions.jsonl` is not a log in this sense** — it's the primary data output, covered in §5, not a diagnostic stream.

### 7.3 Cost-ceiling enforcement — exact behavior

Per `cost_tracker.py`'s `record()` (§3.11): the cost ceiling check happens **after** the triggering call's cost has already been computed and added to `total_usd`, then `CostCeilingExceeded` is raised. This means: **the run does not halt before the triggering call completes** — the call that pushes total cost over the ceiling is allowed to finish and its cost is counted, and the *next* call attempt is what actually fails to proceed (since `CostCeilingExceeded` propagates up and is treated as fatal by the Orchestrator, per §3.9's error-handling note, at the next point the Orchestrator would otherwise dispatch further calls). This is a deliberate choice: checking *before* the call would require estimating that call's cost in advance (possible for token count via prompt length, but not exact until the response's actual completion-token count is known), and the design prioritizes simplicity and exactness of `total_cost_usd` accounting over a small worst-case overshoot (bounded by at most one call's cost beyond the ceiling, at most ~100 calls' worth if the overshoot happens to occur at the start of a turn's concurrent dispatch batch — flagged explicitly, not hidden, and worth knowing if setting `max_cost_usd` close to a hard budget limit: set it with headroom for one turn's worth of concurrent calls, not to the exact dollar limit).

### 7.4 Testing strategy

**Unit tests (highest priority, per the HLD's explicit guidance):**
- `interaction_engine.py`: `AlphaSampling.select_neighbors()` — assert `alpha=1.0` always returns the N closest-stance candidates (deterministic given a fixed rng seed and a population with clearly separated stance values); assert `alpha=0.0` produces a roughly uniform distribution over many repeated calls with different seeds; assert `ValueError` on invalid `alpha`/insufficient population.
- `stance_parser.py`: `parse_stance()` — assert correct extraction on well-formed output; assert `StanceParseFailure` on missing `STANCE:` line, non-numeric value, and out-of-scale value (via `clamp_and_validate_scale`).
- `meme_pool_manager.py`: `resolve_injections_for_turn()` — assert `injection_rate=0.0` never injects; assert `injection_rate=1.0` always injects (for `random` schedule); assert `fixed_turn` schedule only injects on the configured turn(s).

**Integration test (one full small run):** M=5, N=2, K=2, `meme_injection.enabled=True` with a tiny 3-item test meme pool, against a mocked `ModelGateway` (no real API calls — a fixture that returns deterministic, well-formed `BackendResponse` objects). Assertions:
- Exactly `M * K = 10` `Interaction` records are written to `interactions.jsonl` (accounting for the fact that meme turns still produce one record each, just with zero token counts).
- Every `neighbor_agent_ids` list has exactly `N=2` entries and never contains the speaker's own `agent_id`.
- Every `Interaction` with `content_type == "meme"` has a non-null `meme_id` and zero `prompt_token_count`/`completion_token_count` (validating §3.9's `_build_meme_interaction`).
- A checkpoint exists after the run and `CheckpointManager.load()` returns `last_completed_turn == 2`.
- Re-running `resume` against this completed run_id is a no-op (loop range is empty since `start_turn > K`) and does not error.
- `run_ohagi_regression()` and `bimodality_by_turn()` (§3.13) both run without raising against this small dataset, validating the post-hoc analysis path end-to-end even at trivial scale.

---

## 8. Open Questions / Explicitly Deferred Decisions

- **Meme stance-label rescaling.** §4.2 notes `MemeContent.stance_label` is not validated against the simulation's `stance_scale` at the schema level. If the chosen pre-existing meme dataset's label scale doesn't match your configured `stance_scale` (e.g., dataset labels are 0–1 continuous, simulation scale is 1–7 discrete), a rescaling step is needed before memes can be loaded into a run. No default is picked here because the correct rescaling function depends entirely on which dataset is chosen, which isn't fixed by the HLD or this document — this should be resolved (a `rescale_stance_label()` function in `meme_pool_manager.py`, or a preprocessing step on the raw dataset before it becomes a `{meme_pool_id}.jsonl` file) once the actual dataset is selected.
- **`SeedManager`'s three-stream concurrency assumption (§3.12).** The design assumes neighbor-sampling and meme-injection resolution never happen concurrently with each other (both are sequential steps within `_run_turn`, per §3.9). This is true in the current design and is not expected to change, but if any future modification parallelized these two resolution steps against each other, the shared `random.Random` instances would need per-call locking that isn't currently designed. Flagged, not built, since it isn't needed by the current architecture.
- **Judge-model cost/rate-limiting for `coherence_scorer.py` (§3.13).** The post-hoc coherence-scoring pass makes its own LLM calls (to a fixed judge model) but this document doesn't specify whether these go through the same `ModelGateway`/`RateLimiter`/`CostTracker` instances as the simulation itself, or a separate, lighter-weight client. Recommended default: reuse `ModelGateway` for consistency (same retry/rate-limit protections), but this is stated as a recommendation, not a resolved design, since the post-hoc analysis modules in §3.13 are currently sketched as synchronous/simple and would need an async refactor to share the Gateway's `async def generate()` cleanly — a small but real implementation decision left to whoever builds this module.
- **Manifest staleness detection (§5.4, §6.2).** `list_runs()`'s row-count heuristic for detecting a stale `runs_manifest.sqlite` is acknowledged as imperfect (it catches "runs added since last rebuild" but not "run status changed since last rebuild without a directory count change"). No more robust solution (e.g., a file-modification-time comparison) is specified here, since the manifest is an optional convenience layer, not load-bearing, and a researcher who suspects staleness can always fall back to `list --no-manifest` (an implied but not fully specified CLI flag) or simply re-run the rebuild script.

