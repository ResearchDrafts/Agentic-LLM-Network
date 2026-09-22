# Phase 2 Plan — Tier 1 Features (5, 6, 7)

Status: **merged and shipped.** Phase 2 built `agent_manager.py`, `meme_pool_manager.py`, and `model_gateway.py`, all with tests, in commit `f2db7af`. The text below is the plan as written *before* that work, kept as the build record; it is not a statement of current state. For what is built now, check the filesystem and `python -m pytest`.

Scope: `sandbox_feature_specs.md` Features 5 (Persona & Population Initialization), 6 (Meme Pool Loading & Injection Decisioning), 7 (Model Gateway) — the full Tier 1 layer, which depends only on Tier 0 (already built).

---

## Files to create

- `sandbox/agent_manager.py` — Feature 5
- `sandbox/meme_pool_manager.py` — Feature 6
- `sandbox/model_gateway.py` — Feature 7
- `tests/test_agent_manager.py`
- `tests/test_meme_pool_manager.py`
- `tests/test_model_gateway.py`

No new runtime dependencies beyond what Phase 0 already pinned (`litellm`, `tenacity` are already in `pyproject.toml` and unused until now — Feature 7 is their first real consumer). Mocking `litellm.acompletion` needs only the stdlib `unittest.mock`, so no new dev dependency.

---

## 1. `agent_manager.py` (Feature 5)

**Build**, per the spec's Interface/Contract:

```python
class AgentManager:
    def __init__(self, run: ExperimentRun, rng: random.Random): ...
    def initialize_population(self) -> list[Agent]: ...
    def get_agent(self, agent_id: str) -> Agent: ...          # raises KeyError if unknown, uncaught
    def all_agents(self) -> list[Agent]: ...
    def apply_interaction(self, agent_id: str, interaction: Interaction) -> None: ...
    def snapshot(self) -> list[Agent]: ...                     # for Checkpointing (Feature 13, later)
    def restore_from_snapshot(self, snapshot: list[Agent]) -> None: ...
```

`initialize_population()`, `get_agent()`, `all_agents()` are fully specified already (full_design_doc.md §3.2) and are a direct port. `snapshot()`/`restore_from_snapshot()` aren't consumed until Feature 13, but the spec puts them on this feature's interface, so they're built now (trivial: `snapshot()` returns `copy.deepcopy(self.all_agents())` or a Pydantic `model_copy(deep=True)` list; `restore_from_snapshot()` rebuilds the internal dict from a `list[Agent]`).

**Design decision this phase must make (was flagged as Question C1 in the original plan — `apply_interaction()`/`memory_window` maintenance is never shown in any source document):**

Proposed design, to resolve before writing the implementation:
- `apply_interaction(agent_id, interaction)` appends a `StanceRecord(turn=interaction.turn, stance_value=interaction.stance_after, reason_text=interaction.reason_text, interaction_id=interaction.interaction_id)` to `agent.stance_history` — this part is unambiguous from the schema and Orchestrator's documented behavior (Feature 11's Edge Cases: "no new `stance_history` entry is appended" on failure).
- `memory_window` (schema-capped at 5, per `Agent.memory_window: Field(max_length=5)`) is read as: **the agent's own most recent up-to-5 interaction_ids** — i.e. a bounded pointer to its own last 5 `stance_history` entries, not a record of which neighbor posts it saw. Reasoning: the schema caps it at exactly 5 total entries, which is only consistent with "one id per turn, most recent 5 turns" — a per-turn record of *neighbor* posts would immediately exceed 5 entries once `N > 1` in a single turn. This makes `memory_window` a redundant-but-explicit cache of `[r.interaction_id for r in agent.stance_history[-5:]]`, recomputed on every `apply_interaction()` call. Its purpose is to give downstream prompt construction (and `agents_final.jsonl` consumers) a direct, explicit "last 5 turns" pointer without needing to slice `stance_history` themselves.
- This directly informs how Phase 4's prompt construction will work later: an agent's own recent reasoning comes from `memory_window` → `stance_history` lookups (bounded, Fix A's turn-count window), while sampled neighbors' current-turn posts come fresh from that turn's `neighbor_map` (not from memory at all). Flagging this now, before Phase 2 code is written, since it's exactly the kind of gap-filling decision the earlier questions pass called out — **this was a proposal at time of writing; it was confirmed and shipped as described.**

**Test fixtures:** already have `data/personas/test_pool.jsonl` (4 entries) from Phase 1 — reusable as-is. May add a second, smaller pool (e.g. 1-entry) specifically to exercise the "cycling with M > pool size" acceptance criterion cleanly.

**Tests to write (from Feature 5's Acceptance Criteria):**
- `M=100`, 20-entry persona pool → exactly 100 agents, personas cycle 5×.
- Two `AgentManager`s with identically-seeded `persona_assignment_rng` → identical `initial_stance` sequences.
- `get_agent("agent_9999")` on a 100-agent manager → `KeyError`, uncaught.
- Missing persona pool file → `FileNotFoundError` at `initialize_population()` time, not an empty population.
- New (not in the source spec, added because Phase 2 owns the design decision above): `apply_interaction()` correctly appends to `stance_history` and caps `memory_window` at 5 after more than 5 turns.
- `snapshot()`/`restore_from_snapshot()` round-trip: mutating the manager after taking a snapshot does not affect the snapshot; restoring from a snapshot reproduces agent state exactly.

---

## 2. `meme_pool_manager.py` (Feature 6)

**Build**, per spec:

```python
class MemePoolManager:
    def __init__(self, config: MemeInjectionConfig, rng: random.Random): ...
    def resolve_injections_for_turn(
        self, scheduled_speakers: list[Agent], turn: int
    ) -> dict[str, MemeContent | None]: ...
```

Straightforward port of `full_design_doc.md` §3.4 — load pool at construction if `enabled`, decide per-speaker per-turn via `random` schedule (independent probability check) or `fixed_turn` schedule (eligibility gated on `turn in fixed_turns`, same per-speaker rate applied on eligible turns).

**Test fixtures:** already have `data/memes/test_pool.jsonl` (3 `MemeContent` records) from Phase 1 — reusable as-is; may add a dedicated empty-file fixture (`data/memes/empty_pool.jsonl`, zero lines) to exercise the empty-pool `ValueError` cleanly rather than constructing it inline in the test.

**Tests to write (from Feature 6's Acceptance Criteria):**
- `enabled=False` → `None` for every speaker regardless of `injection_rate`.
- `injection_rate=0.0` → never injects, across many repeated calls.
- `injection_rate=1.0`, `random` schedule → always injects for every speaker.
- `fixed_turn` schedule with `fixed_turns={5}` → never injects on any turn but 5, even at `injection_rate=1.0`.
- Two `MemePoolManager`s with identically-seeded `meme_injection_rng` → identical injection decisions across an identical call sequence (the concrete Fix M verification named in the spec).
- Empty pool file → `ValueError` at construction, not at first use.
- Missing pool file → `FileNotFoundError` at construction.

---

## 3. `model_gateway.py` (Feature 7)

**Build**, per spec — the async, LiteLLM-backed interface:

```python
class ModelGateway:
    def __init__(self, model_backend_id: str, rate_limiter: RateLimiter, cost_tracker: CostTracker): ...
    supports_vision: bool
    model_backend_id: str
    async def generate(self, prompt_text: str, image: bytes | None = None, temperature: float = 0.7) -> BackendResponse: ...

class TransientGatewayError(Exception): ...
class FatalGatewayError(Exception): ...

@dataclass
class BackendResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    raw_provider_response: dict
```

**Note on the `cost_tracker.record()` call site:** Phase 1 deliberately made `CostTracker.record()` synchronous (not `async def`, see Phase 1's summary) — so `ModelGateway.generate()` calls it as a plain `self._cost_tracker.record(...)`, not `await`ed. This is consistent with `cost_tracker.py` as actually built, not the doc's ambiguous sketch.

**`_resolve_vision_support` lookup table** — needs to be pinned to a concrete set, ideally matching `pricing_table.yaml`'s existing entries so the two tables don't drift:

| Vision-capable | Text-only |
|---|---|
| `gpt-4o`, `gpt-4o-mini`, `claude-sonnet-4-6`, `gemini-2.0-flash` | `gpt-3.5-turbo`, `llama-3.1-8b`, `llama-3.1-70b` |

This matches `pricing_table.yaml` exactly (built in Phase 1) — no new model ids introduced. Per Feature 7 FR1, any `model_backend_id` outside both lists raises `ValueError` at construction — this is intentional and already covered by an acceptance-criterion test.

**Mocking strategy:** patch `litellm.acompletion` (via `unittest.mock.patch`, `AsyncMock`) to return a fake response object exposing `.choices[0].message.content`, `.usage.prompt_tokens`, `.usage.completion_tokens`, `.response_ms`, `.model_dump()` — matching exactly what `_call_with_retry` reads. Separate fixtures for: a clean success; a `litellm.exceptions.RateLimitError` that succeeds on retry N; a `litellm.exceptions.AuthenticationError` that never retries. No real network call, no API key needed anywhere in this phase.

**Tests to write (from Feature 7's Acceptance Criteria):**
- Mocked transient-error-then-success → succeeds on 2nd/3rd attempt.
- Mocked `AuthenticationError` → fails on first attempt, zero retries observed.
- `ModelGateway("totally-made-up-model-id", ...)` → `ValueError` at construction, before any `generate()` call.
- `image=None` call → exactly one `cost_tracker.record()` call with `modality="text"`; `image=<bytes>` → `modality="vision"`.
- `rate_limiter.release()` called exactly once per `generate()` invocation, even when the call raises (verify via a spy/mock on `RateLimiter`, not the real timing-based one, to keep this test fast and deterministic).

---

## Sequencing within Phase 2

1. `agent_manager.py` first — smallest extension of Phase 1's established patterns (Pydantic model + seeded RNG + fixture file), and its `snapshot()`/`apply_interaction()` design decision should be nailed down before anything else in this phase touches `Agent` mutation.
2. `meme_pool_manager.py` second — reuses the same fixture-file-plus-seeded-RNG pattern as step 1, lowest risk.
3. `model_gateway.py` last — the only feature in this phase requiring mock infrastructure (`litellm` patching) rather than just fixture files; builds on nothing from steps 1–2, so ordering here is about complexity, not dependency.

## Open items to confirm before execution

- The `memory_window`/`apply_interaction()` design proposal above (Question C1) — confirm or redirect before implementing `agent_manager.py`.
- The vision/text-only model table above — confirm the model set matches intent, or extend it (e.g. add `llava`/`qwen-vl` per the HLD's architecture diagram mention of local vision models) before implementing `model_gateway.py`.
- Per the earlier Phase 6 checkpoint in the original implementation plan: Question B1 (meme stance-label rescaling) only blocks pointing `meme_pool_manager.py` at a *real* dataset — the Phase 2 fixture pool already uses a matching 1–7-ish scale, so it doesn't block this phase's tests, only a future real-dataset swap.

## Out of scope for Phase 2 (deferred to later phases per the approved build order)

- Vision Fallback (Feature 9) — Tier 2, not touched here even though Feature 7 exposes `supports_vision`. (Built in Phase 3, and finally wired to a caller in Phase 4.)
- Any real (non-mocked) LiteLLM call, any real API key or credential wiring (Question D from the original plan is still open and still doesn't block mocked-Gateway testing).
- The Simulation Orchestrator's actual per-turn prompt construction that will *consume* `memory_window` (Tier 3, Feature 11) — this phase only builds the data structure and its maintenance, not its downstream use.
