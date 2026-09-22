# Phase 4 Plan: Tier 3 Integration (Simulation Orchestrator + Prompt Builder)

Status: **planning only, not started**. This document is the plan for Phase 4 of the sandbox build. Phase 0/1 (`models.py`, `config_loader.py`, `seed_manager.py`, `cost_tracker.py`, `rate_limiter.py`), Phase 2 (`agent_manager.py`, `meme_pool_manager.py`, `model_gateway.py`), and Phase 3 (`interaction_engine.py`, `stance_parser.py`, `vision_fallback.py`, `logging_writer.py`, `checkpoint_manager.py`) are all merged and unit-tested: 13 modules, 83 passing tests.

Scope: the integration tier. `simulation_orchestrator.py` (`full_design_doc.md` §3.9) is the only module that calls every other component, wiring them into the per-turn loop. `prompt_builder.py` has no design anywhere in either architecture document and must be designed from scratch here before it can be built. This is the phase where the system first becomes capable of running an actual simulation end to end.

A pre-implementation audit of the source, the tests, the architecture docs, and the docs site turned up six real defects and one broken fixture that Phase 4 must clear first. Section 0 covers those; the rest of the document is the new work.

---

## Files to create

- `sandbox/prompt_builder.py` (new design, Section 1)
- `sandbox/simulation_orchestrator.py` (port of §3.9 plus the gaps in Section 2)
- `tests/test_prompt_builder.py`
- `tests/test_simulation_orchestrator.py`
- `tests/test_integration_run.py` (the first end-to-end test in the repo)
- `tests/test_models.py` (`sandbox/models.py` currently has zero dedicated tests)
- `configs/example_run.yaml` (no run-config YAML exists anywhere in the repo today)

## Files to modify

- `sandbox/model_gateway.py`, `sandbox/config_loader.py`, `sandbox/interaction_engine.py`, `sandbox/models.py` (defect fixes, Section 0)
- `sandbox/meme_pool_manager.py` (add `get_meme()`, per Q4.7)
- `tests/conftest.py` (hoist the duplicated factories, Section 3)
- `data/memes/images/` (two missing fixtures), `data/personas/pool_20.jsonl` (persona prefix, per Q4.10)
- `CLAUDE.md`, `plan.md`, `phase3.md`, `docs/assets/content.js` (Section 5)

No new runtime dependencies. Everything Phase 4 needs is already pinned.

---

## Section 0: Preconditions, defects to fix first

Two of these will abort the first real run on its first API response. Both are faithful ports of errors in `full_design_doc.md`'s own sample code, invisible today because every existing test mocks around them. The port is correct; the spec is wrong.

| # | Location | Severity | Defect |
|---|---|---|---|
| D1 | `model_gateway.py:82` | **Blocking** | Pricing lookup is unreachable for 6 of 7 priced models |
| D2 | `model_gateway.py:117` | **Blocking** | `raw.response_ms` does not exist on litellm's `ModelResponse` |
| D3 | `data/memes/images/` | **Blocking** | 2 of 3 fixture images are missing from disk |
| D4 | `config_loader.py:56` | High | Non-mapping YAML escapes the `ConfigLoadError` contract |
| D5 | `config_loader.py:59-60` | Low | No-op `except`/`raise` discards the real config path |
| D6 | `interaction_engine.py:91` | Low | Unguarded `1.0 / w` |
| D7 | `models.py:27` | Low | `max_length=5` is not the guarantee the spec claims |

### D1. Pricing lookup fails for every non-bare-OpenAI model

`generate()` passes `provider=self._provider_name()`, which strips the prefix (`model_gateway.py:122`), but `model_id=self.model_backend_id`, which does not. `pricing_table.yaml` is keyed by bare model ids, so the two disagree:

```
gpt-4o                       -> provider=openai     OK
anthropic/claude-sonnet-4-6  -> looks up pricing["anthropic"]["anthropic/claude-sonnet-4-6"]  ValueError
local/llama-3.1-70b          -> looks up pricing["local"]["local/llama-3.1-70b"]              ValueError
claude-sonnet-4-6            -> provider defaults to "openai"                                  ValueError
```

Only bare OpenAI ids resolve. Worse, `CostTracker._lookup_rate` raises *after* `_call_with_retry` has already returned, so the run dies having already spent the money on the call. This directly contradicts `config_loader`'s stated "fail before spending API budget" philosophy.

**Fix:** `model_id=self.model_backend_id.split("/")[-1]` at `:82`, matching how `_resolve_vision_support` already normalizes at `:131`. Add a test parametrized over all seven ids in `pricing_table.yaml` with a real (not autospec'd) `CostTracker`, since the current `ModelGateway`-to-real-`CostTracker` seam has no coverage at all.

### D2. `raw.response_ms` raises `AttributeError`

Verified against the installed litellm 1.101.0:

```
hasattr(ModelResponse(), "response_ms")  -> False
hasattr(ModelResponse(), "_response_ms") -> False
r.response_ms -> AttributeError: 'ModelResponse' object has no attribute 'response_ms'
```

`AttributeError` is neither `TransientGatewayError` nor `FatalGatewayError`, so per the error contract in §3.9 it propagates as an unanticipated exception and aborts the entire run on the first *successful* API response. `tests/test_model_gateway.py:38-45` uses a `SimpleNamespace` that supplies `response_ms`, which is exactly why this has never surfaced.

**Fix:** measure latency in the gateway rather than reading it off the provider response. Wrap the `litellm.acompletion` await in `time.monotonic()` and compute the delta. Note `Interaction.latency_ms` is `int` with `Field(ge=0)`, and a float with a fractional part fails Pydantic validation, so the result must be `int(...)`. Add a test asserting the mock's supplied timing is ignored and a real non-negative int is produced.

### D3. Two of three meme fixture images do not exist

`data/memes/test_pool.jsonl` references `images/test_00{1,2,3}.jpg`. Only `test_001.jpg` (54 bytes, a 1x1 JPEG) exists. `phase3.md:30` called for this fixture and one was created, but only one of the three.

This is invisible today because `tests/test_vision_fallback.py:8` hardcodes the path to `test_001.jpg` and hand-builds its `MemeContent` rather than loading the pool, while `tests/test_meme_pool_manager.py` loads the real pool but never reads `image_path`. The moment Phase 4 runs a meme-enabled integration test against a vision-capable backend, `_load_image` raises `FileNotFoundError` for roughly two out of every three sampled memes, non-deterministically depending on which the RNG draws.

**Fix:** add `test_002.jpg` and `test_003.jpg` (copies of the 1x1 are fine). Add a test that loads the real pool and asserts every `image_path` resolves, so this class of fixture drift fails loudly in future.

### D4. Non-mapping YAML escapes the error contract

`load_run_config`'s docstring promises "Raises `ConfigLoadError` on any failure", but `raw.pop("git_commit_hash", None)` at `:56` runs before anything confirms `raw` is a mapping. `:45-46` handles only `raw is None`:

```
"- a\n- b\n"      -> TypeError: pop expected at most 1 argument, got 2
"just_a_string\n" -> AttributeError: 'str' object has no attribute 'pop'
"42\n"            -> AttributeError: 'int' object has no attribute 'pop'
```

**Fix:** an `isinstance(raw, dict)` guard after `:46` raising `ConfigLoadError`. One line, plus three parametrized tests.

### D5. No-op exception handler discards the config path

```python
try:
    raw["git_commit_hash"] = _resolve_git_commit_hash()
except ConfigLoadError:
    raise
```

Catching and bare-`raise`ing is identical to having no handler. The consequence is that the error propagates carrying `Path("<config>")`, set inside `_resolve_git_commit_hash` at `:101`, instead of the real path the caller passed in, so the user is told a config is invalid without being told which file.

**Fix:** re-wrap with the correct `path`, which is what the handler was clearly meant to do. Assert the raised error's `.path` in a test.

### D6. Unguarded reciprocal in the sampling key

`keyed = [(rng.random() ** (1.0 / w), item) for w, item in zip(weights, items)]` divides by `w` with no guard:

```
w == 0.0  -> ZeroDivisionError
w == -1.0 -> no error; random() ** -1 > 1, and keys sort descending,
             so a negative-weight item is GUARANTEED to be selected first
```

The negative case is the dangerous one: a silent wrong answer rather than a raise, in the one module whose entire thesis is "never silently clamp". It is currently unreachable from `AlphaSampling`, since `blended = alpha * sim + (1 - alpha) * 1.0` is strictly positive for any `alpha` in `[0, 1]` and any finite distance. It becomes reachable if any stance is `NaN`, which `initial_stance: float` accepts in Pydantic's lax mode and which would propagate through `distance` to `blended` and make the sort order garbage.

**Fix:** raise `ValueError` on any non-positive or non-finite weight. Also pass `strict=True` to `zip`, which silently truncates on a length mismatch today (Python 3.10+; the venv is 3.14).

### D7. The memory-window cap is not a schema guarantee

`full_design_doc.md:1107` claims `memory_window`'s cap is "a schema-level guarantee, not just an implementation habit that could silently drift". That is not true as written. Without `model_config = ConfigDict(validate_assignment=True)`, Pydantic v2 validates only at construction:

```
Agent(..., memory_window=[6 items])  -> ValidationError    (construction)
a.memory_window = [8 items]          -> accepted, len == 8 (assignment)
```

In-place `stance_history.append(...)` at `agent_manager.py:91` bypasses validation entirely. The sole actual enforcement is the `[-5:]` slice at `agent_manager.py:99`, which is correct and idempotent, but is precisely the "implementation habit" the spec says it isn't.

**Fix:** either enable `validate_assignment=True` on `Agent`, or correct the claim. Recommend enabling it, since Fix A's window is load-bearing for RQ1's validity and Phase 4 is the first code to mutate agent state in anger. Add a test pinning whichever behavior is chosen.

---

## Section 1: `sandbox/prompt_builder.py`

This module has no specification anywhere. The total existing guidance amounts to: one call signature in §3.9's sample code, one table row in `sandbox_hld.md:230` naming its four inputs, the Fix A memory rule, the `STANCE:` regex it must satisfy, and three sentences of meme framing in `vision_fallback.py`. Everything about language conditions, persona framing, topic framing, scale explanation, and the actual template is undesigned.

Each decision below is stated as a resolved question, following how `plan.md` handled Question C1 rather than leaving gaps for the implementer to guess at.

### Interface

```python
@dataclass
class BuiltPrompt:
    text: str
    image: bytes | None
    used_vision_fallback: bool


class PromptBuilder:
    def __init__(
        self,
        run: ExperimentRun,
        supports_vision: bool,
        meme_lookup: Callable[[str], MemeContent],
    ): ...

    def build_discussion_prompt(
        self,
        agent: Agent,
        neighbor_posts: list[Interaction],
        turn: int,
        *,
        emphasize_format: bool = False,
    ) -> BuiltPrompt: ...
```

Pure and stateless past construction: no I/O except the image read that `vision_fallback._load_image` performs, and no mutation of any argument. Safe to call concurrently from dispatch tasks.

### Q4.1: What the builder receives for neighbors

**Resolved: `list[Interaction]`, not `list[Agent]`.**

The two architecture documents contradict each other. `full_design_doc.md:754` calls `build_discussion_prompt(agent, neighbors, turn)` where `neighbors` is `list[Agent]`; `sandbox_hld.md:230` states the input is `list[Interaction]` (neighbor posts, possibly meme-flagged). Resolve in favour of the HLD, because only `Interaction` carries `content_type` and `meme_id`. `StanceRecord`, which is all you can reach from an `Agent`, carries `reason_text` and `stance_value` but has no notion of whether a post was a meme, so the `list[Agent]` form makes the entire meme-receiver path in `sandbox_hld.md:284-286` impossible to implement.

The orchestrator therefore maintains `dict[agent_id, Interaction]` of each agent's most recent post, snapshotted at turn start and not updated until step 6. That snapshot discipline *is* Fix F.

### Q4.2: How `language_condition` enters the prompt

**Resolved: an English scaffold with exactly one varying directive line.**

```python
_LANGUAGE_DIRECTIVE = {
    "english":  "Write your reply in English.",
    "hinglish": "Write your reply in Hinglish (Hindi-English code-mixed, Roman script).",
    "hindi":    "Write your reply in Hindi (Devanagari script).",
}
```

Persona framing, topic framing, memory, neighbor posts, and the format specification are byte-identical across conditions. Only this one line changes.

This is the RQ1 validity decision and the reasoning must be recorded, not just the choice. `sandbox_hld.md:390` establishes that Fix A's memory window counts *turns* rather than tokens specifically to prevent a language-driven truncation asymmetry. A prompt builder that produced systematically longer prompts for Hinglish than for English would reintroduce exactly the confound Fix A exists to eliminate, one layer up. The three directives above are therefore deliberately near-identical in length and structure, and any future edit to one must preserve that parity. Add a test asserting the three rendered prompts differ only in that line.

Lookup raises `KeyError` on an unrecognized value rather than defaulting, consistent with the codebase-wide invariant. Pydantic already constrains the enum at both `models.py:23` and `:103`, so this is defense in depth.

Note that `hindi` is a valid schema value but appears in no RQ table: `sandbox_hld.md:438` scopes RQ1 to `{english, hinglish}`. Support it for schema completeness, but flag that no planned campaign uses it and it is therefore untested against real research output.

### Q4.3: How memory is rendered

**Resolved: read `agent.stance_history[-5:]` directly.**

`plan.md:43-44` already settled what `memory_window` contains: the agent's own most recent up-to-five `interaction_id`s, a redundant-but-explicit cache of `[r.interaction_id for r in agent.stance_history[-5:]]`, not a record of which neighbor posts it saw. `plan.md:44` states the consequence for this phase directly:

> an agent's own recent reasoning comes from `memory_window` to `stance_history` lookups (bounded, Fix A's turn-count window), while sampled neighbors' current-turn posts come fresh from that turn's `neighbor_map` (not from memory at all).

So the builder needs no external interaction store. Render the last up-to-five entries of the agent's own `stance_history` as its prior reasoning, and take neighbor content entirely from `neighbor_posts`. The `memory_window` id list is not read by the builder at all; it exists for `agents_final.jsonl` consumers.

### Q4.4: Stance scale and anchor labels

`stance_scale` is a full enumeration of valid points (`[1, 2, 3, 4, 5, 6, 7]` in every test), not a `(min, max)` pair. But `clamp_and_validate_scale` (`stance_parser.py:54`) only ever reads `min()` and `max()`, so intermediate points are not enforced and a model returning `4.5` is accepted. Render the bounds, not the enumeration: "a number from 1 to 7".

**No anchor-label field exists anywhere in the schema.** `ExperimentRun` has `topic` but nothing saying what the low and high ends of the scale mean, and an unanchored numeric scale is interpreted inconsistently by an LLM across turns and personas.

**Resolved: add optional `stance_low_label` and `stance_high_label` to `ExperimentRun`,** defaulting to "strongly opposed" and "strongly in favour". This is a small additive schema change (two optional string fields with defaults, so every existing config and test stays valid) and it materially improves construct validity, since topic-appropriate anchors are what make a stance number mean the same thing to every agent.

### Q4.5: The output-format contract

The prompt must produce output that satisfies `stance_parser.py:15`:

```python
_STANCE_PATTERN = re.compile(r"STANCE:\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)
```

Three consequences the template must respect, the second of which is non-obvious and load-bearing for RQ2:

1. The literal token `STANCE:` is required. Case-insensitive, so `stance:` parses, but emit canonical uppercase. A leading `-` and decimals are accepted; `+`, thousands separators, and scientific notation are not.
2. **`reason_text` is `_STANCE_PATTERN.sub("", raw_text).strip()`, which strips *every* `STANCE: <n>` occurrence, not just the first.** The first match supplies the value; all matches are deleted from the reasoning. So the prompt must instruct the model to emit the line once, first, and never restate it in prose. A model that writes "...so I am moving to STANCE: 6 because..." has that fragment silently excised from `reason_text`, which is precisely the field RQ2's coherence, code-mix, and sentiment analyses consume. This would corrupt RQ2's inputs invisibly.
3. An out-of-range value raises `StanceParseFailure` identically to an unparseable one and burns a retry, so the prompt must state the numeric bounds explicitly.

### Q4.6: `emphasize_format`

Used only on retries 2 and 3, after a `StanceParseFailure` (`full_design_doc.md:757-786`). It prepends a stronger, more insistent restatement of the format block.

**It must not change persona, topic, memory, neighbor content, or the language directive.** A retry that altered the experimental stimulus would mean the recorded `Interaction` is not a response to the same input as a first-attempt one, silently mixing two different conditions in the same dataset. Add a test asserting that the two renderings differ only in the format-emphasis block.

### Q4.7: Meme neighbor posts

When a neighbor's most recent `Interaction` has `content_type == "meme"`, the prompt must render that meme rather than its (empty or caption-derived) `reason_text`.

**The builder must call `vision_fallback.build_meme_prompt()` rather than constructing its own framing.** That module's docstring states it is "the ONLY place in the codebase that constructs a meme-derived prompt", and duplicating the `_INSTRUCTION` text would create exactly the kind of drift-prone second copy the docstring exists to prevent. `vision_fallback.py:15-19` is also the only prompt text that exists anywhere in this repo today, so it sets the register the rest of the template should match.

Resolving `meme_id` to `MemeContent` requires a lookup that does not currently exist. **Add `MemePoolManager.get_meme(meme_id) -> MemeContent`,** backed by a dict built at load time. This belongs on the pool manager (resolving a meme id is the meme pool's job) rather than being cached ad hoc in the orchestrator. It should raise `KeyError` on an unknown id rather than returning `None`.

### Q4.8: Multiple meme neighbors, one image slot

`ModelGateway.generate` accepts a single `image: bytes | None`, but with N=5 neighbors, several could have posted memes in the same turn. The design documents never address this.

**Resolved: render at most one image.** Take the first meme-bearing neighbor in `neighbor_posts` order, which is the order `AlphaSampling` produced and is therefore RNG-deterministic and reproducible, and render its image. Render every additional meme caption-only.

**Define `used_vision_fallback = True` if at least one meme post was rendered caption-only for any reason.** This is a superset of `sandbox_hld.md:285`'s original meaning (a text-only backend receiving the caption fallback) and it also captures the new second-meme downgrade case. The broader definition keeps the field honest: it answers "did this agent see less than the full meme content it was shown", which is what any analysis using it actually wants to know. Record this widening explicitly, since it is a deliberate departure from the HLD's narrower wording.

Widening `generate()` to accept multiple images is deliberately out of scope: it is a Tier 1 signature change for a case no RQ currently requires.

### Q4.9: Turn 1

At turn 1 no agent has a prior `Interaction`, so `neighbor_posts` is empty for every speaker.

**Resolved: omit the neighbor block entirely on turn 1** and state in the prompt that the discussion is just beginning. The alternative, showing neighbors' `initial_stance` values with no accompanying text, would present bare numbers with no reasoning attached, which is not a post and has no analogue in any later turn. Omitting keeps turn 1 an honest "state your opening position" turn.

### Q4.10: The persona prefix leak

`data/personas/pool_20.jsonl` lines read `"Persona 00: a cautious accountant who double-checks every claim."`. Personas are loaded as raw strings (`agent_manager.py:62-69` does no JSON parsing despite the `.jsonl` extension), so interpolating one directly yields:

```
You are: Persona 00: a cautious accountant who double-checks every claim.
```

The `Persona NN: ` prefix is fixture scaffolding leaking into the experimental stimulus.

**Resolved: fix the fixture,** stripping the prefix from all 20 lines. `tests/test_agent_manager.py:67-70` asserts `agent.persona == pool[i % 20]` against the file's own lines, so it stays green. Stripping in the builder instead would be a parsing rule applied to opaque free text, which is worse.

### Template sketch

```
You are: {persona}

You are taking part in an ongoing discussion about: {topic}

Your current position is {current_stance} on a scale from {lo} to {hi},
where {lo} means {low_label} and {hi} means {high_label}.

Your recent reasoning:
  Turn 3: "..."
  Turn 4: "..."

What others in your feed posted most recently:
  agent_0011 (position 5.0): "..."
  agent_0077 (position 3.0): {meme framing from vision_fallback.build_meme_prompt}

{language_directive}

Reply in exactly this format, with the STANCE line first and stated only once:
STANCE: <a number from {lo} to {hi}>
<your reasoning, two or three sentences>
```

### Tests

- Each of the three language conditions renders, and the three outputs differ only in the directive line (the Fix A parity guard).
- Unknown `language_condition` raises rather than defaulting.
- Output of a well-formed model reply against this template round-trips through `parse_stance` and `clamp_and_validate_scale`.
- Memory renders at most 5 entries, and fewer when `stance_history` is shorter.
- Turn 1 omits the neighbor block.
- A meme neighbor with `supports_vision=True` produces `image is not None` and `used_vision_fallback is False`.
- A meme neighbor with `supports_vision=False` produces `image is None`, `used_vision_fallback is True`, and never touches the filesystem.
- Two meme neighbors with `supports_vision=True` produce exactly one image and `used_vision_fallback is True`.
- `emphasize_format=True` differs from `False` only in the format block.
- No argument is mutated.

---

## Section 2: `sandbox/simulation_orchestrator.py`

A port of §3.9, plus everything that section's sketch leaves out. The per-turn sequence itself (`Architecture-docs/plantuml/per_turn_simulation.puml` is the authoritative diagram) is a direct port and is not restated here.

### 2a. Run-directory creation

`phase3.md:166` asked whether `LoggingWriter` or its caller creates `run_dir`. **Resolved in favour of the caller**, already pinned by `logging_writer.py:9-15` and by `tests/test_logging_writer.py:110`, which asserts `FileNotFoundError` when the directory is absent. Note this is asymmetric with `CheckpointManager.save`, which does its own `mkdir(parents=True, exist_ok=True)`.

So `SimulationOrchestrator.run()` must create `runs/{run_id}/` before calling `write_run_config()`. §3.9's sketch does not do this and would fail immediately.

### 2b. Exception taxonomy, and a contradiction to resolve

The design document contradicts itself. §3.11:928 states `CostCeilingExceeded` "is treated as a fatal run-level exception (not caught per-agent like `StanceParseFailure`)". But §3.9 catches only `StanceParseFailure` and `FatalGatewayError` inside `_process_one_agent_turn`, then collects with `asyncio.gather(..., return_exceptions=True)` and does:

```python
for agent, result in zip(all_agents, results):
    if isinstance(result, Exception):
        continue
```

which silently swallows `CostCeilingExceeded` along with everything else, and proceeds to the next turn. The cost ceiling would never actually stop a run.

**Resolved:**
- `StanceParseFailure` after the retry budget, and `FatalGatewayError`, become a `failed_logged_null` `Interaction` for that agent; the turn continues for everyone else.
- Every other exception type is **re-raised** out of `_run_turn` after the gather completes, aborting the run. This preserves the CLAUDE.md invariant that an unexpected exception surfaces loudly rather than being absorbed into a soft-failure record, and it makes the cost ceiling actually function.

Add tests for both branches. `api_call_status="failed_logged_null"` is declared in `models.py` but is **never constructed by any existing test**; Phase 4 is the first code to produce it.

### 2c. Run-lifecycle fields that nothing currently reads

`checkpoint_every_n_turns`, `started_at_utc`, `completed_at_utc`, `status`, and `total_cost_usd` are all declared at `models.py:112-117` and are read or written by no module in the repo. §4.5:1247 requires `run_config.json` to be re-written at completion with `completed_at_utc`, `status`, and final `total_cost_usd` populated. This is the one field set that is mutated and re-persisted, unlike the append-only `interactions.jsonl`.

Phase 4 owns all of it:
- Set `status="running"` and `started_at_utc` at start; `completed_at_utc`, final `status`, and `total_cost_usd` (from `CostTracker.total_usd`) at the end.
- Honour `checkpoint_every_n_turns` rather than checkpointing unconditionally every turn as §3.9's sketch does.
- On a fatal abort, persist `status="failed_partial"` if any turn completed, `failed_total` otherwise.

### 2d. SeedManager wiring

Phase 4 is the first code that actually routes the three streams to their consumers. Every existing test passes a bare `random.Random(n)`, so the `SeedManager`-to-consumer wiring has never been exercised. `neighbor_sampling_rng` goes to `AlphaSampling`, `meme_injection_rng` to `MemePoolManager`, `persona_assignment_rng` to `AgentManager`.

Honour `full_design_doc.md:1528`: neighbor resolution and meme-injection resolution must stay strictly sequential. Each `random.Random` instance is not safe for concurrent use, and the current design's safety rests entirely on those two steps never running in parallel with each other. Add a comment at the call site so a future refactor does not quietly break it.

### 2e. Fix F is enforced by convention, not by the API

`AgentManager.all_agents()` returns `list(self._agents.values())`, a new list of **live `Agent` references**. §3.9 hands that same list to both `select_neighbors` and the dispatch tasks. Nothing structurally prevents a dispatch coroutine from mutating agent state mid-turn and breaking the frozen-snapshot guarantee that CLAUDE.md calls load-bearing for reproducibility.

Nothing does mutate today, so this is not a live bug. But Phase 4 is where the risk becomes real, so state the invariant explicitly in the module docstring: **no dispatch coroutine mutates agent state; all mutation happens in step 6, after the gather returns.** Add a test that runs two turns with a fixed seed twice and asserts byte-identical `interactions.jsonl`.

### 2f. Missing helpers

`_now_utc_iso()` and `_current_stance_for_logging()` are referenced by §3.9's sample (`:755`, `:779`) but defined nowhere. Both are trivial; note that `_current_stance_for_logging` must read from the frozen snapshot, matching `interaction_engine._current_stance`'s semantics exactly, or `stance_before` will disagree with the value neighbor sampling actually used.

### 2g. Vision dispatch

§3.9:759 hardcodes `image=None` in the `generate()` call, which contradicts `sandbox_hld.md:284-286`'s multimodal path. **Resolve in favour of the HLD:** pass `BuiltPrompt.image` through to `generate()`, and set `Interaction.used_vision_fallback` from `BuiltPrompt.used_vision_fallback`.

Without this, `vision_fallback.py` remains fully built, fully tested, and entirely unreachable dead code, and `used_vision_fallback` stays permanently `False` in every run ever produced.

Note that `generate()` derives `modality = "vision" if image is not None else "text"` for cost lookup, so wiring the image through also makes vision pricing take effect. This interacts with D1: both must be fixed for a vision run to be costed correctly.

### Tests

Per-component, with a mocked gateway: resume-from-checkpoint path, fresh-run path, the two soft-failure branches, the fatal re-raise branch, `checkpoint_every_n_turns` honoured, lifecycle fields persisted, and determinism under a fixed seed.

---

## Section 3: Test infrastructure

`tests/conftest.py` is 40 lines defining two fixtures, both used only by `test_config_loader.py`. It needs work before Phase 4 adds a fourth and fifth test module.

- **Hoist the duplicated factories.** `_make_run` is copy-pasted verbatim into **four** files (`test_agent_manager.py:10`, `test_checkpoint_manager.py:12`, `test_cost_tracker.py:12`, `test_logging_writer.py:29`), differing only in `M`/`N`. `_make_agent` appears in three, `_make_interaction` in two. Move all three into `conftest.py` as parametrized factory fixtures before Phase 4 adds more copies.
- **Add a `mock_gateway` fixture.** Follow the existing pattern at `test_model_gateway.py:26-35`: `create_autospec(ModelGateway, instance=True)`, then reassign `generate = AsyncMock(...)` because autospec does not produce async mocks, and set `supports_vision` explicitly since it is a plain attribute assigned at `model_gateway.py:54`, not a property.
- **Add a stance-shaped response factory.** Every existing mock returns `text="ok"` or `"hello"`, which `parse_stance` rejects outright. Nothing in the repo currently produces a well-formed `"STANCE: 5\n<reason>"` response, and every Phase 4 test needs one.
- **Keep `no_real_sleep` file-local.** The autouse fixture at `test_model_gateway.py:16-23` patches `asyncio.sleep`. It is tempting to promote it to conftest, but doing so breaks the four wall-clock tests in `test_rate_limiter.py`, which measure real elapsed time via `time.monotonic()`.
- **Add `tests/test_models.py`.** `models.py` is the second-largest module (129 lines, three `@model_validator`s plus a property) and has no dedicated test file. `Interaction.check_meme_id_consistency` is never directly tested and its `content_type="meme"` branch is **never exercised anywhere in the suite**; `is_valid_for_analysis` is never called. Phase 4 is the first code to hit both.

---

## Section 4: Integration test

**No end-to-end test exists today.** Every test file imports exactly one production module plus `sandbox.models`. The two near-misses (`test_checkpoint_manager.py` driving a real `AgentManager` for fixture data, `test_model_gateway.py` composing autospec'd collaborators) do not assert any cross-module behavior.

Build the test §7.4 specifies: M=5, N=2, K=2, meme injection enabled against the 3-item fixture pool, mocked gateway returning deterministic well-formed responses. Assertions:

- Exactly `M * K = 10` `Interaction` records in `interactions.jsonl`, meme turns included.
- Every `neighbor_agent_ids` has exactly N=2 entries and never contains the speaker.
- Every `content_type == "meme"` row has a non-null `meme_id` and zero token counts.
- A checkpoint exists and `load()` returns `last_completed_turn == 2`.
- Re-running against the completed run_id is a no-op (`start_turn > K`) and does not error.
- `run_config.json` round-trips and carries populated lifecycle fields.

Also add **`configs/example_run.yaml`**. `load_run_config()` takes a YAML path, but no run-config YAML exists anywhere in the repo; tests build them inline via `tmp_path`, so there is no reference config a researcher can copy. This is also the natural place to demonstrate the new `stance_low_label`/`stance_high_label` fields from Q4.4.

---

## Section 5: Documentation corrections

The audit found the docs site (`docs/assets/content.js`) to be the most accurate document in the repo: it correctly marks Tier 2 built and Tier 3 planned, and has zero phantom file entries. The markdown docs have drifted badly.

### `CLAUDE.md` (DONE, corrected ahead of the rest of Phase 4)

Fixed out of sequence because this file is loaded into every session, so its errors propagate into every future piece of work. All five corrections are applied:

- Tier 2 was listed as "planned, not yet built". **False**; all five modules exist with 33 tests between them. Now marked built, with Tier 3 and the post-hoc modules marked "not yet built" so the reader can tell the two states apart.
- The "where to look" section described `phase3.md` as "planning-only, not yet implemented" while simultaneously telling agents to treat the phase docs as the current statement of what is built. Now explicitly warns that every phase doc's `Status:` header is stale by construction, and points at `phase4.md`'s defect list.
- Persona files were described as "one JSON object per line, loaded via `Model.model_validate_json()`". **False**; they are plain text (`agent_manager.py:62-69` does no JSON parsing). Split into separate meme and persona bullets, with the missing-image fixture gap noted.
- The `CostTracker.record()` sync note cited `phase3.md`, which does not mention `CostTracker` at all. Corrected to `plan.md`.
- `logging_writer.py` was called a "planned design". Corrected, and the empty `runs/` directory is now explained by the absent orchestrator rather than left as a bare fact.

Added alongside these: a standing note that tier status drifts in these docs, with the current ground truth (13 modules, 83 tests, Tiers 0-2 complete) and the instruction to verify against the filesystem and `pytest` rather than any prose status line.

### `plan.md` and `phase3.md`

Both open with `Status: **planning only, not started**` at `:3`, and both shipped. The pattern is structural: each phase doc is written before its code and committed *with* it (`f2db7af` bundled `plan.md` with Tier 0/1; `1603f1a` bundled `phase3.md` with Tier 2), so the header is accurate when written and stale the moment it lands. **Phase 4 should update this document's own status line when it merges, and add a one-line note to `CLAUDE.md` making the convention explicit so the next phase does not repeat it.**

Also: `plan.md:44` ("this is a proposal, not yet confirmed", which it was, and it shipped), `plan.md:143` (lists vision fallback as out of scope; now built), `phase3.md:30` (says none of the three meme images exist; one now does, see D3), and `phase3.md:84`/`:126`/`:167` (written as open, all three resolved in code).

### `docs/assets/content.js`

Build-status claims are correct and need no fix now, but five places flip when Phase 4 lands: `structure.intro`, `architecture.intro`, `hld.sections[1].paragraphs[0]`, `architecture.components[13]`/`[14]` (both `"files": []`, labelled PLANNED), and the `per-turn-loop` pipeline description.

There is no single versioned field to bump; the status claims live in three separate prose blocks. **Recommend adding `version` and `lastUpdated` to `meta`**, which currently holds only `projectName` and `description`, to give future updates one obvious hook.

Independently stale: `files['README.md']` still describes the pre-`f4b06c4` one-line README, and `structure.tree` claims "one test module per sandbox module (13 files)" when there are 12 (`models.py` has none, see Section 3).

---

## Sequencing

1. **Section 0 defects.** D1, D2, D3 are blocking and cheap. Do them first so nothing downstream is built against broken foundations.
2. **Section 3 test infrastructure.** Hoisting the factories and adding `mock_gateway` unblocks every test written after this point.
3. **`prompt_builder.py`.** No dependency on the orchestrator; fully testable alone. Resolving its design is what unblocks everything else.
4. **`MemePoolManager.get_meme()`** (Q4.7) and the `ExperimentRun` anchor fields (Q4.4). Small, and the builder needs both.
5. **`simulation_orchestrator.py`.** The integration point, built once its two new collaborators are stable.
6. **Integration test + `configs/example_run.yaml`.**
7. **Section 5 documentation corrections**, including this document's own status line.

---

## Open items

**Confirmed, no longer open:**

- **The `stance_low_label`/`stance_high_label` schema addition (Q4.4)** is approved. It is the only change here that touches Tier 0, and it is additive with defaults, so every existing config and test stays valid.
- **The `used_vision_fallback` widening (Q4.8)** is approved. The field means "did this agent see less than the full meme content it was shown", which is broader than `sandbox_hld.md:285`'s text-only-backend reading. Recorded here because data produced under this definition cannot be reinterpreted after the fact.

**Still to confirm:**

- **D7's resolution direction** (enable `validate_assignment` versus correct the spec's claim). Recommended above, but either is defensible.

---

## Out of scope for Phase 4

- **Phase 5, post-hoc analysis** (§3.13): `stance_regression.py`, `coherence_scorer.py`, `codemix_ratio_tracker.py`, `sentiment_proxy.py`, `bimodality_analysis.py`, `embedding_cluster.py`. These depend only on `interactions.jsonl`'s schema, which was locked in Phase 3, so this phase could in principle run in parallel with Phase 4 using hand-written fixture JSONL.
- **Phase 6, CLI and Python API** (§6): `python -m sandbox {run,status,resume,list}`, `sandbox/api.py`, and the optional SQLite run manifest and HTTP status endpoint. Blocked on a working orchestrator.

### Still-open items that Phase 4 makes acute

- **Real API credentials.** Still entirely unaddressed. Every test through Phase 3 runs against mocks, and Phase 4 is the first phase capable of a live run. Nothing in this plan requires real credentials (the integration test is mocked), but the first real run does.
- **Meme stance-label rescaling.** `MemeContent.stance_label` is a bare `float` with no link to `stance_scale`, and `MemePoolManager` never receives the run, so it structurally cannot validate. The fixture's 1-7 range is coincidental fixture authorship, not enforcement. Any real meme dataset (typically 0-1, -1..+1, or 1-5) silently produces garbage.
- **A real persona pool.** Only `test_pool.jsonl` (4 entries) and `pool_20.jsonl` (20, test-only) exist.
- **`pricing_table.yaml`'s model set.** Its own header flags the entries as illustrative placeholders. D1 makes them reachable; it does not make them correct.
- **CWD-relative asset paths.** `Path("data/memes")`, `Path("data/personas")`, `Path("runs")`, and `Path("pricing_table.yaml")` are all resolved against the process working directory in `config_loader.py`, `agent_manager.py`, `meme_pool_manager.py`, `cost_tracker.py`, and `checkpoint_manager.py`. The system only works when invoked from the repo root. Consistent with the spec, but Phase 6's CLI should pin it down.
