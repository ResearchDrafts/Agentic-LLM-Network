# Phase 3 Plan — Tier 2 Features (Interaction Engine, Stance Parser, Vision Fallback, Logging Writer, Checkpoint Manager)

Status: **merged and shipped.** Phase 3 built `interaction_engine.py`, `stance_parser.py`, `vision_fallback.py`, `logging_writer.py`, and `checkpoint_manager.py`, all with tests, in commit `1603f1a`. The text below is the plan as written *before* that work, kept as the build record; it is not a statement of current state. The open items near the end were resolved during execution or in Phase 4, and are annotated there.

Scope: everything in `full_design_doc.md` §3 that depends only on Tier 0/1 (already built) and nothing else — the five remaining "leaf" modules that the Orchestrator (Tier 3) will wire together, but which have no dependency on the Orchestrator or on each other:

- `interaction_engine.py` (§3.3) — neighbor selection (`AlphaSampling`)
- `stance_parser.py` (§3.7) — extracts `(stance_value, reason_text)` from raw model output
- `vision_fallback.py` (§3.6) — caption-only prompt substitution for text-only backends
- `logging_writer.py` (§3.8) — append-only JSONL writer
- `checkpoint_manager.py` (§3.14) — per-turn checkpoint save/load with atomic rename

Note: `sandbox_feature_specs.md` (referenced by `plan.md` for Phase 2's "Feature 5/6/7"/"Tier 0/1" numbering) is not present in this repository — it isn't in `Architecture-docs/` and no file by that name exists anywhere in the tree. This plan is therefore scoped directly off `full_design_doc.md`'s own module list (§3) and its explicit dependency notes, not off feature numbers that can't be verified against a source file. If `sandbox_feature_specs.md` exists elsewhere and assigns different feature numbers/tiers to these five modules, reconcile before treating this doc as authoritative.

---

## Files to create

- `sandbox/interaction_engine.py` — `RecommendationStrategy` protocol + `AlphaSampling`
- `sandbox/stance_parser.py`
- `sandbox/vision_fallback.py`
- `sandbox/logging_writer.py`
- `sandbox/checkpoint_manager.py`
- `tests/test_interaction_engine.py`
- `tests/test_stance_parser.py`
- `tests/test_vision_fallback.py`
- `tests/test_logging_writer.py`
- `tests/test_checkpoint_manager.py`

New fixture needed: a tiny real image file under `data/memes/images/` (e.g. a 1x1-pixel JPEG) so `vision_fallback.py`'s success path has something real to read — `data/memes/test_pool.jsonl`'s `image_path` fields (`data/memes/images/test_00{1,2,3}.jpg`) currently point at files that don't exist on disk. (Resolved: `test_001.jpg` was added in Phase 3 and the remaining two in Phase 4 as defect D3, which the meme path would otherwise have hit two times in three.) No new runtime dependency: all five modules use only the stdlib plus `pydantic` (already pinned).

---

## 1. `interaction_engine.py` (§3.3)

**Build**, per spec's `RecommendationStrategy` Protocol + `AlphaSampling`:

```python
class RecommendationStrategy(Protocol):
    def select_neighbors(self, speaker: Agent, all_agents: list[Agent], turn: int, rng: random.Random) -> list[Agent]: ...

class AlphaSampling:
    def __init__(self, alpha: float, N: int): ...
    def select_neighbors(self, speaker, all_agents, turn, rng) -> list[Agent]: ...
```

Direct port of `full_design_doc.md` §3.3: `alpha=0` → uniform sampling, `alpha=1` → maximally homophilous (weighted by inverse stance-distance), blended in between via `_weighted_sample_without_replacement` (Efraimidis-Spirakis weighted reservoir sampling). `_current_stance(agent, turn)` reads `initial_stance` at `turn==1` or when `stance_history` is empty, else the last `stance_history` entry — this duplicates `Agent.current_stance()`'s logic rather than calling it, exactly as the source doc's sample code does (the doc's version doesn't route through `Agent.current_stance()`); keep it that way for fidelity unless there's a reason to consolidate, in which case note the deviation explicitly rather than silently diverging from the spec.

**Design point already resolved by the spec, not a new decision:** `ValueError` on invalid `alpha`/`N` at construction, and on insufficient candidate population at call time — never silently clamped (§3.3's own stated rationale: silent clamping is exactly the kind of implementation-level confound the HLD's critique passes warn against).

**Tests to write** (from §7.4's testing-strategy guidance plus the class's own contract):
- Construction: `alpha` outside `[0,1]` → `ValueError`; `N < 1` → `ValueError`.
- `alpha=1.0` with a population whose stances are clearly separated → deterministically returns the N closest-stance candidates (given a fixed seed).
- `alpha=0.0` → over many repeated calls with different seeds, selection frequency across candidates is roughly uniform (statistical assertion, wide tolerance).
- Never returns the speaker itself; always returns exactly N distinct agents.
- Population too small (`len(candidates) < N`) → `ValueError`, not a short list.
- Two calls with identically-seeded `random.Random` instances → identical neighbor sets (reproducibility, same pattern as Phase 2's seeded-RNG tests).
- `turn=1` always reads `initial_stance` regardless of `stance_history` contents (guards against a future bug where turn 1 accidentally reads history that shouldn't exist yet).

---

## 2. `stance_parser.py` (§3.7)

**Build**, per spec:

```python
class StanceParseFailure(Exception): ...

def parse_stance(raw_text: str) -> tuple[float, str]: ...
def clamp_and_validate_scale(value: float, stance_scale: list[float]) -> float: ...
```

Direct port: `parse_stance` regex-matches a `STANCE: <number>` line (case-insensitive), raises `StanceParseFailure` if absent or non-numeric; `reason_text` is the input with the matched `STANCE:` line stripped out. `clamp_and_validate_scale` raises `StanceParseFailure` (never silently clamps) if the parsed value falls outside `[min(stance_scale), max(stance_scale)]` — Fix I's "never fabricate a placeholder stance" principle.

This module does **not** retry — retry orchestration belongs to the Orchestrator (Tier 3), which calls this module up to 3 times per agent-turn with an amended prompt in between. Nothing in Phase 3 needs to simulate that loop; these are pure-function unit tests only.

**Tests to write** (from §7.4):
- Well-formed `"STANCE: 5\nBecause..."` → `(5.0, "Because...")`, `STANCE:` line removed from `reason_text`.
- Missing `STANCE:` line → `StanceParseFailure`.
- Non-numeric value (`"STANCE: high"`) → `StanceParseFailure`.
- Negative and decimal values parse correctly (`"STANCE: -2.5"` → `-2.5`) — regex must handle the leading `-` and decimal point.
- Case-insensitivity (`"stance: 3"` matches).
- `clamp_and_validate_scale`: value inside `[1,7]` passes through unchanged; value outside (`0` or `8`) → `StanceParseFailure`.
- Multiple `STANCE:`-like substrings in the text: confirm the first match wins and `reason_text` reflects `re.sub`'s behavior (document actual behavior rather than assuming; write the test to pin down whatever the port's regex actually does, since `full_design_doc.md` doesn't spell out multi-match behavior).

---

## 3. `vision_fallback.py` (§3.6)

**Build**, per spec:

```python
def build_meme_prompt(meme: MemeContent, receiver_supports_vision: bool) -> tuple[str, bytes | None]: ...
def _load_image(image_path: str) -> bytes: ...
```

Direct port: vision-capable receiver gets `(prompt_text_with_caption_framing, image_bytes)`; text-only receiver gets `(prompt_text_with_caption_woven_in, None)`. `_load_image` raises `FileNotFoundError` (uncaught) on a missing file — treated as a data-integrity problem with the meme pool, not something to silently degrade around, per §3.6's own error-handling note.

**Tests to write** (from §3.6's stated contract):
- `receiver_supports_vision=True` with an existing image file → returns `(prompt_text, image_bytes)` where `image_bytes` matches the file's actual contents, and `prompt_text` contains the caption.
- `receiver_supports_vision=False` → returns `(prompt_text, None)`, and the caption text is woven into `prompt_text` (not silently dropped).
- Missing image file with `receiver_supports_vision=True` → `FileNotFoundError`, uncaught.
- `receiver_supports_vision=False` never touches the filesystem at all — construct a `MemeContent` pointing at a nonexistent `image_path` and confirm `build_meme_prompt` still succeeds (the caption-only path must not call `_load_image`).

---

## 4. `logging_writer.py` (§3.8)

**Build**, per spec:

```python
class LoggingWriter:
    def __init__(self, run_dir: Path): ...
    async def write_interaction(self, interaction: Interaction) -> None: ...
    def write_run_config(self, run: ExperimentRun) -> None: ...
    def write_agents_final(self, agents: list[Agent]) -> None: ...
```

Direct port: `write_interaction` appends one `model_dump_json()` line under an `asyncio.Lock` (serializes concurrent dispatch tasks writing to the same file handle within one process); `write_run_config`/`write_agents_final` are unlocked single-shot writes called only outside the concurrent-dispatch window, per the module's own concurrency note. Disk-write failures (`OSError`) propagate uncaught — fatal, since a silently lost `Interaction` record would corrupt data integrity in a way no downstream analysis could detect.

**Tests to write** (new, since this module has no acceptance criteria spelled out anywhere beyond its docstring contract — write from the contract directly):
- `write_interaction` called N times concurrently (via `asyncio.gather`) → `interactions.jsonl` ends up with exactly N valid JSON lines, no torn/interleaved lines, regardless of write ordering (this is the actual thing the lock exists to guarantee — test it under real concurrency, not just call it N times sequentially).
- Each line round-trips via `Interaction.model_validate_json()`.
- `write_run_config` produces a file `run_config.json` that round-trips via `ExperimentRun.model_validate_json()`.
- `write_agents_final` writes exactly `len(agents)` lines, each round-tripping via `Agent.model_validate_json()`, and **overwrites** (not appends) on a second call — confirm calling it twice with different agent lists leaves only the second list's contents.
- Writing to a `run_dir` that doesn't exist yet — decide and pin down actual behavior (`full_design_doc.md`'s sample doesn't call `run_dir.mkdir()` anywhere in `logging_writer.py` itself; confirm whether that's the Orchestrator's job — likely yes, since `checkpoint_manager.save()` does its own `mkdir(parents=True, exist_ok=True)` but `LoggingWriter.__init__` doesn't per the doc's code — and write a test that documents whichever behavior is actually implemented, either "raises `FileNotFoundError` if `run_dir` doesn't exist" or "creates it," rather than leaving this undocumented).

---

## 5. `checkpoint_manager.py` (§3.14)

**Build**, per spec:

```python
class CheckpointManager:
    def __init__(self, runs_dir: Path = Path("runs")): ...
    def save(self, run_id: str, turn: int, agent_snapshot: list[Agent]) -> None: ...
    def load(self, run_id: str) -> CheckpointState | None: ...
```

Direct port: `save()` writes to `latest.json.tmp` then atomically renames to `latest.json` (`Path.replace`, atomic on POSIX) to avoid a torn checkpoint on a mid-write crash; overwrites the previous checkpoint each turn (not append-only, unlike `interactions.jsonl` — only the latest checkpoint is ever needed for resume). `load()` returns `None` if no checkpoint exists (fresh run), or raises `pydantic.ValidationError` uncaught if a checkpoint exists but fails schema validation (treated as a fatal resume failure requiring manual intervention, never silently falling back to turn 0, which would discard completed work and re-spend budget).

**Tests to write** (from §3.14's own contract):
- `load()` on a run_id with no checkpoint directory → `None`.
- `save()` then `load()` round-trips `run_id`, `last_completed_turn`, and `agent_snapshot` exactly (use `AgentManager.snapshot()` output from Phase 2 as the fixture input, since that's literally what the Orchestrator will pass here).
- Two sequential `save()` calls for the same `run_id` at different turns → `load()` returns only the latest (confirm overwrite, not accumulation).
- Corrupted/truncated `latest.json` (write malformed JSON directly to the path, bypassing `save()`) → `load()` raises, uncaught.
- `save()` creates `runs/{run_id}/checkpoints/` if it doesn't exist yet (no pre-existing directory required).
- No `.tmp` file left behind after a successful `save()` (confirm the rename actually happens, not just that `latest.json` exists).

---

## Sequencing within Phase 3

All five modules depend only on Tier 0 (`sandbox/models.py`) and, for tests only, Phase 2's `AgentManager` as a convenient fixture source (`checkpoint_manager.py`'s tests) — none depend on each other. Suggested build order, easiest/highest-value first:

1. `stance_parser.py` — pure functions, no I/O, fastest to get right and de-risk since the Orchestrator's retry loop leans on its exact failure semantics.
2. `checkpoint_manager.py` — small, self-contained, and unblocks writing a realistic resume-path test once `simulation_orchestrator.py` exists in Phase 4.
3. `logging_writer.py` — similar shape to `checkpoint_manager.py`, but needs the concurrency test (`asyncio.gather` + lock verification) called out above, so scope it as the first "needs a real concurrency test" module of this batch.
4. `interaction_engine.py` — more logic (weighted sampling), needs a small population fixture; independent of 1-3, ordered here for build-momentum reasons only, not dependency.
5. `vision_fallback.py` — last, since it's the only module needing a new binary fixture file (`data/memes/images/*.jpg`), so its test setup has the most incidental yak-shaving.

## Open items to confirm before execution

- **`prompt_builder.py` does not exist anywhere.** `full_design_doc.md` §3.9's `SimulationOrchestrator` takes a `prompt_builder: "PromptBuilder"` constructor argument and calls `self._prompt_builder.build_discussion_prompt(agent, neighbors, turn, emphasize_format=...)`, but no section of `full_design_doc.md` or `sandbox_hld.md` specifies `PromptBuilder`'s implementation, its prompt template, how persona/language_condition/memory_window/neighbor posts get woven into the prompt text, or where it lives. This is a real gap (not a Phase 3 blocker, since none of this phase's five modules touch it) but it **is** the central open question blocking Phase 4 (`simulation_orchestrator.py`) — flagging it now, the same way Phase 2 flagged and resolved its own `memory_window`/`apply_interaction()` gap (Question C1), so it isn't a surprise when Phase 4 planning starts.
- **`logging_writer.py`'s `run_dir` creation responsibility** — confirm whether `LoggingWriter.__init__` or its caller (the Orchestrator, in Phase 4) is responsible for `mkdir`, per the test note in section 4 above, before locking in that module's test assertions.
- **`interaction_engine.py`'s duplicated `_current_stance` logic** — confirm whether to port the spec's standalone `_current_stance()` function verbatim (duplicating `Agent.current_stance()`) or consolidate to call the model method directly; either is fine, but pick one deliberately rather than by accident, since the spec's sample code and the `Agent` model already disagree on this in a small way.

## Out of scope for Phase 3 (deferred to later phases)

Recorded here so the full remaining-work picture is visible in one place, not because any of it is being started now:

- **Phase 4 — `simulation_orchestrator.py` (§3.9) + `prompt_builder.py` (undesigned).** The integration tier: wires every Tier 0/1/2 module together into the per-turn loop (freeze snapshot → resolve neighbors → resolve meme injections → concurrent dispatch → advance snapshot), implements the retry-twice-then-exclude policy around `stance_parser.py`, and requires `prompt_builder.py` to be designed from scratch (see Open Items above). This is the largest remaining single module and the one most other phases wait on — it's the first point where Tier 0/1/2 modules actually get exercised together end-to-end. Blocked on nothing except the `PromptBuilder` design decision.
- **Phase 5 — Post-hoc analysis modules (§3.13):** `stance_regression.py` (Ohagi-style OLS), `coherence_scorer.py` (LLM-judge-based, Fix N filtering), `codemix_ratio_tracker.py`, `sentiment_proxy.py`, `bimodality_analysis.py`, `embedding_cluster.py`. All share a `load_run_dataframe()` loader and operate offline on a completed run's `interactions.jsonl` — no dependency on the Orchestrator itself, only on its *output format*, so this phase could in principle start once `logging_writer.py`'s `interactions.jsonl` schema is locked (Phase 3), using hand-written fixture JSONL files rather than waiting for Phase 4's real Orchestrator — worth considering as a way to parallelize Phase 4 and Phase 5 if that becomes useful later.
- **Phase 6 — CLI + Python API (§6):** `python -m sandbox {run,status,resume,list}`, `sandbox/api.py`'s `launch_run`/`get_run_status`/`load_run_dataframe`/`list_runs`, and optionally the run-manifest SQLite index (§5.4, explicitly optional/non-load-bearing) and the local HTTP status endpoint (§6.3, explicitly optional). Fully blocked on Phase 4 (needs a working Orchestrator to launch).
- **Real dataset wiring (still-open questions from `full_design_doc.md` §8, not owned by any single phase above):**
  - Meme stance-label rescaling (`MemeContent.stance_label` isn't validated against `stance_scale` — a rescaling step is needed once a real meme dataset is chosen; the current `data/memes/test_pool.jsonl` fixture already happens to be on a compatible 1-7-ish scale, so nothing here blocks Phase 3's tests).
  - A real, non-tiny persona pool (currently only `test_pool.jsonl` (4 entries) and Phase 2's `pool_20.jsonl` (20 entries, test-only) exist).
  - Real API keys/credentials for any actual (non-mocked) LiteLLM dispatch — still entirely unaddressed; every test through Phase 3 continues to run against mocks/fixtures only.
  - Confirming `pricing_table.yaml`'s model set against whichever models are actually selected to run the real experiments (the file's own header comment already flags its entries as "illustrative placeholders... not a confirmed final choice").
  - `SeedManager`'s three-stream concurrency assumption (§8) — holds today, flagged as a future risk only if neighbor-sampling and meme-injection resolution are ever parallelized against each other, which no phase through Phase 6 currently proposes doing.
