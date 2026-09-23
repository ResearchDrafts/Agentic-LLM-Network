# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`sandbox` is a Python simulation of populations of LLM-backed agents discussing a topic over multiple turns under a tunable echo-chamber (homophilous neighbor-sampling) mechanism. It exists to answer three research questions: whether Hindi-English code-mixed (Hinglish) discussion produces different polarization dynamics than English (RQ1); whether the quality of an agent's stance-updating reasoning differs by language and correlates with polarization speed (RQ2, a post-hoc analysis of RQ1's data); and whether injecting memes from a labeled dataset into the discussion accelerates polarization convergence (RQ3). There is exactly **one simulation mode** (multi-turn); memes are a content-replacement mechanism — a meme-bearing agent's turn uses a dataset stance label instead of an LLM call, then gets sampled by the normal echo-chamber mechanism like any other agent.

The project was built in tiers, bottom-up, one phase per commit. Each phase was planned in a doc before implementation; those docs were removed in `c0facd4` and the build log now lives in git history.

## Commands

Activate the venv first: `source .venv/bin/activate` (a `.venv` already exists at repo root with all deps installed).

```bash
# Run the full test suite
python -m pytest

# Run a single test file
python -m pytest tests/test_agent_manager.py

# Run a single test
python -m pytest tests/test_agent_manager.py::test_name -q

# Reinstall the package in editable mode (after pyproject.toml changes)
pip install -e ".[dev]"
```

There is no lint/format command configured yet. `asyncio_mode = "auto"` is set in `pyproject.toml`, so `async def test_*` functions run without `@pytest.mark.asyncio`.

## Architecture

### Layered dependency tiers

The system is built as strict dependency tiers — a module only depends on tiers below it, never sideways or up. This ordering is intentional and drives both the build sequence and how you should reason about changes:

- **Tier 0** (`sandbox/models.py`, `config_loader.py`, `seed_manager.py`, `cost_tracker.py`, `rate_limiter.py`) — pure data models and standalone infrastructure. No dependency on each other.
- **Tier 1** (`agent_manager.py`, `meme_pool_manager.py`, `model_gateway.py`) — depends only on Tier 0.
- **Tier 2** (`interaction_engine.py`, `stance_parser.py`, `vision_fallback.py`, `logging_writer.py`, `checkpoint_manager.py`) — depends only on Tier 0/1, and not on each other. Built and unit-tested.
- **Tier 3** (`simulation_orchestrator.py`, `prompt_builder.py`) — the integration tier; the only module that calls every other component, wiring them into the per-turn loop. Built and unit-tested, including one end-to-end integration test. `prompt_builder.py` has no design in either `Architecture-docs/` file: it was designed from scratch, and the reasoning lives in its module docstring and in git history.
- **Post-hoc analysis** (**not yet built**: `stance_regression.py`, `coherence_scorer.py`, `codemix_ratio_tracker.py`, `sentiment_proxy.py`, `bimodality_analysis.py`, `embedding_cluster.py`) — offline, operates only on a completed run's `interactions.jsonl`, no orchestrator dependency.

Tier status drifts in prose. Don't trust a status line here; check the filesystem and `python -m pytest`. Current: **15 modules, 228 passing tests, Tiers 0-3 complete**. The post-hoc analysis tier is the only unbuilt part of the original design.

Check `Architecture-docs/full_design_doc.md` §3 for the authoritative module-by-module LLD (function signatures, error-handling contract, concurrency notes) before implementing any new module — the design doc's sample code is the spec each module is a direct port of, not just background reading. `Architecture-docs/sandbox_hld.md` carries the HLD, the schema derivation, and three passes of adversarial self-critique with numbered Fixes (Fix A–N) — when a design decision seems arbitrary (e.g. why the memory window is turn-count-based, not token-budget-based; why the same RNG seed is deliberately reused across language conditions), it's almost certainly resolving a specific numbered Fix in that document, not an accident.

### Core data flow (implemented in `simulation_orchestrator.py`)

Config Loader → `ExperimentRun` → Agent Manager creates M agents → for each of K turns: freeze snapshot → Interaction Engine resolves N neighbors per agent (`AlphaSampling`) → Meme Pool Manager resolves which scheduled speakers post a meme this turn → dispatch concurrently (generated-text speakers → Model Gateway; meme speakers → free dataset lookup, no LLM call) → collect results → write `Interaction` records → advance snapshot → checkpoint → next turn.

The frozen-snapshot ordering guarantee (neighbor and meme-injection resolution both complete *before* any dispatch begins, so no agent's turn-t output is ever read by another agent's turn-t neighbor/meme resolution) is load-bearing for reproducibility and is referred to throughout the docs and codebase as "Fix F."

### Key design invariants (apply when touching any module)

- **Never silently clamp or default.** Invalid `alpha`/`N`, out-of-range parsed stances, unrecognized model ids, missing pricing entries — all raise immediately (`ValueError`/custom exceptions) rather than being coerced to a safe value. This is a deliberate, repeated pattern across the codebase (see Fix I: "never fabricate a placeholder stance") — don't add fallback/default behavior to "fix" a raised error unless the design doc says to.
- **Seeded RNGs only, never the global `random` module.** `seed_manager.py`'s `SeedManager` hands out three independently-seeded `random.Random` instances (`neighbor_sampling_rng`, `meme_injection_rng`, `persona_assignment_rng`), deliberately offset (`seed`, `seed+1`, `seed+2`) so the three streams never correlate. Reproducibility depends on every stochastic choice consuming from the correct purpose-specific instance.
- **Fatal vs. per-agent-recoverable errors are distinguished deliberately.** `FatalGatewayError`/`StanceParseFailure` (after retry budget exhausted) become a `failed_logged_null` `Interaction` for that one agent and the turn continues for everyone else (`asyncio.gather(..., return_exceptions=True)`); anything else propagates and aborts the whole run. Don't broaden what gets caught — an uncaught, unexpected exception type is meant to surface loudly as a bug, not be absorbed into a soft-failure record.
- **`CostTracker.record()` is synchronous**, not async, even though it's called from `ModelGateway.generate()` (async) — this was a deliberate early decision, not an oversight. Don't add `await` to that call site.
- **Pydantic v2 models throughout** (`sandbox/models.py`), not dataclasses — the HLD's own sketches use dataclasses, but the codebase supersedes that with `pydantic.BaseModel` for validation (regex-constrained enums via `pattern=`, cross-field `@model_validator`). When porting a design-doc snippet, translate its dataclasses to the existing Pydantic models rather than introducing a parallel dataclass.
- **Vision support is a static lookup table, not a runtime probe**, keyed by `pricing_table.yaml`'s model set — keep the vision-capable/text-only lists in `model_gateway.py` and the pricing table in sync; an unrecognized `model_backend_id` should raise, never silently default to text-only.

### Data and fixtures

- `data/memes/*.jsonl` — one JSON object per line, loaded via `MemeContent.model_validate_json()`.
- `data/personas/*.jsonl` — **despite the extension, these are plain text, one free-text persona description per line, not JSON**. `agent_manager.py`'s `_load_persona_pool()` just strips and splits lines; nothing parses JSON. Don't add `model_validate_json()` here.
- `test_pool.jsonl` variants are small (3-4 entries) fixtures for tests; `pool_20.jsonl`/`empty_pool.jsonl` exist for specific acceptance-criteria tests (cycling, empty-pool error path). All three images referenced by `data/memes/test_pool.jsonl` now exist; two were missing until a pre-flight audit found the meme path would raise `FileNotFoundError` for roughly two sampled memes in three.
- `pricing_table.yaml` — hand-maintained `provider -> model_id -> modality -> {prompt_per_1k, completion_per_1k}`, keyed by the BARE model id (`ModelGateway._bare_model_id` strips the provider prefix). `CostTracker` raises on a missing entry *after* the call completes, so every model actually used must appear here. The openai/anthropic/google entries remain illustrative placeholders; the `hosted_vllm`, `ollama`, `groq` and `cerebras` entries are genuinely 0.0, since self-hosted compute is billed by the hour rather than per token.
- `runs/` — output directory, one run = one subdirectory (`run_config.json`, `interactions.jsonl`, `agents_final.jsonl`, `checkpoints/`), per `logging_writer.py`. Gitignored. Completed campaigns are kept under `outputs/` instead; `outputs/chorus_run1/` holds the first real dataset, 8 runs and 8,000 interactions.
- `configs/` — run configs. `example_run.yaml` documents every field; `smoke_test.yaml` (vLLM) and `smoke_local.yaml` (Ollama on Apple Silicon) are minimal five-call runs for checking plumbing.

### Where to look before starting new work

The two `Architecture-docs/` files are the stable spec: `full_design_doc.md` §3
for the module-by-module LLD, `sandbox_hld.md` for the HLD and the numbered
Fixes. **Both predate the code and contain errors that were found by building
it** (see the invariants above), so treat them as design intent, not as truth
about current behaviour.

The phase-by-phase build log now lives in **git history** rather than the
working tree; the phase docs were deleted in `c0facd4`. `git log --oneline`
walks the tiers in order, and the commit messages carry the design decisions and
the defects found, with their reasoning. Twelve defects were found and fixed
during Phase 4 and a pre-flight audit, several of them faithful ports of errors
in the design doc's own sample code.

### Running things

```bash
python run_simulation.py configs/smoke_test.yaml --dry-run   # validate, no calls
python run_simulation.py configs/smoke_test.yaml             # a real run
python export_csv.py outputs/chorus_run1                     # runs -> analysis CSVs
```

`notebooks/chorus_colab.ipynb` runs a full campaign on a Colab GPU: vLLM
serving, the topic x language grid, and the combining step. Inference is
self-hosted on open-weight models, so a campaign costs GPU time and nothing
else.

Two hazards worth knowing before touching the Colab path: `asyncio.run()` raises
inside a notebook cell, so runs launch as subprocesses or via top-level `await`;
and the Colab filesystem is ephemeral while `runs/` holds the checkpoint resume
depends on, so `--runs-dir` must point at mounted Drive.

### Known gaps

- **Post-hoc analysis does not exist.** No regression, bimodality, code-mix
  index, sentiment, embedding, or network analysis. `export_csv.py` is the only
  analysis tooling.
- **Resumed runs can duplicate rows.** The orchestrator writes interactions one
  at a time and checkpoints only after a full turn, so a crash mid-turn leaves
  rows with no checkpoint and the resumed pass re-logs that turn.
  `export_csv.py` deduplicates on read, keeping the later write, which
  `agents_final.jsonl` confirms is authoritative. The orchestrator itself is
  unfixed.
- **Prompt length differs by language.** Measured at 693 tokens for English and
  809 for Hinglish, a 17% gap, because code-mixed `reason_text` tokenizes less
  efficiently and feeds back into memory and neighbour blocks. Fix A's
  turn-count guarantee still holds; the information content per turn does not.
- **No real persona pool.** Only `test_pool.jsonl` (4) and `pool_20.jsonl` (20).
  At M=100 each persona is shared by five agents.
