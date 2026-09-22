# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`sandbox` is a Python simulation of populations of LLM-backed agents discussing a topic over multiple turns under a tunable echo-chamber (homophilous neighbor-sampling) mechanism. It exists to answer three research questions: whether Hindi-English code-mixed (Hinglish) discussion produces different polarization dynamics than English (RQ1); whether the quality of an agent's stance-updating reasoning differs by language and correlates with polarization speed (RQ2, a post-hoc analysis of RQ1's data); and whether injecting memes from a labeled dataset into the discussion accelerates polarization convergence (RQ3). There is exactly **one simulation mode** (multi-turn); memes are a content-replacement mechanism — a meme-bearing agent's turn uses a dataset stance label instead of an LLM call, then gets sampled by the normal echo-chamber mechanism like any other agent.

The project is being built in tiers, bottom-up, one phase per PR/commit. Each phase is documented in a planning file at the repo root before it's implemented (see `plan.md`, `phase3.md`).

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
- **Tier 3** (**not yet built**: `simulation_orchestrator.py` + `prompt_builder.py`) — the integration tier; the only module that calls every other component, wiring them into the per-turn loop. `prompt_builder.py` has no design in either `Architecture-docs/` file; `phase4.md` is where that design is resolved.
- **Post-hoc analysis** (**not yet built**: `stance_regression.py`, `coherence_scorer.py`, `codemix_ratio_tracker.py`, `sentiment_proxy.py`, `bimodality_analysis.py`, `embedding_cluster.py`) — offline, operates only on a completed run's `interactions.jsonl`, no orchestrator dependency.

Tier status drifts in these docs because each phase's planning doc is written before its code and committed alongside it, then never revised. Don't trust a status line here or in a phase doc; check the filesystem and `python -m pytest`. As of Phase 3 merging: 13 modules, 83 passing tests, Tiers 0-2 complete.

Check `Architecture-docs/full_design_doc.md` §3 for the authoritative module-by-module LLD (function signatures, error-handling contract, concurrency notes) before implementing any new module — the design doc's sample code is the spec each module is a direct port of, not just background reading. `Architecture-docs/sandbox_hld.md` carries the HLD, the schema derivation, and three passes of adversarial self-critique with numbered Fixes (Fix A–N) — when a design decision seems arbitrary (e.g. why the memory window is turn-count-based, not token-budget-based; why the same RNG seed is deliberately reused across language conditions), it's almost certainly resolving a specific numbered Fix in that document, not an accident.

### Core data flow (per `simulation_orchestrator.py`'s design; the orchestrator itself is not yet built)

Config Loader → `ExperimentRun` → Agent Manager creates M agents → for each of K turns: freeze snapshot → Interaction Engine resolves N neighbors per agent (`AlphaSampling`) → Meme Pool Manager resolves which scheduled speakers post a meme this turn → dispatch concurrently (generated-text speakers → Model Gateway; meme speakers → free dataset lookup, no LLM call) → collect results → write `Interaction` records → advance snapshot → checkpoint → next turn.

The frozen-snapshot ordering guarantee (neighbor and meme-injection resolution both complete *before* any dispatch begins, so no agent's turn-t output is ever read by another agent's turn-t neighbor/meme resolution) is load-bearing for reproducibility and is referred to throughout the docs and codebase as "Fix F."

### Key design invariants (apply when touching any module)

- **Never silently clamp or default.** Invalid `alpha`/`N`, out-of-range parsed stances, unrecognized model ids, missing pricing entries — all raise immediately (`ValueError`/custom exceptions) rather than being coerced to a safe value. This is a deliberate, repeated pattern across the codebase (see Fix I: "never fabricate a placeholder stance") — don't add fallback/default behavior to "fix" a raised error unless the design doc says to.
- **Seeded RNGs only, never the global `random` module.** `seed_manager.py`'s `SeedManager` hands out three independently-seeded `random.Random` instances (`neighbor_sampling_rng`, `meme_injection_rng`, `persona_assignment_rng`), deliberately offset (`seed`, `seed+1`, `seed+2`) so the three streams never correlate. Reproducibility depends on every stochastic choice consuming from the correct purpose-specific instance.
- **Fatal vs. per-agent-recoverable errors are distinguished deliberately.** `FatalGatewayError`/`StanceParseFailure` (after retry budget exhausted) become a `failed_logged_null` `Interaction` for that one agent and the turn continues for everyone else (`asyncio.gather(..., return_exceptions=True)`); anything else propagates and aborts the whole run. Don't broaden what gets caught — an uncaught, unexpected exception type is meant to surface loudly as a bug, not be absorbed into a soft-failure record.
- **`CostTracker.record()` is synchronous**, not async, even though it's called from `ModelGateway.generate()` (async) — this was a deliberate Phase 1 decision (see `plan.md`'s note), not an oversight. Don't add `await` to that call site.
- **Pydantic v2 models throughout** (`sandbox/models.py`), not dataclasses — the HLD's own sketches use dataclasses, but the codebase supersedes that with `pydantic.BaseModel` for validation (regex-constrained enums via `pattern=`, cross-field `@model_validator`). When porting a design-doc snippet, translate its dataclasses to the existing Pydantic models rather than introducing a parallel dataclass.
- **Vision support is a static lookup table, not a runtime probe**, keyed by `pricing_table.yaml`'s model set — keep the vision-capable/text-only lists in `model_gateway.py` and the pricing table in sync; an unrecognized `model_backend_id` should raise, never silently default to text-only.

### Data and fixtures

- `data/memes/*.jsonl` — one JSON object per line, loaded via `MemeContent.model_validate_json()`.
- `data/personas/*.jsonl` — **despite the extension, these are plain text, one free-text persona description per line, not JSON**. `agent_manager.py`'s `_load_persona_pool()` just strips and splits lines; nothing parses JSON. Don't add `model_validate_json()` here.
- `test_pool.jsonl` variants are small (3-4 entries) fixtures for tests; `pool_20.jsonl`/`empty_pool.jsonl` exist for specific acceptance-criteria tests (cycling, empty-pool error path). Note `data/memes/test_pool.jsonl` references three images but only `data/memes/images/test_001.jpg` exists on disk.
- `pricing_table.yaml` — hand-maintained `provider -> model_id -> modality -> {prompt_per_1k, completion_per_1k}`. Its own header notes the model ids are illustrative placeholders, not a confirmed final model selection — don't treat entries here as validated production pricing.
- `runs/` — output directory, one run = one subdirectory (`run_config.json`, `interactions.jsonl`, `agents_final.jsonl`, `checkpoints/`), per `logging_writer.py`. Currently empty except `.gitkeep`, since no orchestrator exists yet to launch a run.

### Where to look before starting new work

Check the phase docs — `plan.md` (Phase 2), `phase3.md` (Phase 3), `phase4.md` (Phase 4) — for the concrete record of design decisions and open questions. The architecture docs are the stable spec; the phase docs are the build log and the place where small spec gaps get resolved before code is written.

**But ignore their `Status:` headers.** Every phase doc is written before its code and committed alongside it, so each one still says "planning only, not started" after shipping. `plan.md` and `phase3.md` are both merged and tested despite what their line 3 claims. `phase4.md` is the one that is genuinely not started.

If you're implementing a new module, follow the same pattern: state open design questions and proposed resolutions in a phase doc before writing code, the way Phase 2 resolved `memory_window`/`apply_interaction()` semantics ("Question C1") and Phase 4 resolves the `prompt_builder.py` design (Q4.1-Q4.10). `phase4.md` also carries a defect list (D1-D7) of real bugs found in Tiers 0-2, two of which would abort the first live run — read it before touching `model_gateway.py` or `config_loader.py`.
