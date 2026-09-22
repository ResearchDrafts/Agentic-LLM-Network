<div align="center">

# sandbox

### LLM Agent Echo Chamber Simulation

A Python research harness that simulates populations of LLM-backed agents discussing a topic over multiple turns under a tunable echo-chamber mechanism, built to study code-mixed discourse, reasoning quality, and meme-driven polarization.

[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Pydantic](https://img.shields.io/badge/validation-pydantic%20v2-E92063?logo=pydantic&logoColor=white)](https://docs.pydantic.dev/)
[![Tests](https://img.shields.io/badge/tests-83%20passing-2f9e6e?logo=pytest&logoColor=white)](#testing)
[![Build Status](https://img.shields.io/badge/build-tier%200%2F1%2F2%20complete-4353ff)](#build-status)
[![Async](https://img.shields.io/badge/concurrency-asyncio-informational)](#architecture)
[![License](https://img.shields.io/badge/license-unreleased-lightgrey)](#license)

</div>

---

## Overview

`sandbox` seeds a population of `M` LLM-backed agents, each carrying a persona, a language condition, and a stance on a topic, then runs them through `K` discussion turns. On every turn, each scheduled speaker is shown a homophily-weighted sample of its neighbors' recent posts (an alpha-blend of similarity-weighted and uniform-random sampling, so `alpha` tunes how strong the echo chamber is), updates its stance, and posts. The simulation is built to answer three research questions from one shared run mechanism:

| | Research question |
|---|---|
| **RQ1** | Does Hindi-English code-mixed (Hinglish) discussion produce different polarization dynamics than English? |
| **RQ2** | Does the quality of an agent's stance-updating reasoning differ by language, and does it correlate with polarization speed? *(post-hoc analysis of RQ1's data)* |
| **RQ3** | Does injecting memes from a labeled dataset into the discussion accelerate polarization convergence? |

There is exactly **one simulation mode**. Memes are not a separate mode, they are a content-replacement mechanism: on a turn where a scheduled speaker is chosen to post a meme, that turn uses a dataset stance label instead of an LLM call, then gets sampled by the same echo-chamber mechanism as every other agent.

## Architecture

<div align="center">
<img src="agentic-network.png" alt="Sandbox high-level architecture" width="720">
</div>

The system is built as strict dependency tiers, where a module only depends on the tiers below it, never sideways or up.

```
Config Loader ──▶ ExperimentRun ──▶ Agent Manager creates M agents
                                          │
                          for each of K turns:
                          freeze snapshot ──▶ Interaction Engine resolves N neighbors per agent
                                          ──▶ Meme Pool Manager resolves scheduled meme speakers
                                          ──▶ dispatch concurrently
                                                 generated-text speakers ──▶ Model Gateway
                                                 meme speakers ──▶ free dataset lookup (no LLM call)
                                          ──▶ collect results, write Interaction records
                                          ──▶ advance snapshot, checkpoint ──▶ next turn
```

The frozen-snapshot ordering guarantee, that neighbor and meme-injection resolution both complete before any dispatch begins, so no agent's turn-*t* output is ever read by another agent's turn-*t* neighbor or meme resolution, is load-bearing for reproducibility and is referred to throughout the docs and codebase as **Fix F**.

### Build status

| Tier | Modules | Status |
|---|---|---|
| **Tier 0** | `models.py`, `config_loader.py`, `seed_manager.py`, `cost_tracker.py`, `rate_limiter.py` | Built and tested |
| **Tier 1** | `agent_manager.py`, `meme_pool_manager.py`, `model_gateway.py` | Built and tested |
| **Tier 2** | `interaction_engine.py`, `stance_parser.py`, `vision_fallback.py`, `logging_writer.py`, `checkpoint_manager.py` | Built and tested |
| **Tier 3** | `simulation_orchestrator.py`, `prompt_builder.py` | Planned, not yet implemented |
| **Post-hoc analysis** | `stance_regression.py`, `coherence_scorer.py`, `codemix_ratio_tracker.py`, `sentiment_proxy.py`, `bimodality_analysis.py`, `embedding_cluster.py` | Planned, not yet implemented |

Tier 3 is the integration layer, the only module that calls every other component and wires them into the per-turn loop. Post-hoc analysis modules are offline and operate only on a completed run's `interactions.jsonl`, with no dependency on the orchestrator.

### Key design invariants

- **Never silently clamp or default.** Invalid `alpha` or `N`, out-of-range parsed stances, unrecognized model ids, and missing pricing entries all raise immediately rather than being coerced to a safe value.
- **Seeded RNGs only, never the global `random` module.** `SeedManager` hands out three independently and deliberately offset seeded `random.Random` instances, one each for neighbor sampling, meme injection, and persona assignment, so the three stochastic streams never correlate.
- **Fatal vs. per-agent-recoverable errors are distinguished deliberately.** A gateway or stance-parse failure (after retry budget exhaustion) becomes a `failed_logged_null` interaction for that one agent, and the turn continues for everyone else. An unexpected exception type is meant to propagate and abort the run, not be silently absorbed.
- **`CostTracker.record()` is synchronous**, called from the async `ModelGateway.generate()` by design.
- **Pydantic v2 models throughout**, with regex-constrained enums and cross-field validators, not dataclasses.
- **Vision support is a static lookup table**, keyed against `pricing_table.yaml`, not a runtime capability probe.

See [`Architecture-docs/full_design_doc.md`](Architecture-docs/full_design_doc.md) for the module-by-module low-level design and [`Architecture-docs/sandbox_hld.md`](Architecture-docs/sandbox_hld.md) for the high-level design plus three passes of adversarial self-critique (the numbered "Fixes" referenced throughout the codebase). [`plan.md`](plan.md) and [`phase3.md`](phase3.md) track phase-by-phase build history and open design questions.

## Documentation site

A static, dependency-free docs site lives in [`docs/`](docs/index.html): project structure, architecture, per-pipeline sequence diagrams, the high-level design, and a searchable file-by-file reference. Open `docs/index.html` directly in a browser, no build step required.

Diagram source lives separately from the rendered site: the docs render pre-generated PNGs under `docs/assets/images/`, and the editable PlantUML behind each one is kept in [`Architecture-docs/plantuml/`](Architecture-docs/plantuml/). Edit the `.puml` source there, regenerate the PNG, and overwrite the matching file under `docs/assets/images/`.

## Getting started

```bash
# Clone and enter the repo
git clone git@github.com:ResearchDrafts/Agentic-LLM-Network.git
cd Agentic-LLM-Network

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

Requires **Python 3.11+**. Core dependencies: `pydantic`, `pyyaml`, `litellm`, `tenacity`, `pandas`, `statsmodels`.

## Testing

```bash
# Full test suite
python -m pytest

# A single test file
python -m pytest tests/test_agent_manager.py

# A single test
python -m pytest tests/test_agent_manager.py::test_name -q
```

`asyncio_mode = "auto"` is set in `pyproject.toml`, so `async def test_*` functions run without needing `@pytest.mark.asyncio`.

## Repository layout

```
sandbox/              Package source, organized by dependency tier
tests/                Unit tests, one file per module
data/personas/        Agent persona fixtures (JSONL)
data/memes/           Meme dataset fixtures (JSONL)
pricing_table.yaml    Hand-maintained provider/model pricing table
Architecture-docs/    HLD, LLD, and PlantUML diagram source
docs/                 Static developer documentation site
runs/                 Simulation output, one subdirectory per run
plan.md, phase3.md    Phase-by-phase build log and open design questions
```

## License

No license has been published for this repository yet. All rights reserved by the authors until one is added.
