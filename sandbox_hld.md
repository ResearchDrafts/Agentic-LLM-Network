# LLM Agent Echo Chamber Sandbox — High-Level Design (Revised: RQ1–RQ3 Scope)

---

# PART 1: INITIAL HIGH-LEVEL DESIGN

## 1. System Architecture

Six components, one simulation mode. The original two-mode split (multi-turn vs. single-exposure) is gone: with the old matched-pair meme RQ replaced by in-loop meme injection, there is no longer any RQ that needs a mode outside the multi-turn discussion loop.

```
                        ┌─────────────────────────┐
                        │   Config / Run Loader    │
                        │  (topic, alpha, M,N,K,   │
                        │   model, lang, seed,     │
                        │   meme_injection block)  │
                        └────────────┬─────────────┘
                                     │
                    ┌────────────────┴─────────────────┐
                    │      Simulation Orchestrator       │
                    │   (single mode: multi-turn,        │
                    │    RQ1 / RQ2 / RQ3 all run here)   │
                    └────────────────┬────────────────────┘
                                     │
          ┌───────────────────┬──────┴──────┬──────────────────┐
          │                   │              │                  │
┌─────────┴──────────┐ ┌──────┴──────────┐ ┌─┴────────────────┐│
│   Agent Manager     │ │ Interaction/    │ │  Meme Pool       ││
│ (creates/holds Agent│ │ Recommendation  │ │  Manager         ││
│  objects, persona,  │ │ Engine (alpha-  │ │ (loads pre-      ││
│  stance, memory)    │ │ sampling)       │ │  existing meme   ││
└─────────┬────────────┘ └──────┬──────────┘ │  dataset items,  ││
          │                     │             │  exposes them    ││
          │                     │             │  for injection)  ││
          │                     │             └─┬────────────────┘│
          └───────────┬──────────┴───────────────┘                │
                       │                                           │
          ┌────────────┴──────────────────────────────────────────┘
          │           Model Backend Abstraction Layer
          │  (LiteLLM-based; uniform call contract; text vs
          │   vision-capable dispatch; retry/rate-limit/cost)
          └────────────────────────┬───────────────────────────┘
                                    │
                       ┌────────────┴────────────┐
                       │   External Model APIs     │
                       │ OpenAI / Anthropic /       │
                       │ Google / Local (Llama)     │
                       └────────────────────────────┘

          ┌──────────────────────────────────────────────────┐
          │            Logging / Persistence Layer              │
          │  (every component writes here; append-only,        │
          │   flat files, one run = one directory)              │
          └────────────────────────┬─────────────────────────┘
                                    │
                       ┌────────────┴────────────┐
                       │   Post-hoc Analysis       │
                       │   Module (offline,        │
                       │   separate from sim run)  │
                       │  - bimodality/convergence  │
                       │    analysis                │
                       │  - Ohagi-style regression   │
                       │  - coherence/code-mix/      │
                       │    sentiment diagnostics    │
                       └────────────────────────────┘
```

**Data flow (single mode, serves RQ1/RQ2/RQ3):** Config Loader instantiates a run → Agent Manager creates M agents with persona/stance/language-condition assignment → for each of K turns: Interaction Engine samples N neighbors per agent per Ohagi's alpha function → for each agent selected to post this turn, the Meme Pool Manager determines (per `meme_injection` config) whether this agent's post is a generated response or a meme drawn from the pool → Agent Manager (generated case) or Meme Pool Manager (meme case) supplies the content → Model Backend Layer dispatches the call (generated case) or, if the meme condition requires the *receiving* agents to react to a meme, the Backend Layer dispatches the receivers' reaction calls with the meme as stimulus content → response (stance + reason) is parsed and written to agent state AND to the Logging Layer as an Interaction record → next turn repeats, now using updated state.

**Why the old mode split is gone:** the previous design reserved a separate single-exposure mode specifically because the old RQ3/RQ4 (matched meme-vs-text-paraphrase comparison) needed a categorically different population (one disposable receiver per exposure, no discussion loop, no neighbor sampling). That RQ no longer exists. The current RQ3 — does meme presence in an ongoing discussion accelerate convergence to a polarized distribution, compared to text-only discussions — is answered by running the *same* multi-turn loop RQ1 uses, with memes injected as content at some points and not at others, and comparing convergence speed across those two conditions. There is no remaining reason to maintain a second Orchestrator path.

---

## 2. Agent Population Sizing, Per RQ

| RQ | Mode | M | N | K | Trials/cell | Justification |
|---|---|---|---|---|---|---|
| RQ1 | Multi-turn | 100 | 5 | 10 | 5 | Matches Ohagi's baseline exactly (M=100, N=5, K=10) since RQ1 is a direct extension of his design with language as the new variable. No reason to deviate — changing M/N/K here would confound "language effect" with "population-size effect" relative to the paper you're extending. |
| RQ2 | Multi-turn (same runs as RQ1) | 100 | 5 | 10 | 5 | RQ2 is a post-hoc diagnostic computed on RQ1's *same transcripts* (coherence, code-mixing ratio, sentiment-proxy scoring per turn). No separate simulation runs needed — this is an Analysis Module addition, not a new sandbox mode. |
| RQ3 | Multi-turn (same core design as RQ1) | 100 | 5 | 10 | 5 | RQ3 reuses RQ1's exact M=100/N=5/K=10 design. It does **not** need a separate, smaller population design the way the old single-exposure mode did, because RQ3 is no longer a one-shot stimulus-response measurement — it's a question about how fast a *population* of agents converges under otherwise-identical echo chamber conditions, with `meme_injection.enabled` as the manipulated variable instead of, or alongside, `language_condition`. Holding M/N/K/alpha/topic fixed across meme-present and meme-absent runs is exactly what makes "meme presence accelerates convergence" a controlled claim rather than an artifact of population-size differences between conditions. |

**Net simulation modes needed: 1** (multi-turn only). **Net distinct simulation *campaigns*: 2** (RQ1/RQ2 language campaign; RQ3 meme-injection campaign) — RQ2 adds no new runs, only new analysis code. RQ3 can optionally share topic/alpha/trial settings with RQ1 and simply add the `meme_injection` config block as an additional swept variable, meaning in practice RQ1 and RQ3 may be run as a single combined sweep (language condition × meme-injection condition) rather than two fully separate campaigns — this is a config-level choice, not an architectural one, and is discussed further in Critique Pass 2.

---

## 3. Data Model / Schema

### 3.1 Agent

```python
@dataclass
class Agent:
    agent_id: str                    # uuid4, e.g. "agent_0042"
    run_id: str                      # foreign key to ExperimentRun
    persona: str                     # free text, persona description used in system prompt
    language_condition: str          # enum: "english" | "hinglish" | "hindi"
    model_backend_id: str            # e.g. "gpt-4o" — single backend per run in this scope
                                      # (no cross-model sweep; see Removed Components below)
    initial_stance: float            # numeric stance on Ohagi's finite-category scale (e.g. 1-7)
    stance_history: list[StanceRecord]   # append-only, one entry per turn
    memory_window: list[str]         # rolling buffer of interaction_ids this agent has seen
                                      # (raw text NOT duplicated here — pointer into Interaction log,
                                      #  fixed-turn-count policy, see Fix A)
    created_at_turn: int             # always 0
```

There is only one agent type in this scope. The earlier revision's split into `DiscussionAgent`/`ExposureReceiver` (Fix C) existed solely to separate the multi-turn population from single-exposure mode's disposable receivers. With single-exposure mode removed entirely, every agent in the sandbox is a full `Agent` as defined above — the split is no longer needed and has been removed.

### 3.2 StanceRecord (embedded in Agent, also independently logged — see §7)

```python
@dataclass
class StanceRecord:
    turn: int
    stance_value: float
    reason_text: str                 # free-text justification, the raw model output;
                                      # empty string if this turn's post was a meme injection
                                      # (see content_type in Interaction, §3.3)
    interaction_id: str              # links to the Interaction record this stance update came from
```

### 3.3 Interaction (the core record — one row per agent-turn)

```python
@dataclass
class Interaction:
    interaction_id: str              # uuid4
    run_id: str
    turn: int
    speaker_agent_id: str
    neighbor_agent_ids: list[str]    # the N agents sampled this turn whose posts speaker saw
    stance_before: float
    stance_after: float
    reason_text: str
    content_type: str                # NEW: "generated_text" | "meme"
    meme_id: str | None              # NEW: FK to MemeContent (§3.4), populated iff content_type == "meme"
    used_vision_fallback: bool       # NEW: True if a text-only backend received the meme's
                                      # caption-only fallback rather than the full image+caption
    prompt_token_count: int          # for cost tracking + confound checks (Critique Pass 3)
    completion_token_count: int
    model_backend_id: str
    latency_ms: int
    api_call_status: str             # "success" | "retried_success" | "failed_logged_null"
    timestamp_utc: str
```

`content_type`, `meme_id`, and `used_vision_fallback` are the three fields added in this revision. They exist so meme-present vs. meme-absent turns, and vision-fallback substitutions specifically, are directly queryable from the logged data rather than requiring cross-referencing against run-level config — this matters because `meme_injection` can vary *within* a run at the agent-turn level (per `injection_rate`), not just at the run level, so a run-level flag alone would not tell you which specific turns were meme turns.

### 3.4 MemeContent (replaces the old MemeStimulus/matched-pair schema)

```python
@dataclass
class MemeContent:
    meme_id: str
    image_path: str
    caption_text: str
    stance_label: float              # pre-existing label from the source dataset —
                                      # used directly as this meme's "stance" for
                                      # Interaction Engine sampling purposes (§6)
    offensiveness_label: float | None  # pre-existing label from the source dataset, if available
    source_dataset: str               # provenance, e.g. name/version of the pre-existing dataset used
```

This schema is deliberately smaller than the old `MemeStimulus`. It has no `pair_id` (there is no matched text paraphrase to link to — memes are simply content items competing for a "slot" in the discussion, not one half of an engineered comparison pair), no `modality` field (every `MemeContent` record is a meme by definition; the text-vs-meme comparison now lives at the *turn* level via `Interaction.content_type`, not at the stimulus-record level), and no `intensity_label` (not assumed to exist in the pre-existing dataset's metadata; if your chosen dataset does provide one, add it back as an optional field — it isn't required for RQ3 as currently scoped since RQ3 asks about presence/absence of memes, not graded intensity).

### 3.5 ExperimentRun (top-level config/metadata, one per simulation execution)

```python
@dataclass
class ExperimentRun:
    run_id: str                      # uuid4, also the output directory name
    rq_target: str                   # "RQ1_RQ2" | "RQ3"  — which campaign this run belongs to
                                      # (may be a combined "RQ1_RQ2_RQ3" if language and meme-injection
                                      #  are swept together in one campaign — see §2)
    mode: str                        # "multi_turn" — the only value; field retained for schema
                                      # stability / forward compatibility, not because a second
                                      # value currently exists
    topic: str
    alpha: float
    M: int
    N: int
    K: int
    trial_number: int
    language_condition: str          # for RQ1/RQ2 campaign runs; still required even for
                                      # meme-only runs (set to a fixed baseline, e.g. "english")
                                      # so meme-injection runs remain comparable to RQ1's baseline
    model_backend_id: str            # single string, not a list — no cross-model sweep in this scope
    meme_injection: MemeInjectionConfig   # see below; required field, set enabled=false for
                                            # pure RQ1/RQ2 runs
    seed: int                        # feeds neighbor-sampling RNG, persona assignment, and
                                      # (new) which agents/turns receive a meme injection —
                                      # NOT an LLM sampling seed (providers don't guarantee
                                      # determinism even at temperature=0)
    temperature: float
    git_commit_hash: str
    started_at_utc: str
    completed_at_utc: str | None
    status: str                      # "running" | "completed" | "failed_partial" | "failed_total"
    total_cost_usd: float
```

```python
@dataclass
class MemeInjectionConfig:
    enabled: bool
    meme_pool_id: str | None         # required if enabled == True; resolves to a MemeContent dataset
    injection_rate: float            # fraction of turns/agents assigned a meme instead of a
                                      # generated response; must be in [0, 1]
    injection_schedule: str          # enum: "random" | "fixed_turn"
                                      # "random": each eligible (agent, turn) pair is independently
                                      #   assigned a meme with probability injection_rate
                                      # "fixed_turn": memes are introduced only at a specific,
                                      #   pre-declared turn (or set of turns), letting you test
                                      #   whether *when* a meme enters the discussion matters
```

The old `capability_tier_mapper`-adjacent framing (a list of `model_backends` to sweep) is gone from `ExperimentRun`: `model_backend_id` is now a single string. This scope has no RQ that compares across model families, so there is nothing for a list-valued field to serve.

---

## 4. Modules to Build

| Module | Purpose (one line) | Inputs → Outputs | RQ dependency |
|---|---|---|---|
| `config_loader.py` | Parse a run config (YAML/JSON) into an `ExperimentRun` object, validate required fields | config file → `ExperimentRun` | All |
| `agent_manager.py` | Create, hold, and mutate `Agent` objects for a run | `ExperimentRun`, persona pool → `list[Agent]` | RQ1, RQ2, RQ3 |
| `meme_pool_manager.py` | **NEW.** Load a pre-existing meme dataset into `MemeContent` records, expose lookup/sampling of pool items to the Orchestrator per `meme_injection` config | dataset files, `MemeInjectionConfig` → `list[MemeContent]`, plus a `sample_meme(rng) -> MemeContent` method | RQ3 |
| `interaction_engine.py` | Given current agent (and meme) stances, select N neighbors per agent per turn using a swappable strategy | `list[Agent]`, `alpha`, turn number → `dict[agent_id, list[neighbor_id]]` | RQ1, RQ2, RQ3 |
| `recommendation_strategies.py` | Concrete strategy implementations: `AlphaSampling` (default), stub `IdeologyProximityScoring` for future extension | strategy-specific | RQ1, RQ2, RQ3 |
| `prompt_builder.py` | Construct the actual text/multimodal prompt sent to a model, given an agent's persona, memory, sampled neighbor posts (which may include memes), and language condition | `Agent`, `list[Interaction]` (neighbor posts, possibly meme-flagged) → formatted prompt (text) or (text, image) tuple | RQ1, RQ2, RQ3 |
| `model_backend.py` | Uniform interface over LiteLLM; dispatches text-only or vision calls; enforces per-backend rate limits; retries with backoff; converts to `Interaction`-ready output | prompt (+ optional image) → raw response, token counts, latency | RQ1, RQ2, RQ3 |
| `vision_fallback.py` | For text-only backends whose prompt this turn includes a meme (a sampled neighbor posted a meme), substitutes `caption_text` alone in place of image+caption, tagged so this substitution is logged distinctly. **Reframed from the prior revision:** previously built to handle the old RQ3's single-exposure receiver seeing a meme stimulus directly; now handles the same substitution logic but triggered whenever a meme appears in a *neighbor's* post inside the ongoing multi-turn loop, not in a standalone exposure step. | `MemeContent`, `model_backend_id` → adjusted prompt path | RQ3 |
| `stance_parser.py` | Extract structured stance value + reason text from raw model output (handles parsing failures) | raw completion text → `(stance_value, reason_text)` or parse-failure flag | RQ1, RQ2, RQ3 |
| `simulation_orchestrator.py` | Drives the K-turn loop (the single mode); the only module that calls Agent Manager / Interaction Engine / Meme Pool Manager / Model Backend in sequence | `ExperimentRun` → completed run, all records written to Logging Layer | RQ1, RQ2, RQ3 |
| `logging_writer.py` | Append-only writer for all record types to disk | records → files on disk | All |
| `checkpoint_manager.py` | Periodically snapshot full run state so a failed run can resume rather than restart | run state → checkpoint file; checkpoint file → resumed run state | All |
| `cost_tracker.py` | Track $ cost per API call using per-provider/per-model/per-modality pricing table, alert on threshold | token counts + model_backend_id + modality → running `total_cost_usd` | All |
| `rate_limiter.py` | Per-provider token-bucket rate limiting | provider name, call → throttled dispatch | All |
| `seed_manager.py` | Central RNG seeding for all non-LLM stochastic choices (neighbor sampling, persona assignment, meme-injection assignment) | seed int → seeded `random.Random` instances | All |
| `stance_regression.py` | Ohagi-style linear regression: stance-before vs neighbor-average vs stance-after | Interaction log → regression coefficients, per run/condition | RQ1 |
| `coherence_scorer.py` | Post-hoc NLP diagnostic: coherence/consistency scoring of reason_text, calibrated for Hinglish vs English (Fix B: LLM-as-judge rubric, fixed judge model) | reason_text (+ language_condition) → coherence score | RQ2 |
| `codemix_ratio_tracker.py` | Post-hoc: compute code-mixing ratio (CMI) per reason_text | reason_text → ratio | RQ1, RQ2 |
| `sentiment_proxy.py` | Post-hoc: sentiment/emotion scoring that works on Hinglish (Fix B: multilingual transformer, not VADER) | reason_text → sentiment/emotion score | RQ2 |
| `bimodality_analysis.py` | Post-hoc: fit/test for bimodal vs unimodal final stance distribution, and convergence-speed (turns-to-bimodal) per run | final stance distribution (or per-turn distribution series) → bimodality metric + classification | RQ1 (as a secondary polarization measure), RQ3 (primary metric — meme-present vs. meme-absent convergence speed) |
| `embedding_cluster.py` | Sentence-BERT/SimCSE clustering of reason_text (Ohagi's method), multilingual-capable embedding model for Hinglish support | reason_text corpus → cluster assignments | RQ1, RQ2 |

**Removed from the original module list, and why:**
- `capability_tier_mapper.py` — existed solely to support the old RQ5 (capability-tier vs. polarization susceptibility). No RQ in this scope compares across models or capability tiers.
- `exposure_receiver_manager.py`, `stimulus_manager.py` — existed solely to support the old single-exposure mode and its matched meme/text-paraphrase pairing. Replaced by `meme_pool_manager.py`, which is simpler: it loads a pre-existing dataset and exposes items for injection, with no pairing or receiver-assignment logic needed.
- `graph_export.py`, `network_metrics.py` — the original document listed these for RQ1/RQ4/RQ5's structural analysis using the 12 Impiccichè & Viviani network metrics. RQ3 as reframed in this scope is about convergence *speed and shape* (bimodality), not interaction-graph structure, and neither RQ1 nor RQ2 in this scope requires network-based metrics either (RQ1's validation table in the prior revision listed them, but they are not load-bearing for RQ1's actual claim — Ohagi-style regression and bimodality/embedding-cluster analysis suffice). These modules are not deleted from the codebase in spirit — the Interaction log still contains everything needed to build them later (`neighbor_agent_ids` is still a valid edge list) — but they are out of scope for this three-RQ build and are not listed as required modules here. If network-structure analysis becomes desirable later, it can be added without any schema changes.

---

## 5. Model Backend Abstraction

**Contract every backend must satisfy** (unchanged from the original design):

```python
class ModelBackend(Protocol):
    supports_vision: bool
    provider_name: str          # "openai" | "anthropic" | "google" | "local"
    model_id: str

    def generate(
        self,
        prompt_text: str,
        image: bytes | None = None,
        temperature: float = 0.7,
    ) -> BackendResponse:
        ...

@dataclass
class BackendResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    raw_provider_response: dict
```

**Implementation:** a single `LiteLLMBackend` class implements this contract. In this scope, a given `ExperimentRun` uses exactly one `model_backend_id` (§3.5) — there is no cross-model sweep to support, so the abstraction's main value here is *not* "swap models across runs for comparison" (that was RQ4/RQ5's use case) but simply "keep the calling code identical regardless of which single backend a given campaign happens to use," which still matters because RQ1/RQ2 and RQ3 may reasonably be run against different backends in different campaigns without requiring different orchestrator code.

**Vision vs. text-only dispatch, reframed for in-loop meme injection:**
- When `prompt_builder.py` constructs a speaker's prompt and that prompt includes a sampled neighbor's meme post (i.e., one of `neighbor_agent_ids`' most recent `Interaction` had `content_type == "meme"`), it constructs both representations: the multimodal version (image + caption) and the caption-only fallback.
- `model_backend.py` checks the *speaker's* `backend.supports_vision` before dispatch (not the meme-posting agent's — the meme was posted by whichever agent's turn assigned it a meme, but it's the *receiving* speaker's backend capability that determines whether they can actually see the image). If `True`, sends the multimodal version. If `False`, sends the caption-only fallback via `vision_fallback.py`, and tags the resulting `Interaction.used_vision_fallback = True`.
- Same open design question as before, same default: since this scope uses a single `model_backend_id` per run, this only becomes relevant if that backend is itself text-only — in which case *every* meme encounter in that run uses the fallback, which is a simpler situation than the original design's mixed-backend case. If you run RQ3 against a vision-capable backend (recommended default, given the fallback path degrades meme content to text-only and would otherwise make RQ3's meme-vs-text comparison partially moot), `used_vision_fallback` will simply always be `False` for that run, and the field exists mainly as a safeguard/audit trail rather than an actively-triggered path.

---

## 6. Interaction / Recommendation Engine Design

```python
class RecommendationStrategy(Protocol):
    def select_neighbors(
        self,
        speaker: Agent,
        all_agents: list[Agent],
        turn: int,
        rng: random.Random,
    ) -> list[Agent]:
        ...

class AlphaSampling(RecommendationStrategy):
    """Ohagi's mechanism: probability of sampling an agent is a function of
    stance similarity to speaker, controlled by alpha (0 = random/open,
    1 = maximally homophilous/closed)."""
    def __init__(self, alpha: float, N: int): ...
    def select_neighbors(self, speaker, all_agents, turn, rng): ...
```

`IdeologyProximityScoring` is retained as a stub for future extension but is not required by any RQ in this scope; it is not detailed further here.

**Meme injection and the Interaction Engine — the key design decision for this revision:**

Memes enter a turn by **replacing an agent's generated post**, not by being introduced as a separate exogenous content pool that agents are shown independently of the normal sampling process. Concretely: at the start of a turn, for each agent scheduled to "post" this turn, the Orchestrator consults `MemeInjectionConfig` (via `meme_pool_manager.py`) to decide whether that agent's post this turn is a generated response (normal case) or a meme drawn from the pool (injected case, at rate `injection_rate`, per `injection_schedule`). If a meme is selected, that agent's `Interaction.content_type = "meme"`, `meme_id` is set, `stance_after` is set directly to the meme's `stance_label` (no LLM call is needed to produce a stance for the meme-posting agent's own turn, since the meme already carries a fixed label), and `reason_text` is left empty or set to the meme's `caption_text`.

This meme-bearing agent is then sampled by `AlphaSampling.select_neighbors()` in subsequent turns **exactly like any other agent** — using `stance_label` as its current stance for stance-distance-based sampling purposes, with no changes to the Interaction Engine's code at all. This is the deliberate design choice this revision makes, and it has two direct consequences:

1. **Zero changes to `interaction_engine.py` or `recommendation_strategies.py`.** The Interaction Engine has no notion of "meme" versus "generated text" — it only ever reads a stance value and computes sampling weights from it, regardless of whether that stance came from an LLM call or a dataset label. The meme-injection feature is entirely a concern of the Orchestrator (deciding whether to call the Model Backend or the Meme Pool Manager for a given agent's turn) and the schema (tagging the resulting record), not of the recommendation logic.
2. **A meme's influence on the population is entirely mediated through being sampled as a neighbor by other agents in later turns** — exactly the same mechanism by which any generated post influences the population. This is what makes "does meme presence accelerate convergence" a meaningful multi-turn dynamics question rather than a single-exposure effect: a meme's stance can pull nearby agents' stances over several subsequent turns via the same alpha-sampling reinforcement loop that drives ordinary polarization in RQ1.

`simulation_orchestrator.py` receives a `RecommendationStrategy` instance exactly as before; the meme-injection decision is a separate step inserted immediately before content generation for each scheduled speaker, not a change to neighbor selection itself.

---

## 7. Logging and Export Design

**Format:** unchanged in structure from the original design. One run = one directory (`runs/{run_id}/`), containing `run_config.json`, `interactions.jsonl`, `agents_final.jsonl`, `checkpoints/`.

**Why JSONL over a database:** unchanged rationale — single-process, append-only, sequential writes; no transactional or concurrent-writer requirements in this scope.

**Mapping to downstream analysis, updated for this scope:**
- `stance_regression.py` reads `interactions.jsonl` into a DataFrame with columns `stance_before, stance_after, neighbor_avg_stance` — unchanged from the original design; meme-posted interactions contribute a `stance_after` value (the meme's label) exactly like any generated interaction, so no special-casing is needed for the regression to include meme turns in the neighbor-average computation.
- `bimodality_analysis.py` reads the per-turn stance distribution across all M agents (from `interactions.jsonl`, grouping by `turn`) and computes a bimodality metric per turn, letting you plot/compare convergence speed across `meme_injection.enabled ∈ {true, false}` conditions directly — this is RQ3's primary analysis path.
- `coherence_scorer.py`, `codemix_ratio_tracker.py`, `sentiment_proxy.py` read `reason_text` + `language_condition` from the same DataFrame, **filtering to `content_type == "generated_text"` first** — these diagnostics are about the quality of an agent's own generated reasoning and are not meaningful applied to a meme's `caption_text`, so the filter step is a required part of the RQ2 analysis pipeline, not optional.

---

## 8. Explicit Non-Goals

- **No human-facing web UI.**
- **No real-time streaming.**
- **No relational database.**
- **No live/dynamic social-graph mutation (follow/unfollow).** Unchanged.
- **No support for arbitrary numbers of languages beyond what's configured.** Unchanged.
- **No cross-model sweep or capability-tier comparison.** (New non-goal, explicit for this scope.) `model_backend_id` is single-valued per run; nothing in this design needs to run the same simulation across multiple model families and compare outcomes, since RQ1–RQ3 all concern language and content-type effects within a fixed model, not between-model differences.
- **No single-exposure / one-shot stimulus-response mode.** (New non-goal, explicit for this scope.) All meme exposure happens through in-loop injection into the ongoing multi-turn discussion; there is no standalone "show one agent one meme and measure the shift" code path.
- **No new meme annotation or human-agreement validation pipeline.** (New non-goal, explicit for this scope.) `MemeContent.stance_label` and `offensiveness_label` are taken as given from the source dataset's existing metadata; this design does not include a module for validating those labels against fresh human annotation, since RQ3 as scoped does not depend on the labels' accuracy in the same way the old RQ5 (model-vs-human meme-intensity agreement) did — RQ3 only needs the labels to be internally consistent enough to drive sampling, not to be validated ground truth.

---

---

# PART 2: ADVERSARIAL SELF-CRITIQUE (re-derived for this scope)

## Critique Pass 1 — RQ Coverage Gaps

**RQ1 — NO NEW GAP** (carrying forward Fix A from the prior revision, which already resolved the original memory-truncation gap): the fixed 5-turn window policy applies here unchanged, and remains sufficient for RQ1 as scoped.

**RQ2 — NO NEW GAP** (Fix B, naming concrete tools for coherence/sentiment scoring, still applies unchanged): the only addition worth noting is the filtering requirement described in §7 — `coherence_scorer.py` and `sentiment_proxy.py` must operate only on `content_type == "generated_text"` rows, since a meme's `caption_text` (if used as a fallback `reason_text`) is not the agent's own reasoning and scoring it for coherence would be measuring the wrong thing. This filtering step is now stated explicitly in §7, closing what would otherwise be a new, scope-specific gap.

**RQ3 — GAP.** Does the meme-injection mechanism actually produce data that lets you compare "meme-present" vs. "meme-absent" convergence speed cleanly? As designed in §6, yes for the *core* comparison — but there's an unaddressed granularity question: `injection_rate` controls what fraction of *agent-turns* become meme posts, but nothing in the design specifies whether this rate is applied **per-run** (a run either has memes injected at some rate throughout, or doesn't, and you compare two whole distributions of runs) or whether a **single run** could contain a controlled sub-comparison (e.g., half the agents never receive meme-sourced neighbors, half do, within the same run). The current design's schema and Orchestrator description only support the first interpretation (rate applies uniformly across the whole run's population), which is the simpler and more defensible design, but this needs to be stated as the resolved decision rather than left ambiguous, since a reader could otherwise reasonably assume the second interpretation from the phrase "what fraction of turns/agents." **Resolved in Part 3.**

**RQ3 — SECOND GAP.** Does injecting a meme mid-turn interact with the memory/context-window truncation policy (Fix A, fixed 5-turn count) in any way that wasn't true for pure-text turns? Fix A's fixed-turn-count window (as opposed to a token-budget window) was specifically designed to avoid a *language*-driven truncation asymmetry. But a meme turn and a generated-text turn are not the same size in the window either — a meme "turn" contributes an image (for vision-capable receivers) plus a short caption, while a generated-text turn contributes a longer free-text reason. Because Fix A counts *turns*, not tokens, this asymmetry doesn't cause a truncation-count problem (5 turns is still 5 turns, regardless of content type) — but it does mean the *effective information content* an agent retains from a meme-turn neighbor is different in kind from a text-turn neighbor, which is worth stating explicitly as an acknowledged property of the design rather than an oversight. **Resolved in Part 3** with a logging addition, not a mechanism change (the turn-count window itself needs no change).

**Does RQ3 need any new module beyond what RQ1/RQ2 already require, besides the meme injection point itself?** Checking against §4: the only genuinely new modules are `meme_pool_manager.py` and the reframed `vision_fallback.py` (which existed before but is now triggered differently). `bimodality_analysis.py` was already listed for RQ1 as a secondary metric in the prior revision and becomes RQ3's *primary* metric here — no new module required for that, just a different primary-vs-secondary role. Everything else (`agent_manager`, `interaction_engine`, `model_backend`, `stance_parser`, `logging_writer`, `checkpoint_manager`, `cost_tracker`, `rate_limiter`, `seed_manager`) is shared, unmodified infrastructure. **Confirmed: RQ3 adds exactly two new modules to the RQ1/RQ2 baseline.**

---

## Critique Pass 2 — Engineering / Scalability / Cost Issues

**Recomputed call volume for this three-RQ scope only.**

RQ1/RQ2 campaign (shared runs, single model backend, dropping the old multi-topic assumption per Fix E's reasoning — 1 topic as the default): M=100 × K=10 = 1,000 calls per trial-condition. With 2 language conditions × 5 trials × 1 topic = **10,000 LLM calls.**

RQ3 campaign: same 1,000 calls per trial-condition (meme-posting agents' turns replace an LLM call with a free lookup against `MemeContent`, so meme injection *reduces* the call count slightly rather than adding to it — a meme turn costs zero LLM calls for the posting agent, though it still costs normal calls for every *other* agent whose turn involves seeing that meme as a neighbor). With 2 meme-injection conditions (`enabled: true/false`) × 5 trials × 1 topic = **~10,000 LLM calls** (slightly under, proportional to `injection_rate`, since some fraction of turns are free).

**If RQ1 and RQ3 are run as a combined sweep** (language × meme-injection, 2×2 = 4 cells) rather than two separate campaigns, per §2's noted option: 4 cells × 5 trials × 1,000 calls ≈ **20,000 LLM calls total** — roughly the same as running them separately (10,000 + 10,000), since the combined design doesn't add extra cells beyond what running both campaigns independently would need; it just organizes the same total call volume into a cleaner factorial structure. Recommending the combined-sweep framing for this reason: same cost, better experimental design (lets you check for a language × meme interaction as a side benefit, even though no RQ in this scope currently asks for one).

**Total for this scope: ~20,000 LLM calls**, no vision-model cost multiplier beyond whatever fraction of turns are meme-injected with a vision-capable backend (moderate, not the earlier design's separate large vision-call campaign, since meme exposure here happens as a normal part of the discussion loop rather than as a dedicated single-exposure campaign with its own trial grid). At the same conservative $0.01–0.03/call blended estimate used in the original document, this is roughly a **$200–$600 budget** — a substantial reduction from the original 5-RQ design's $1,200–$3,600+ estimate, consistent with dropping the cross-model sweep (previously the largest cost driver) entirely. Still recommend a small pilot (1 trial per cell) before committing to the full 5-trial run, per the original document's guidance, which continues to apply.

**Does vision-model cost tracking (Fix H) still apply unchanged?** Yes. `cost_tracker.py`'s `(provider, model_id, modality)` keying is unchanged — a meme turn dispatched to a vision-capable receiver still costs differently than a text-only turn, and this needs the same modality-aware lookup as before. The only change is *when* vision costs are incurred: previously, a dedicated single-exposure campaign had its own separate, predictable vision-call count; now, vision calls occur unpredictably throughout the multi-turn loop, proportional to `injection_rate` and how often meme-posting agents get sampled as neighbors by vision-capable-backend agents in subsequent turns. This makes vision-cost *forecasting* slightly harder (you can't compute it purely from `injection_rate` without also knowing typical neighbor-sampling frequency) — worth noting as a minor added uncertainty in budget planning, resolved by monitoring `total_cost_usd`'s modality breakdown live during a pilot run rather than trying to predict it exactly in advance.

**Ordering/race condition (Fix F), checkpointing (Fix G), and stance-parse-failure handling (Fix I)** — all unchanged and still fully applicable; none of these were specific to the removed RQ4/RQ5/single-exposure scope, so nothing here needs re-derivation. One addition specific to this scope: the meme-injection decision (does this agent post a meme or generate text this turn?) must be made using the same seeded RNG as neighbor sampling (per `seed_manager.py`), and must be decided *before* dispatch within Fix F's frozen-snapshot sequence — i.e., the Orchestrator's per-turn sequence is now: freeze snapshot → resolve neighbor assignments → resolve meme-injection assignments → dispatch (LLM calls for generated-text agents; free lookups for meme agents) → collect → advance snapshot. This is a small addition to Fix F's sequence, not a change to its synchronous-turn guarantee.

---

## Critique Pass 3 — Architecture-Induced Validity Confounds

**Memory/language confound (Fix A) — re-checked, still holds.** The fixed 5-turn-count window continues to prevent the original token-budget-based truncation asymmetry described in the prior revision. No change needed for this scope.

**NEW — meme-injection scheduling vs. language condition interaction.** If RQ1 and RQ3 are run as a combined sweep (per Critique Pass 2's recommendation), does the point in the discussion where a meme is injected (`injection_schedule: "fixed_turn"`, e.g. always at turn 5) get treated identically across the English and Hinglish language conditions it might co-occur with? As designed, yes — `injection_schedule` is a run-level config value applied uniformly regardless of `language_condition`, so a "fixed_turn: 5" run injects at turn 5 whether the discussion is in English or Hinglish. However, this surfaces a subtler risk worth flagging: **if `injection_schedule: "random"` is used instead**, the seeded RNG (per `seed_manager.py`) determines *which* agents and turns get meme-injected, and if the same `seed` value is reused across the English and Hinglish conditions of a combined sweep (a natural thing to do, to hold "everything except language" constant), the *identity* of which agents receive memes will be identical across both language conditions — which is actually the *correct*, confound-avoiding behavior, and should be stated as the deliberate default: **the same seed should be reused across language conditions specifically so that meme-injection patterns are held constant**, not varied, when isolating the language effect. This is the opposite of the seed-reuse caution that applied to LLM sampling itself (§3.5's note that seeds don't control LLM output determinism) — here, seed reuse for the *non-LLM* stochastic decisions (which agents get memes) is exactly what you want, and the design should say so explicitly rather than leaving it to be inferred.

**NEW — does a meme-bearing "turn" cost a different amount of context-window space than a generated-text turn in a way that could bias truncation?** As resolved in Critique Pass 1's second RQ3 gap: because Fix A's window is turn-count-based, not token-budget-based, a meme turn and a text turn each occupy exactly one "slot" in the 5-turn memory window regardless of their actual token cost — so there is no truncation-count bias introduced by mixing meme and text turns in memory. What *is* worth logging (not fixing, since it isn't a bug, just a property to be aware of during analysis) is that the *token cost* of those 5 slots will vary run-to-run depending on how many were memes vs. generated text, which is already captured by the existing `prompt_token_count` field on `Interaction` — no schema change needed, just an explicit note that this field should be checked as a covariate when interpreting RQ3 results, analogous to Fix K's original token-overhead robustness check.

**Stance-parsing robustness (Fix I) — does it interact differently with meme injection?** No new issue: meme-posting agents never produce a stance via `stance_parser.py` at all (their stance is the dataset label, set directly), so there is no parse-failure risk for meme turns specifically. Fix I's retry/exclusion policy continues to apply only to generated-text turns, unchanged.

---

---

# PART 3: REVISED DESIGN AND VALIDATION TABLE

## Fixes Applied (mapped explicitly to flagged issues, this revision)

| Fix | Addresses |
|---|---|
| **Fix A** (carried forward, unchanged) — fixed 5-turn-count memory window, not token-budget-based. | Prior revision's Pass 3 memory/language confound; re-confirmed still holding in this scope's Critique Pass 3. |
| **Fix B** (carried forward, unchanged) — named concrete tools for RQ2's coherence/sentiment diagnostics. | Prior revision's Pass 1 RQ2 gap; unchanged in this scope. |
| **Fix E** (carried forward, reduced scope) — single-topic default, single-model-backend default (no longer "4 models," since cross-model sweeps are entirely removed from this scope). | Cost blowup, recomputed in this revision's Critique Pass 2. |
| **Fix F** (carried forward, extended) — synchronous-turn semantics; extended in this revision to explicitly include meme-injection-assignment as part of the frozen-snapshot per-turn sequence. | Prior revision's Pass 2 ordering/race condition; extended per this scope's Critique Pass 2. |
| **Fix G** (carried forward, unchanged) — per-turn checkpointing with resume. | Prior revision's Pass 2 no-checkpoint gap; unchanged. |
| **Fix H** (carried forward, unchanged mechanism, new forecasting note) — modality-aware cost tracking. | Prior revision's Pass 2 vision/text cost tracking gap; this revision adds a note that vision-cost *forecasting* (not tracking) is harder under in-loop injection, resolved by pilot-run monitoring rather than a design change. |
| **Fix I** (carried forward, unchanged) — standardized stance-parse-failure handling, retry-then-exclude. | Prior revision's Pass 3 stance-parsing confound; confirmed unaffected by meme injection in this scope's Critique Pass 3. |
| **Fix L — Meme-injection rate applies uniformly per-run, not as a within-run sub-comparison.** `injection_rate` and `injection_schedule` are run-level config values applied consistently across the whole population within a given run; comparing meme-present vs. meme-absent is done *across* runs (`meme_injection.enabled: true` vs. `false`), not *within* a single run's population. This is stated explicitly as the resolved design, closing this scope's first RQ3 gap. | This revision's Critique Pass 1, RQ3 first gap (injection-rate granularity ambiguity). |
| **Fix M — Seed reuse across language conditions is the deliberate default when combining RQ1/RQ3 into a single sweep.** When `injection_schedule: "random"` is used and RQ1's language conditions are combined with RQ3's meme-injection conditions in one sweep, the same `seed` value should be reused across the English and Hinglish arms of a given trial, so that which agents/turns receive memes is held constant while only `language_condition` varies. This is stated explicitly, distinguishing it from the unrelated caution (unchanged from the original document) that seeds do not and should not be relied on to control LLM sampling determinism. | This revision's Critique Pass 3, new meme-injection/language-interaction confound. |
| **Fix N — Post-hoc filtering requirement for RQ2's diagnostics.** `coherence_scorer.py` and `sentiment_proxy.py` must filter to `Interaction.content_type == "generated_text"` before scoring, excluding meme-posted turns (whose `reason_text` is either empty or a dataset caption, not the agent's own reasoning) from coherence/sentiment analysis. Stated explicitly in §7 and Critique Pass 1, rather than left as an implicit assumption an analyst might miss. | This revision's Critique Pass 1, RQ2 (new scope-specific clarification). |

## Removed Components (from the prior 5-RQ revision), with justification

| Removed | Why it's no longer needed |
|---|---|
| Single-exposure simulation mode, `ExposureReceiver`, `ExposureResult`, mode-branching in the Orchestrator | The old matched meme-vs-text-paraphrase RQ this mode existed for is gone; the current RQ3 runs entirely inside the multi-turn loop. |
| `stimulus_manager.py`, `exposure_receiver_manager.py` | Replaced by `meme_pool_manager.py`, which is simpler — no matched-pair construction, no receiver/persona-pool assignment logic, since there's no separate receiver population anymore. |
| `MemeStimulus` schema (with `pair_id`, `modality`, `matched_text`, `intensity_label`) | Replaced by the smaller `MemeContent` schema (§3.4) — no pairing concept needed once memes are simply pool content rather than one half of an engineered comparison. |
| Fix C (`DiscussionAgent`/`ExposureReceiver` split) | Only needed to separate two populations that no longer both exist; every agent in this scope is a full `Agent`. |
| Fix D (capability-tier mapping) | Only needed for the old cross-model RQ5; no cross-model comparison remains in scope. |
| Fix J (shared prompt-template enforcement between meme and matched-text arms), Fix K (image-token-overhead pre-check for the matched-pair comparison) | Both existed specifically to protect the validity of the old matched meme-vs-text-paraphrase *pairwise* comparison. That comparison no longer exists; RQ3's comparison is now meme-present vs. meme-absent *populations* across separate runs, which doesn't have a "shared template between two arms" concept in the same way — there is no matched-text arm to keep symmetric with the meme arm. (The general principle — don't let an implementation detail introduce an unintended asymmetry — still applies in spirit and is why Fix M and the token-cost note under Critique Pass 3 exist, but the specific mechanism these two fixes addressed is gone.) |
| `capability_tier_mapper.py`, `graph_export.py`, `network_metrics.py`, cross-model sweep configs, "6 model variants"/"4 model variants" framing | See §4's Removed Modules explanation and §3.5's schema (single `model_backend_id`, not a list). |
| RQ5-specific human-annotator meme-intensity agreement checks | This study no longer generates or validates its own meme ground truth; `MemeContent` labels are taken as-is from the pre-existing dataset, per the new non-goal stated in §8. |

---

## Final Validation Table

| RQ | Sandbox components depended on | Simulation mode/config | Data logged | Metric(s) computed | Support judgment |
|---|---|---|---|---|---|
| **RQ1** | `agent_manager`, `interaction_engine` (`AlphaSampling`), `model_backend` (text-only), `prompt_builder` (Fix A: fixed 5-turn window), `logging_writer`, `stance_regression`, `embedding_cluster`, `codemix_ratio_tracker`, `bimodality_analysis` (secondary) | Multi-turn, M=100/N=5/K=10, `language_condition ∈ {english, hinglish}`, `meme_injection.enabled: false`, alpha fixed, 5 trials, 1 topic (Fix E) | `interactions.jsonl`: stance_before/after, reason_text, neighbor_agent_ids, prompt_token_count, content_type (always "generated_text" for pure RQ1 runs) | Ohagi-style regression per language condition; code-mixing ratio as covariate; bimodality as secondary polarization indicator | **YES** |
| **RQ2** | Same runs as RQ1 (no new simulation) + `coherence_scorer` (Fix B, filtered per Fix N), `sentiment_proxy` (Fix B, filtered per Fix N), `codemix_ratio_tracker` | Re-analysis of RQ1's `interactions.jsonl`, filtered to `content_type == "generated_text"` (Fix N) | Same `interactions.jsonl`, plus derived per-turn coherence/sentiment/code-mix scores | Correlation between coherence/sentiment instability and polarization speed (via `bimodality_analysis`'s convergence-speed output) | **YES**, contingent on Fix B's tool choices being empirically validated against a hand-labeled Hinglish sample before full-scale use — a required pre-analysis step, not a sandbox gap |
| **RQ3** | `agent_manager`, `interaction_engine` (unchanged, per §6), `meme_pool_manager` (new), `vision_fallback` (reframed), `model_backend`, `cost_tracker` (Fix H), `bimodality_analysis` (primary metric), `logging_writer` | Multi-turn, M=100/N=5/K=10, `meme_injection.enabled ∈ {true, false}` (Fix L: applied uniformly per-run), `language_condition` held fixed at a baseline (or combined with RQ1 per Fix M's seed-reuse guidance), alpha/topic/M/N/K fixed, 5 trials, 1 topic | `interactions.jsonl`: same fields plus content_type, meme_id, used_vision_fallback | Convergence speed (turns-to-bimodal) and final bimodality classification, compared across meme-present vs. meme-absent runs; `prompt_token_count` checked as a covariate per Critique Pass 3's token-cost note | **YES** |

**Overall: 3/3 YES**, with two explicitly load-bearing pre-analysis or methodological requirements that must be completed and reported, not silently assumed: RQ2's diagnostic-tool validation (Fix B) and RQ2's content-type filtering (Fix N). RQ3 carries no outstanding contingency beyond standard pilot-run cost verification (Critique Pass 2), since the meme-injection design's validity concerns (Fix L, Fix M) were resolved as explicit, stated defaults rather than open questions.
