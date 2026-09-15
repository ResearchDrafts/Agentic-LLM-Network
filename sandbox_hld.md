# LLM Agent Echo Chamber Sandbox — High-Level Design

---

# PART 1: INITIAL HIGH-LEVEL DESIGN

## 1. System Architecture

Seven components. Two simulation *modes* (multi-turn discussion for RQ1/RQ4/RQ5; single-exposure for RQ3) share the bottom four layers and diverge only at the Orchestrator level.

```
                        ┌─────────────────────────┐
                        │   Config / Run Loader    │
                        │  (topic, alpha, M,N,K,   │
                        │   model list, lang, seed)│
                        └────────────┬─────────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    │      Simulation Orchestrator      │
                    │  ┌──────────────┐ ┌─────────────┐ │
                    │  │ Multi-Turn   │ │  Single-     │ │
                    │  │ Mode         │ │  Exposure    │ │
                    │  │ (RQ1,RQ4,RQ5)│ │  Mode (RQ3)  │ │
                    │  └──────┬───────┘ └──────┬───────┘ │
                    └─────────┼──────────────────┼────────┘
                              │                  │
          ┌───────────────────┴──────┐   ┌───────┴────────────┐
          │      Agent Manager        │   │  Stimulus Manager   │
          │  (creates/holds Agent      │   │ (loads meme/text     │
          │   objects, persona,        │   │  matched pairs,      │
          │   stance, memory)          │   │  assigns receiver     │
          └───────────┬────────────────┘   │  agent + starting     │
                       │                    │  stance)               │
                       │                    └───────┬────────────────┘
          ┌────────────┴─────────────┐              │
          │ Interaction/Recommendation│              │
          │ Engine (alpha-sampling,   │              │
          │ ideology-proximity score, │              │
          │ swappable strategy)       │              │
          └────────────┬──────────────┘              │
                       │                              │
          ┌────────────┴──────────────────────────────┴────────┐
          │           Model Backend Abstraction Layer            │
          │  (LiteLLM-based; uniform call contract; text vs      │
          │   vision-capable dispatch; retry/rate-limit/cost)    │
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
                       │  - NetworkX/igraph graph   │
                       │    reconstruction          │
                       │  - RWC, BC, etc. (12       │
                       │    Impiccichè metrics)     │
                       │  - Ohagi-style regression   │
                       │  - coherence/code-mix/      │
                       │    sentiment diagnostics    │
                       └────────────────────────────┘
```

**Data flow, multi-turn mode (RQ1/RQ4/RQ5):** Config Loader instantiates a run → Agent Manager creates M agents with persona/stance/language-condition/model-backend assignment → for each of K turns: Interaction Engine samples N neighbors per agent per Ohagi's alpha function → Agent Manager builds each agent's prompt (its own memory + sampled neighbor posts) → Model Backend Layer dispatches the call to the correct provider → response (stance + reason) is parsed and written to the agent's state AND to the Logging Layer as an Interaction record → next turn repeats, now using updated state.

**Data flow, single-exposure mode (RQ3):** Config Loader instantiates a run → Stimulus Manager loads a meme/text matched pair and assigns a starting stance to a synthetic "receiver" agent (no multi-agent population, no Interaction Engine call at all) → Model Backend Layer dispatches one call (vision-capable for meme, text for matched paraphrase) → pre/post stance logged → done. This mode does not touch the Interaction/Recommendation Engine and does not run for K turns — it is a single request/response pair per exposure.

**Why this split exists (called out here, revisited in Critique Pass 1):** RQ3 has a categorically different population, no discussion loop, no neighbor sampling, and no requirement for the Interaction Engine at all. Forcing it through the multi-turn Orchestrator path (e.g., "just set K=1, N=1") would be simpler to describe but wrong to build — it would drag in memory/context-history logic, alpha-sampling, and multi-agent bookkeeping that RQ3 doesn't need and that create surface area for RQ3-specific bugs. Two Orchestrator modes sharing lower layers is the deliberate choice.

---

## 2. Agent Population Sizing, Per RQ

| RQ | Mode | M | N | K | Trials/cell | Justification |
|---|---|---|---|---|---|---|
| RQ1 | Multi-turn | 100 | 5 | 10 | 5 | Matches Ohagi's baseline exactly (M=100, N=5, K=10) since RQ1 is a direct extension of his design with language as the new variable. No reason to deviate — changing M/N/K here would confound "language effect" with "population-size effect" relative to the paper you're extending. |
| RQ2 | Multi-turn (same runs as RQ1) | 100 | 5 | 10 | 5 | RQ2 is a post-hoc diagnostic computed on RQ1's *same transcripts* (coherence, code-mixing ratio, sentiment-proxy scoring per turn). No separate simulation runs needed — this is an Analysis Module addition, not a new sandbox mode. |
| RQ3 | Single-exposure | N/A (no population — one receiver agent per exposure) | N/A | 1 (single exposure, not K turns) | 15 exposures per stimulus pair (5 starting stances × 3 reps) × both arms (meme, matched-text) | Fundamentally not a "group of M agents discussing" design — it's a controlled stimulus-response measurement, closer to a psych experiment than an ABM. M/N/K don't apply. Sizing is dictated by covering the starting-stance range (5 points spanning the stance scale) with enough repetition (3) to average out sampling noise from LLM stochasticity, per stimulus pair. |
| RQ4 | Multi-turn | 100 | 5 | 10 | 5 (up from Ohagi's 3, per your locked-in decision) | Same core M/N/K as Ohagi, since the RQ is explicitly about holding echo-chamber parameters identical across model backends — changing M/N/K per model would reintroduce exactly the confound RQ4 is designed to avoid. Trial count raised to 5 to absorb expected added between-run variance from comparing model families (not just comparing same-family checkpoints). |
| RQ5 | Multi-turn (same runs as RQ4) | 100 | 5 | 10 | 5 | RQ5 reuses RQ4's simulation runs; it's a re-analysis of the same data, grouping models by capability tier instead of by individual model identity. No separate simulation mode. |

**Net simulation modes needed: 2** (multi-turn, single-exposure). **Net distinct simulation *campaigns*: 3** (RQ1/RQ2 language campaign; RQ4/RQ5 model-comparison campaign; RQ3 exposure campaign) — RQ2 and RQ5 add no new runs, only new analysis code.

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
    model_backend_id: str            # e.g. "gpt-4o", "claude-sonnet-4-6", "gemini-2.0-flash",
                                      #      "llama-3.1-8b", "llama-3.1-70b"
    initial_stance: float            # numeric stance on Ohagi's finite-category scale (e.g. 1-7)
    stance_history: list[StanceRecord]   # append-only, one entry per turn
    memory_window: list[str]         # rolling buffer of interaction_ids this agent has seen
                                      # (raw text NOT duplicated here — pointer into Interaction log,
                                      #  see 5.3 confound note on memory/context handling)
    created_at_turn: int             # always 0 for RQ1/4/5; N/A for RQ3 (single receiver, not logged as Agent)
```

### 3.2 StanceRecord (embedded in Agent, also independently logged — see 6)

```python
@dataclass
class StanceRecord:
    turn: int
    stance_value: float
    reason_text: str                 # free-text justification, the raw model output
    interaction_id: str              # links to the Interaction record this stance update came from
```

### 3.3 Interaction (the core record for network analysis — one row per agent-turn)

```python
@dataclass
class Interaction:
    interaction_id: str              # uuid4
    run_id: str
    turn: int
    speaker_agent_id: str
    neighbor_agent_ids: list[str]    # the N agents sampled this turn whose posts speaker saw
                                      # THIS is the edge list for the graph: speaker <- each neighbor
    stance_before: float
    stance_after: float
    reason_text: str
    prompt_token_count: int          # for cost tracking + Critique Pass 3 confound check
    completion_token_count: int
    model_backend_id: str
    latency_ms: int
    api_call_status: str             # "success" | "retried_success" | "failed_logged_null"
    timestamp_utc: str
```

Note: `neighbor_agent_ids` is what makes this a graph edge list. A directed edge `neighbor -> speaker` for each entry, weighted by turn, is exactly what the Post-hoc Analysis Module needs to reconstruct a NetworkX `DiGraph` for RWC/BC/etc. — no separate "graph" data structure needs to be maintained live during simulation; it is derivable entirely from the Interaction log.

### 3.4 MemeStimulus (RQ3-specific)

```python
@dataclass
class MemeStimulus:
    stimulus_id: str
    pair_id: str                     # links meme condition to its matched text condition
    modality: str                    # "meme" | "matched_text"
    image_path: str | None           # populated only if modality == "meme"
    caption_text: str | None         # meme's embedded/caption text, if modality == "meme"
    matched_text: str | None         # populated only if modality == "matched_text";
                                      # the paraphrase expressing the same stance+reasoning as the meme
    stance_label: float              # ground-truth stance/intensity of the stimulus itself
                                      # (from Indian Political Memes dataset metadata, or annotated)
    intensity_label: float | None    # if the dataset provides an intensity/strength score separately
    topic: str
    source_dataset: str              # provenance, e.g. "IPM-2023"
```

### 3.5 ExposureResult (RQ3-specific, analogous to Interaction but for single-exposure mode)

```python
@dataclass
class ExposureResult:
    exposure_id: str
    run_id: str
    stimulus_id: str                 # FK to MemeStimulus
    receiver_starting_stance: float  # one of the 5 preset starting-stance points
    repetition_index: int            # 0, 1, 2 (of 3 reps)
    receiver_model_backend_id: str
    stance_after: float
    reason_text: str
    prompt_token_count: int
    completion_token_count: int
    api_call_status: str
    timestamp_utc: str
```

### 3.6 ExperimentRun (top-level config/metadata, one per simulation execution)

```python
@dataclass
class ExperimentRun:
    run_id: str                      # uuid4, also the output directory name
    rq_target: str                   # "RQ1" | "RQ3" | "RQ4_RQ5"  (which campaign this run belongs to)
    mode: str                        # "multi_turn" | "single_exposure"
    topic: str
    alpha: float | None              # echo chamber strength param; None for RQ3
    M: int | None
    N: int | None
    K: int | None
    trial_number: int
    language_condition: str | None   # for RQ1 campaign runs
    model_backends: list[str]        # list because RQ4/5 may run several backends in one config sweep
    seed: int                        # for reproducibility of any stochastic sampling (neighbor selection,
                                      # persona assignment) — NOT for LLM sampling itself, since providers
                                      # don't guarantee deterministic output even at temperature=0
    temperature: float
    git_commit_hash: str             # code version at run time
    started_at_utc: str
    completed_at_utc: str | None
    status: str                      # "running" | "completed" | "failed_partial" | "failed_total"
    total_cost_usd: float            # running total, updated as calls complete
```

---

## 4. Modules to Build

| Module | Purpose (one line) | Inputs → Outputs | RQ dependency |
|---|---|---|---|
| `config_loader.py` | Parse a run config (YAML/JSON) into an `ExperimentRun` object, validate required fields per mode | config file → `ExperimentRun` | All |
| `agent_manager.py` | Create, hold, and mutate `Agent` objects for a multi-turn run | `ExperimentRun`, persona pool → `list[Agent]` | RQ1, RQ4, RQ5 |
| `stimulus_manager.py` | Load meme/text stimulus pairs, assign starting stances, generate exposure schedule | `MemeStimulus` dataset, config → `list[ExposureResult]` (scaffolded, pre-stance-fill) | RQ3 |
| `interaction_engine.py` | Given current agent stances/graph state, select N neighbors per agent per turn using a swappable strategy (alpha-sampling by default) | `list[Agent]`, `alpha`, turn number → `dict[agent_id, list[neighbor_id]]` | RQ1, RQ4, RQ5 |
| `recommendation_strategies.py` | Concrete strategy implementations: `AlphaSampling`, `IdeologyProximityScoring` (Donkers & Ziegler-style), stubs for future strategies | strategy-specific | RQ1, RQ4, RQ5 (strategy choice may itself become a manipulated variable in future extensions, not currently one of your 5 RQs but designed for it) |
| `prompt_builder.py` | Construct the actual text/multimodal prompt sent to a model, given an agent's persona, memory, sampled neighbor posts, and language condition | `Agent`, `list[Interaction]` (neighbor posts) → formatted prompt (text) or (text, image) tuple | All |
| `model_backend.py` | Uniform interface over LiteLLM; dispatches text-only or vision calls; enforces per-backend rate limits; retries with backoff; converts to `Interaction`/`ExposureResult`-ready output | prompt (+ optional image) → raw response, token counts, latency | All |
| `vision_fallback.py` | For text-only backends receiving a meme stimulus (RQ3), substitutes `caption_text` alone in place of image+caption, tagged so this substitution is logged distinctly | `MemeStimulus`, `model_backend_id` → adjusted prompt path | RQ3 |
| `stance_parser.py` | Extract structured stance value + reason text from raw model output (handles parsing failures, malformed output) | raw completion text → `(stance_value, reason_text)` or parse-failure flag | All |
| `simulation_orchestrator.py` | Drives the K-turn loop (multi-turn mode) or the single-exposure loop (RQ3 mode); the only module that calls Agent Manager / Interaction Engine / Stimulus Manager / Model Backend in sequence | `ExperimentRun` → completed run, all records written to Logging Layer | All |
| `logging_writer.py` | Append-only writer for all record types (Interaction, ExposureResult, Agent snapshots, ExperimentRun status) to disk | records → files on disk (format decided in §6) | All |
| `checkpoint_manager.py` | Periodically snapshot full run state (all agents, turn number) so a failed run can resume rather than restart | run state → checkpoint file; checkpoint file → resumed run state | All (engineering necessity flagged in Critique Pass 2) |
| `cost_tracker.py` | Track $ cost per API call using per-provider/per-model pricing table, separate running totals for text vs vision calls, alert on threshold | token counts + model_backend_id → running `total_cost_usd`, per-model breakdown | All (flagged as "boring but essential" per your request) |
| `rate_limiter.py` | Per-provider token-bucket rate limiting so concurrent agent calls don't 429 | provider name, call → throttled dispatch | All |
| `seed_manager.py` | Central RNG seeding for all *non-LLM* stochastic choices (neighbor sampling, persona assignment, stimulus ordering) | seed int → seeded `random.Random` instances passed to Interaction Engine, Stimulus Manager | All |
| `graph_export.py` | Post-hoc: reconstruct a NetworkX `DiGraph` from a run's Interaction log | `run_id` → `.graphml` file(s), one per turn or cumulative | RQ1, RQ4, RQ5 |
| `network_metrics.py` | Compute the 12 Impiccichè & Viviani metrics (RWC, BC, Dipole Moment, etc.) on an exported graph | `.graphml` → metrics table (per run, per turn if applicable) | RQ1, RQ4, RQ5 |
| `stance_regression.py` | Ohagi-style linear regression: stance-before vs neighbor-average vs stance-after | Interaction log → regression coefficients, per run/condition | RQ1, RQ4, RQ5 |
| `coherence_scorer.py` | Post-hoc NLP diagnostic: coherence/consistency scoring of reason_text, separately calibrated or validated for Hinglish vs English | reason_text (+ language_condition) → coherence score | RQ2 |
| `codemix_ratio_tracker.py` | Post-hoc: compute code-mixing ratio (e.g., CMI — code-mixing index) per reason_text | reason_text → ratio | RQ1, RQ2 |
| `sentiment_proxy.py` | Post-hoc: sentiment/emotion scoring that works on Hinglish (not VADER) — e.g., a multilingual transformer-based scorer | reason_text → sentiment/emotion score | RQ2 |
| `bimodality_analysis.py` | Post-hoc: fit/test for bimodal vs unimodal final stance distribution per run (e.g., bimodality coefficient, dip test) | final stance distribution → bimodality metric + classification | RQ4, RQ5 |
| `capability_tier_mapper.py` | Maps `model_backend_id` → a capability tier label/score (your chosen proxy — benchmark score or generation ordinal) | `model_backend_id` → tier value | RQ5 |
| `embedding_cluster.py` | Sentence-BERT/SimCSE clustering of reason_text (Ohagi's method), multilingual-capable embedding model for Hinglish support | reason_text corpus → cluster assignments | RQ1, RQ2, RQ4, RQ5 |

---

## 5. Model Backend Abstraction

**Contract every backend must satisfy** (concretely, a Python protocol):

```python
class ModelBackend(Protocol):
    supports_vision: bool
    provider_name: str          # "openai" | "anthropic" | "google" | "local"
    model_id: str                # provider-specific model string

    def generate(
        self,
        prompt_text: str,
        image: bytes | None = None,   # None for text-only calls
        temperature: float = 0.7,
    ) -> BackendResponse:
        ...

@dataclass
class BackendResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    raw_provider_response: dict     # kept for debugging, not used downstream
```

**Implementation:** a single `LiteLLMBackend` class implements this contract for OpenAI, Anthropic, Google, and any LiteLLM-supported local endpoint (Ollama, vLLM, HF TGI) — this is the one seam described in the earlier conversation, now formalized as a contract with an explicit `supports_vision` flag rather than an implicit assumption.

**Vision vs text-only dispatch (RQ3's meme condition):**
- `prompt_builder.py` always constructs *two* representations for a meme stimulus: the multimodal prompt (image + caption + instruction) and the caption-only fallback prompt.
- `model_backend.py` checks `backend.supports_vision` before dispatch. If `True`, sends the multimodal prompt. If `False`, sends the caption-only fallback via `vision_fallback.py`, and — critically — tags the resulting `ExposureResult` with a `used_vision_fallback: bool` field (added to §3.5's schema) so this substitution is visible in the data and can be excluded or analyzed separately, not silently conflated with "true" meme-condition results from vision-capable models.
- This is the initial design's answer to a fallback question that has no clean default: **should text-only models be included in the meme condition at all, using caption-only as a proxy, or excluded from that condition entirely?** Initial default: include them, flagged, and let the analysis decide whether to filter them out — excluding them outright would silently reduce your available model-comparison set for RQ3 without you necessarily noticing during data collection.

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
    ) -> list[Agent]:   # returns exactly N agents
        ...

class AlphaSampling(RecommendationStrategy):
    """Ohagi's mechanism: probability of sampling an agent is a function of
    stance similarity to speaker, controlled by alpha (0 = random/open,
    1 = maximally homophilous/closed)."""
    def __init__(self, alpha: float, N: int): ...
    def select_neighbors(self, speaker, all_agents, turn, rng): ...

class IdeologyProximityScoring(RecommendationStrategy):
    """Donkers & Ziegler-style: score = w1*popularity + w2*ideological_proximity,
    top-N by score. Not required by your locked RQs but implemented as a second
    concrete strategy to prove the interface is genuinely swappable, and available
    if a reviewer asks you to test sensitivity to recommendation mechanism."""
    def __init__(self, w_popularity: float, w_proximity: float, N: int): ...
    def select_neighbors(self, speaker, all_agents, turn, rng): ...
```

`simulation_orchestrator.py` receives a `RecommendationStrategy` instance (chosen via config, e.g. `strategy: "alpha_sampling", alpha: 0.7`) and calls `.select_neighbors(...)` once per agent per turn. This is the "swappable, testable" requirement satisfied directly — a unit test can instantiate `AlphaSampling(alpha=1.0, N=5)` and assert it always returns same-stance neighbors, independent of the rest of the simulation.

---

## 7. Logging and Export Design

**Format:** One run = one directory (`runs/{run_id}/`), containing:
- `run_config.json` — the `ExperimentRun` record
- `interactions.jsonl` — append-only, one `Interaction` (or `ExposureResult`) JSON object per line, written incrementally as the run progresses (not buffered in memory until the end — this matters for Critique Pass 2's checkpoint discussion)
- `agents_final.jsonl` — final `Agent` state snapshot at run completion (stance history embedded)
- `checkpoints/` — periodic full-state snapshots (see Critique Pass 2)

**Why JSONL over a database:** every record is append-only and write-once; there's no need for transactional updates, joins at write time, or concurrent-writer conflict resolution beyond what a single orchestrator process handles sequentially. JSONL is trivially resumable (append and continue), trivially diffable/inspectable by hand, and loads directly into a pandas DataFrame with `pd.read_json(lines=True)` for the regression and diagnostic analyses. A SQLite file remains a reasonable alternative if concurrent multi-process runs are later needed (see Non-Goals) but is not necessary for the current single-process-per-run design.

**Mapping to downstream analysis:**
- `graph_export.py` reads `interactions.jsonl`, builds a `networkx.DiGraph()` by adding an edge `(neighbor_id, speaker_id, {turn: t, weight: 1})` for every `neighbor_agent_ids` entry in every `Interaction` record, and writes `.graphml` — directly consumable by any of the 12 Impiccichè & Viviani metric implementations, all of which operate on NetworkX/igraph-compatible graphs.
- `stance_regression.py` reads `interactions.jsonl` into a DataFrame with columns `stance_before, stance_after, neighbor_avg_stance` (the last computed by joining `neighbor_agent_ids` against that turn's stance snapshot) — directly the three columns Ohagi's regression needs.
- `coherence_scorer.py`, `codemix_ratio_tracker.py`, `sentiment_proxy.py` all read `reason_text` + `language_condition` columns from the same DataFrame — no separate export needed.

---

## 8. Explicit Non-Goals

- **No human-facing web UI.** No RQ involves live human participants; Donkers & Ziegler's UI layer is explicitly irrelevant here.
- **No real-time streaming.** All runs are batch/offline; nothing needs to render live.
- **No relational database.** Flat JSONL files are sufficient given single-process, append-only, sequential writes (see §7). Revisit only if you later parallelize multiple concurrent runs writing to shared state, which the current design does not require.
- **No live/dynamic social-graph mutation (follow/unfollow).** Unlike OASIS, agents in your design don't need to structurally alter a persistent follow-graph — the "graph" is emergent from *who-saw-whom-per-turn* (the Interaction log), not a maintained adjacency structure agents edit. This is a deliberate simplification consistent with Ohagi's design, which your RQ1/4/5 are directly extending.
- **No support for arbitrary numbers of languages beyond what's configured.** The sandbox supports a fixed enum (`english`, `hinglish`, `hindi`) rather than an open-ended language plugin system — extending to a new language later means adding an enum value and corresponding prompt templates, not a redesign.

---

---

# PART 2: ADVERSARIAL SELF-CRITIQUE

## Critique Pass 1 — RQ Coverage Gaps

**RQ1 — PARTIAL GAP.** The `Agent.memory_window` field (§3.1) is described as "a rolling buffer of interaction_ids" with no explicit size limit or truncation policy specified anywhere in the design. For a 10-turn simulation with N=5 neighbors sampled per turn, an agent could accumulate up to 50 prior interactions in its context by turn 10. Nothing in the design says *how* this gets truncated (sliding window? summarized? full history always included?) — and this omission is actually a Pass 3 confound risk too (flagged there), but as a Pass 1 gap: the design as written doesn't specify this decision at all, so it cannot be said to "support" RQ1 until it's made explicit, because the truncation policy directly determines how much Hinglish vs English text is actually in-context at turn 10, which affects the very quality-of-opinion-updating comparison RQ1/RQ2 are testing.

**RQ2 — GAP.** `coherence_scorer.py` and `sentiment_proxy.py` are listed as modules but the design never specifies *what tool or model* performs coherence scoring or sentiment/emotion proxying for Hinglish text, only that VADER won't work (correctly identified as a known limitation, but not resolved). Without naming a concrete approach (e.g., a multilingual sentiment model, an LLM-as-judge coherence rubric), this is a placeholder, not a design. This matters because RQ2's core measurement — "does instability correlate with polarization speed" — depends entirely on this being a validated, non-arbitrary metric, and picking the wrong one wrongly could produce an artifact that looks like a language effect but is actually a measurement-tool effect.

**RQ3 — GAP.** The design specifies a `receiver_starting_stance` as one of 5 preset points but never specifies **how the receiver agent's persona is held constant** across the meme and matched-text arms for the same starting stance. If persona is randomly regenerated per exposure rather than fixed per (starting_stance, repetition_index) pair and reused identically across the meme/text arms, you introduce a persona confound directly into the modality comparison RQ3 exists to measure cleanly. The current `stimulus_manager.py` module description doesn't address persona assignment at all.

**RQ3 — SECOND GAP.** Nothing in the design specifies whether the receiver agent in single-exposure mode has *any* memory/context beyond the single stimulus (i.e., is it a truly fresh agent with no discussion history, or does it inherit persona/history from the multi-turn pool?). If RQ3's receiver agents are meant to be freshly initialized each time (which the "single exposure" framing implies), the `Agent` dataclass's `memory_window` and `stance_history` fields are irrelevant baggage for this mode, but the schema doesn't distinguish a "full" `Agent` from a lightweight "Receiver" — this is exactly the "conflating a genuinely different code path with the main loop" risk you asked me to check for, and it's present here at the schema level, not just the orchestrator level.

**RQ4 — NO GAP FOUND.** The model-backend abstraction, recommendation-strategy swappability, and fixed M/N/K/alpha/topic across model conditions directly satisfy RQ4 as designed, assuming the Critique Pass 3 cross-backend-consistency issues (below) are fixed.

**RQ5 — GAP.** `capability_tier_mapper.py` is listed as a module but the design never states *what proxy* is used (a specific benchmark, e.g. MMLU score, or an ordinal "generation number"), nor whether this mapping is fixed at analysis time (defensible, arbitrary-but-declared) or somehow used to *select* which models get run (which would be circular). This needs to be pinned down as a design decision, not left as a module stub, because it affects reviewers' ability to evaluate whether RQ5's "capability tier" variable is a legitimate operationalization or an ad hoc post-hoc bucketing.

---

## Critique Pass 2 — Engineering / Scalability / Cost Issues

**Cost blowup is real and severe — the design as specified is not affordable on a student/small-lab budget.** Rough math, using your own locked-in parameters:

- RQ1 campaign: M=100 agents × K=10 turns = 1,000 LLM calls per trial-condition. With 2 language conditions × 3 topics (a reasonable topic count matching Ohagi's multi-topic design, though not explicitly locked in your document — flagged as an assumption) × 5 trials = **30,000 LLM calls** for RQ1/RQ2 alone.
- RQ4/RQ5 campaign: same 1,000 calls per trial-condition × 6 model variants (GPT-4o, GPT-3.5, Claude, Gemini, Llama-3.1-8B, Llama-3.1-70B) × 3 topics × 5 trials = **90,000 LLM calls**, and critically, most of these are against *paid* APIs (OpenAI, Anthropic, Google), not free local inference.
- RQ3 campaign: 10 stimulus pairs (assumed — not locked in your document, flagged) × 5 starting stances × 3 reps × 2 arms = 300 exposures — comparatively trivial, but each meme-condition call uses a vision-capable model, which is typically priced 3-10x higher per call than text-only depending on provider and image size.

**Total: ~120,000 text LLM calls plus ~150 vision calls, run across 4 paid provider APIs.** At even a conservative blended average of $0.01-0.03 per call (varies hugely by model and token count — GPT-4-tier models with full conversation history in context by turn 10 will run considerably higher per call than that), this is a **plausible $1,200-$3,600+ budget**, and that's before accounting for failed/retried calls, development/debugging runs (which realistically 2-5x the total call count before a "real" run set is finalized), or the fact that longer-context later-turn calls in the multi-turn mode cost more per call than early-turn ones. This is very likely **not viable on a typical student/small-lab budget**, and the original design says nothing about this — it needs to be surfaced and addressed with concrete de-scoping options, not left implicit. (Addressed in Part 3.)

**Ordering/race condition risk in the turn-based loop.** The design doesn't specify whether within a turn, all M agents' calls happen sequentially or concurrently. If concurrent (needed for practical runtime — sequential would mean 1,000 calls done one-at-a-time per trial, unacceptably slow), there's an unaddressed ordering question: does agent #47's neighbor sampling at turn 5 see agent #12's *turn-5* stance (if #12 was already processed this turn) or agent #12's *turn-4* stance (if not yet processed)? Ohagi's original design almost certainly assumes synchronous turns — all agents see the *same* prior-turn snapshot when selecting turn-t neighbors, updating simultaneously. The current design doesn't state this, and if implemented naively with concurrent-but-mutating shared agent state, different agents within the same turn could nondeterministically see different "current" stances for the same neighbor depending on execution order — an actual bug, not just an ambiguity, if `Agent.stance_history` is mutated in place during concurrent dispatch.

**No checkpoint/resume mechanism is actually specified, only named.** `checkpoint_manager.py` appears in the module table but §1-§7 never states *when* checkpoints are written or what "resume" actually restores. Given API calls fail routinely at this volume (network errors, provider outages, rate limit exhaustion), and a single RQ4/5 trial-condition run involves 1,000 sequential-dependent calls (turn t+1 depends on turn t's completed state), a failure at turn 8 of 10 without a real checkpoint policy means re-running the *entire trial from turn 0*, at full cost, or worse, silently corrupting partial state if the crash happens mid-write.

**Memory/storage blowup from full-text logging is moderate, not severe, but was not actually calculated.** 100 agents × 10 turns × 5 trials × 6 models × 3 topics ≈ 90,000 reason_text entries for the RQ4/5 campaign alone, each perhaps 50-150 words. This is a few hundred MB of raw text at most — not a real storage problem — but the design's claim of "no persistent database beyond flat files" was asserted without checking this, and should be stated as a verified conclusion, not an assumption.

**Vision vs text-only calls are not actually cost-tracked separately despite being priced differently.** `cost_tracker.py` is specified generically; §5's vision/text-only dispatch logic exists, but nothing connects the two — the cost tracker as described would need an explicit per-modality cost table lookup keyed on whether a given call included an image, which isn't stated.

---

## Critique Pass 3 — Architecture-Induced Validity Confounds

**Memory/context truncation likely differs systematically by language — a real confound for RQ1/RQ2, not just a coverage gap.** This is the same issue flagged in Pass 1 but reframed as a validity problem: Hindi-English code-mixed text (especially if agents write in Devanagari or a mixed script) very likely tokenizes at a different rate per "unit of meaning" than English, for any of the major tokenizers (BPE-based tokenizers generally fragment non-Latin scripts and code-switched text into more tokens per word than English). If `memory_window` truncation is token-budget-based (a common, reasonable default), Hinglish agents would systematically retain *fewer prior turns* in context than English agents for the same token budget — meaning by turn 8-10, an English agent might "remember" 6 prior neighbor posts while a Hinglish agent in the identical experimental cell only fits 4. This would make Hinglish agents look like they update opinion differently not because of any genuine language-driven reasoning difference (your actual RQ1/RQ2 hypothesis) but because they're mechanically working with less context — a sandbox implementation artifact masquerading as a research finding. **This is the single most serious validity risk in the whole design and the current spec does not address it at all.**

**The Interaction Engine's neighbor-sampling could behave differently across model backends for reasons unrelated to the models themselves — a real risk for RQ4/RQ5's "identical conditions" requirement.** `AlphaSampling.select_neighbors()` (§6) takes `all_agents` and reads their current `stance` values to compute sampling probabilities. But stance values are *produced by* the model backend being tested (via `stance_parser.py`). If different models fail to produce a cleanly parseable stance value at different rates (e.g., a smaller/weaker model occasionally returns malformed output that `stance_parser.py` has to handle with some fallback — unstated what that fallback is), then the *distribution of stance values feeding the sampling engine* is not actually apples-to-apples across model conditions purely as a parsing-robustness artifact, independent of any genuine difference in polarization behavior. This directly threatens RQ4/RQ5's core premise of "identical echo chamber parameters" across models — the parameters are identical, but the *data quality* feeding into them isn't, and nothing in the design addresses stance-parsing failure handling or its downstream effect on sampling.

**RQ3's meme vs. matched-text delivery is not actually symmetric as designed, beyond the intended modality difference.** §5 states both a multimodal prompt and a caption-only fallback prompt get constructed for meme stimuli, and separately a `matched_text` field exists for the text-arm stimulus (§3.4). But nothing confirms these two prompts (meme+caption prompt vs. matched-text prompt) are built from the *same underlying prompt template* with only the stimulus-delivery portion varying. If `prompt_builder.py` has any different instruction framing, different length, or different surrounding context between the two arms (e.g., "Here is a meme, describe your reaction" vs. "Here is an argument, describe your reaction" — subtly different verbs/framing, not just different stimulus content), that's an implementation-level asymmetry beyond the intended meme-vs-text variable, and the design doesn't specify a shared-template constraint anywhere. This is exactly the kind of confound Impiccichè-style rigor would flag in review.

**Token budget for the meme condition may differ from the text condition simply due to image encoding overhead**, independent of any prompt-template issue above — vision API calls typically consume additional "tokens" for image encoding (provider-dependent, often several hundred to a couple thousand equivalent tokens depending on resolution) that have no analog in the text condition. If your matched-text condition's prompt is deliberately length-matched to the caption alone (reasonable) but the meme condition implicitly costs more "attention budget" due to image tokens, this isn't a confound in the *stance outcome* necessarily, but it is a confound in your cost/rate-limiting design (Pass 2) that could lead to inconsistent truncation behavior between arms if the model has any input length limit interacting with the rest of its context — worth flagging even though its main manifestation is engineering rather than validity.

---

---

# PART 3: REVISED DESIGN AND VALIDATION TABLE

## Fixes Applied (mapped explicitly to flagged issues)

| Fix | Addresses |
|---|---|
| **Fix A — Explicit memory truncation policy: fixed turn-count window, not token-budget window.** `Agent.memory_window` is redefined to hold the **most recent 5 turns' worth of sampled neighbor posts** (a fixed count, not a token budget), regardless of language condition. If this makes Hinglish prompts longer in raw token count than English prompts for the same 5-turn window, that's an acknowledged, *measured and logged* side effect (log `prompt_token_count` per Interaction, already in schema §3.3) rather than a silent, uncontrolled truncation difference. Turn-count-based windowing is a deliberate, defensible default precisely because the alternative (token-budget windowing) is the mechanism that created the Pass 3 confound. | Pass 3 (memory/language confound), Pass 1 (RQ1 gap on unstated truncation policy) |
| **Fix B — Name concrete tools for RQ2's diagnostics.** `coherence_scorer.py` uses an LLM-as-judge rubric (a separate, fixed judge model — e.g., always GPT-4o regardless of which model generated the text being judged, to avoid a self-evaluation bias where each model judges its own output most favorably) scoring 1-5 on a stated rubric (logical consistency with stated stance, emotional appropriateness to topic). `sentiment_proxy.py` uses a multilingual transformer sentiment model with documented Hindi/code-mixed training data (e.g., a model from the IndicNLP or MuRIL family) rather than an unspecified placeholder — the specific model choice should be validated against a small hand-labeled sample before full-scale use, but the *approach* (multilingual transformer, not VADER, not a from-scratch tool) is now a stated design decision. | Pass 1 (RQ2 gap) |
| **Fix C — Split `Agent` into `DiscussionAgent` and `ExposureReceiver` as distinct types.** `ExposureReceiver` (RQ3) is a new, minimal dataclass containing only `receiver_id, model_backend_id, starting_stance, persona_id` — no `memory_window`, no `stance_history` list (just a single before/after pair, already captured in `ExposureResult`). `persona_id` references a **fixed pool of personas reused identically across the meme and matched-text arms for the same (starting_stance, repetition_index) cell** — enforced by `stimulus_manager.py` assigning persona *before* branching into the two arms, guaranteeing persona is held constant within a comparison pair. | Pass 1 (both RQ3 gaps: memory/schema conflation and persona confound) |
| **Fix D — Pin down RQ5's capability-tier proxy as a fixed, pre-declared mapping, decided before any runs, not derived from run results.** `capability_tier_mapper.py` uses a static lookup table set at experiment-design time (e.g., ordinal generation number: GPT-3.5=1, GPT-4o=2, Claude/Gemini current-gen=2 or 3 per your own judgment call, Llama-3.1-8B=1, Llama-3.1-70B=2 — the exact assignment is a defensible-but-arbitrary call you make once, document in the paper's methods section, and never revisit based on results). This is stated explicitly in the revised design as a decision with no uniquely correct answer, resolved by picking a documented default rather than leaving it open. | Pass 1 (RQ5 gap) |
| **Fix E — De-scope the trial/topic/model matrix to fit a realistic budget, with an explicit tiered fallback.** Default: **reduce topics from 3 to 1 fixed topic per campaign** (matching Ohagi's core design, which primarily used single-topic-per-experiment analysis even though multiple topics appeared across his paper's experiments) — this alone cuts total calls by 3x. Further: **reduce RQ4/RQ5's model set from 6 to 4** (GPT-4o, GPT-3.5, Claude, one Llama size — drop the second Llama size, since the primary capability-tier contrast is proxy-model vs. flagship, not fine-grained open-source scaling, which could be a follow-up rather than a launch requirement). Recomputed: RQ1 ≈ 100×10×2×5 = 10,000 calls; RQ4/5 ≈ 100×10×4×5 = 20,000 calls; RQ3 unchanged at 300. **New total ≈ 30,300 calls**, roughly a 4x reduction, bringing estimated cost into a more plausible several-hundred-dollar range depending on model mix — still non-trivial, and you should run a small pilot (e.g., 1 trial per cell) to get an actual empirical per-call cost average from your specific prompts before committing to the full trial count. | Pass 2 (cost blowup) |
| **Fix F — Synchronous-turn semantics, explicitly stated.** All agents' neighbor selection and prompt construction for turn *t* reads a frozen snapshot of turn *t-1*'s completed stance state; no agent's turn-*t* output is visible to any other agent's turn-*t* neighbor sampling. Implementation: `simulation_orchestrator.py` computes the *entire* turn's neighbor assignments first (using the frozen snapshot), dispatches all M calls concurrently (safe now, since no shared mutable state is read mid-dispatch), collects all results, and *only then* advances the snapshot for turn *t+1*. This removes the race condition entirely by construction rather than requiring locking. | Pass 2 (ordering/race condition) |
| **Fix G — Concrete checkpoint policy.** Checkpoint written after every completed turn (not mid-turn), containing the full frozen snapshot used for Fix F plus a `last_completed_turn` marker. Resume logic: on restart, `simulation_orchestrator.py` checks for an existing checkpoint for the `run_id`; if found, resumes from `last_completed_turn + 1` rather than turn 0. Worst-case re-run cost on failure is now capped at one turn's worth of calls (≤100), not the full trial. | Pass 2 (no resume mechanism) |
| **Fix H — Modality-aware cost tracking.** `cost_tracker.py` keys its per-call cost lookup on `(provider, model_id, modality)` where `modality ∈ {text, vision}`, using a small static pricing table maintained alongside the config (pricing changes over time and across providers, so this table needs periodic manual updates, flagged as an ongoing maintenance cost, not a one-time build cost). | Pass 2 (vision/text cost tracking not connected) |
| **Fix I — Stance-parse-failure handling, standardized and logged, not backend-specific.** `stance_parser.py` uses a single shared retry policy across all backends: on parse failure, retry the same call up to 2 times with an added instruction emphasizing the required output format; if still unparseable, log the interaction with `api_call_status = "failed_logged_null"` and **exclude that agent from that turn's neighbor-sampling pool** (rather than silently defaulting to some placeholder stance value, which would inject fabricated data into the Interaction Engine). This failure rate is itself logged per model_backend_id and reported in the paper as a data-quality metric — if one model has a meaningfully higher parse-failure rate, that's disclosed as a limitation, not hidden inside the sampling mechanism. | Pass 3 (stance-parsing robustness confound) |
| **Fix J — Shared prompt template enforced structurally, not by convention.** `prompt_builder.py` is restructured so there is exactly **one** template function, `build_exposure_prompt(stimulus_content, instruction_frame)`, called identically by both arms — the meme arm passes `stimulus_content = (image, caption)` and the text arm passes `stimulus_content = matched_text`, but `instruction_frame` (the surrounding "you are about to see X, please respond with Y" language) is the *same string* for both, with only a single templated noun ("the following meme" vs. "the following argument") varying. This is enforced by having a single function signature that both call sites must use, rather than two independently-written prompt-construction code paths that could drift. | Pass 3 (meme/text prompt asymmetry) |
| **Fix K — Image token overhead acknowledged, not solved, with a stated mitigation.** Since image-token overhead is a genuine property of vision APIs (not an implementation bug), the revised design logs `prompt_token_count` per exposure (already in schema) and treats "does image-token overhead affect response length or quality independent of content" as an explicit **robustness check to run before trusting RQ3's results** — e.g., confirm response length/coherence isn't correlated with prompt_token_count within the meme arm alone, before comparing across arms. This is stated as a required pre-analysis validity check, not silently left unaddressed. | Pass 3 (image token overhead) |

## Revised Module List Changes

- `agent_manager.py` now explicitly produces `DiscussionAgent` objects only (Fix C); a new `exposure_receiver_manager.py` (small, ~30 lines) handles `ExposureReceiver` creation and persona-pool assignment for RQ3, replacing the earlier plan to reuse `Agent`/`agent_manager.py` for both modes.
- `simulation_orchestrator.py`'s multi-turn path is restructured per Fix F: **turn-batch neighbor resolution → concurrent dispatch → collect → snapshot advance**, as a named internal sequence, not left as an implementation detail.
- `checkpoint_manager.py` gains a concrete contract: `save(run_id, turn, agent_snapshot)` / `load(run_id) -> (turn, agent_snapshot) | None`, called by the orchestrator at the end of Fix F's per-turn sequence.
- `cost_tracker.py`'s pricing table is now an explicit external config file (`pricing_table.yaml`), not inline logic, so it can be updated without code changes as providers change pricing.

---

## Final Validation Table

| RQ | Sandbox components depended on | Simulation mode/config | Data logged | Metric(s) computed | Support judgment |
|---|---|---|---|---|---|
| **RQ1** | `agent_manager`, `interaction_engine` (`AlphaSampling`), `model_backend` (LiteLLM, text-only), `prompt_builder` (Fix A: fixed 5-turn window), `logging_writer`, `graph_export`, `network_metrics`, `stance_regression`, `embedding_cluster`, `codemix_ratio_tracker` | Multi-turn, M=100/N=5/K=10, `language_condition ∈ {english, hinglish}`, alpha fixed, 5 trials, **1 topic** (Fix E) | `interactions.jsonl` per run: stance_before/after, reason_text, neighbor_agent_ids, prompt_token_count | RWC/BC/Dipole/etc. (12 metrics) per language condition; Ohagi-style regression per condition; code-mixing ratio as covariate | **YES** |
| **RQ2** | Same runs as RQ1 (no new simulation) + `coherence_scorer` (Fix B: LLM-judge rubric), `sentiment_proxy` (Fix B: multilingual transformer), `codemix_ratio_tracker` | Re-analysis of RQ1's `interactions.jsonl` — no separate mode | Same `interactions.jsonl`, plus derived per-turn coherence/sentiment/code-mix scores | Correlation between coherence/sentiment instability and polarization speed (time-to-bimodal or stance-variance-over-time, computed via `bimodality_analysis`) | **YES**, contingent on Fix B's specific tool choices being empirically validated against a hand-labeled Hinglish sample before full-scale use — flagged as a pre-analysis step, not a sandbox gap |
| **RQ3** | `exposure_receiver_manager` (Fix C), `stimulus_manager`, `prompt_builder` (Fix J: shared template), `vision_fallback`, `model_backend` (vision + text dispatch), `cost_tracker` (Fix H), `logging_writer` | Single-exposure, no M/N/K, 5 starting stances × 3 reps × {meme, matched_text} arms per stimulus pair, persona held constant per Fix C | `exposures.jsonl`: receiver_starting_stance, stance_after, reason_text, used_vision_fallback flag, prompt_token_count | Opinion-shift magnitude (stance_after − starting_stance) compared meme vs. matched_text, controlling for prompt_token_count per Fix K's pre-check | **YES**, contingent on completing Fix K's robustness pre-check (token-overhead-vs-response independence) before trusting the primary comparison — stated as a required step, not an open gap |
| **RQ4** | `agent_manager`, `interaction_engine`, `model_backend` (all backends via Fix I's shared parse-failure policy), `stance_parser`, `graph_export`, `network_metrics`, `bimodality_analysis` | Multi-turn, M=100/N=5/K=10, alpha/topic/M/N/K fixed across `model_backend_id ∈ {gpt-4o, gpt-3.5, claude, llama-3.1-70b}` (Fix E: 4 models), 5 trials, **1 topic** | `interactions.jsonl` per model-condition run, plus per-model stance-parse-failure rate (Fix I) | Bimodality coefficient/classification per model; convergence speed (turns-to-bimodal) per model | **YES** |
| **RQ5** | Same runs as RQ4 (no new simulation) + `capability_tier_mapper` (Fix D: fixed pre-declared mapping) | Re-analysis of RQ4's runs, grouped by tier instead of individual model | Same as RQ4 | Correlation between declared capability tier (Fix D) and bimodality/convergence-speed metrics from RQ4 | **YES**, contingent on the capability-tier assignment (Fix D) being stated and justified in the paper's methods as a declared, non-circular choice — this is a documentation requirement, not a code gap |

**Overall: 5/5 YES**, with three explicitly load-bearing pre-analysis validity checks (RQ2's tool validation, RQ3's token-overhead check, RQ5's tier-mapping justification) that must be completed and reported, not silently assumed, for the YES to hold in a way that survives review.
