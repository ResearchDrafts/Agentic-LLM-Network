# Phase 5 and 6 Plan: Post-hoc Analysis, CLI, and a Self-Hosted Model Strategy

Status: **planning only, not started.** Phases 0 through 4 are merged: Tiers 0-3 built, 15 modules, 201 passing tests, eight defects found and fixed (`phase4.md` Section 0). The system runs a simulation end to end against a mocked gateway, and **has never made a real API call.**

Per the convention in `CLAUDE.md`, update this status line as the phase progresses rather than leaving it at its pre-implementation value. Both `plan.md` and `phase3.md` claimed "planning only, not started" long after shipping, for exactly that reason, until Phase 4 corrected them.

Scope: two phases in one document, because they share a question neither architecture document addresses: **where inference actually runs.** Both docs assume a hosted API and a per-token bill. This project will instead **self-host open-weight models on free or cheap GPU** (Kaggle T4 x2, Colab compute units), which changes what Phase 6 must build and what RQ3 can claim.

- **Section A** resolves the model and serving strategy. No design-doc counterpart exists.

> **Revision note.** An earlier draft of Section A recommended hosted free-tier APIs and narrowed RQ3 to caption-only, on the mistaken premise that using HuggingFace models meant paying for HuggingFace's hosted inference. That conflated open weights with hosted inference (A.1). Self-hosting is free, uncapped, and restores RQ3's original claim. The error is recorded rather than quietly overwritten, because the reasoning behind it is exactly the kind that recurs.
- **Section B** is Phase 5: the six post-hoc analysis modules (`full_design_doc.md` §3.13).
- **Section C** is Phase 6: the CLI and Python API (§6).
- **Section D** carries forward what remains open.

---

## Section A: Model and serving strategy

### A.1 Open weights are not hosted inference, and the difference is not cosmetic

This distinction produced a wrong recommendation during planning. It is written down because it is an easy mistake and an expensive one.

Two things share the name "using a HuggingFace model":

| Path | What happens | Cost |
|---|---|---|
| `from_pretrained()` + `HF_TOKEN` | Weights download, **your** hardware runs them | **Free and unlimited.** The token authenticates *downloads* of gated repos, not usage |
| `InferenceClient`, `router.huggingface.co` | **HuggingFace's** servers run them | $0.10/month credit on a free account |

HuggingFace's own pricing docs state the first plainly: running models locally with `transformers` costs nothing through HuggingFace, and the token exists for authentication. The $0.10 figure applies only to their **hosted** inference product, which happens to share the brand.

So the real question was never "can these models be used". It is **where do 20,000 generative calls get computed**, and that is a hardware question with a completely different answer.

Kimi K2 still illustrates the upper bound: 1T parameters, roughly 630 GB at INT4, needing about 8xH200. Genuinely open, genuinely out of reach here. But that is a statement about *Kimi's size*, not about open models generally. A 7B model is four orders of magnitude cheaper to serve and sits comfortably on a free T4.

### A.2 Available compute, measured

| Option | Hardware | Limit | Suits |
|---|---|---|---|
| **Kaggle** | **T4 x2** (16 GB each, 32 GB total) | 30 GPU-h/week, 9-h sessions, no card | The primary campaign |
| **Colab (paid units)** | L4 24 GB / A100 40 GB | Pay per compute unit | Larger models, longer sessions, vision |
| Colab free | T4 16 GB | ~15-30 h/week, not guaranteed | Pilots only |
| Local M1 | 8 GB unified | unlimited, ~10-15 tok/s | Phase 5 encoders, not generation |
| Groq / Cerebras | hosted | 1,000-1,800 calls/day, free | **Fallback when GPU hours run out** |

**A trap worth stating loudly: Kaggle's P100 cannot run vLLM.** vLLM requires CUDA compute capability 7.0 or above; the P100 is 6.0 and raises `RuntimeError` at startup. Select **T4 x2**, not P100. The T4 is 7.5, which clears 7.0 but is below the 8.0 that bfloat16 needs, so serving must pass `--dtype half`.

### A.3 Resolved: self-host open-weight models with vLLM

Primary path. The models are the ones originally intended (Qwen, Llama, and their vision variants, straight off the Hub with `HF_TOKEN`); only the compute moves.

Three reasons this beats the hosted free tiers:

1. **No request cap.** Hosted free tiers cap at 1,000-1,800 calls/day, making the 20,000-call campaign a multi-week affair. A T4 x2 session has no per-request limit, only wall-clock.
2. **Continuous batching matches the workload exactly.** The orchestrator dispatches M=100 agents concurrently per turn via `asyncio.gather`. That is precisely what vLLM's continuous batching is built for, so aggregate throughput is far above the single-stream figures usually quoted for a T4.
3. **Vision becomes possible**, which restores a research claim (A.6).

Groq and Cerebras remain configured as a **fallback**, not deleted. If GPU hours run out mid-campaign, a text-only run can continue there, and because both serve `gpt-oss-120b` the fallback is at least internally consistent. Any switch mid-campaign must be recorded per run, since it is a model change and therefore a confound if it happens inside one comparison.

### A.4 Model selection

**Text (RQ1, RQ2, and the meme-absent arm of RQ3):** a Qwen2.5-7B-Instruct class model. ~14.9 GB at FP16, so it fits one T4 with little headroom, or comfortably across T4 x2. Qwen has the strongest documented Hindi performance among open models at this size, which is the single biggest risk to RQ1 (A.10).

**Vision (RQ3's meme-receiving turns):** **Qwen2.5-VL-7B-Instruct**, ~18 GB at FP16. That does **not** fit a single 16 GB T4. Options, in order of preference:

- Kaggle **T4 x2** with `--tensor-parallel-size 2 --dtype half`: 32 GB total, fits.
- Colab **L4 (24 GB)** or **A100 (40 GB)** on purchased units: fits on one device, simpler.
- **INT4 quantization**, roughly 4.5 GB, fits anywhere, at some quality cost that should be sanity-checked before use.
- **Qwen2.5-VL-3B** if throughput matters more than capability.

**Use one model family across a whole comparison.** Mixing text and vision models inside a single run reintroduces the cross-model confound `sandbox_hld.md:346` rules out of scope. Since Qwen2.5-VL handles text as well as images, the clean design is to serve the **VL model for the entire RQ3 campaign**, text-only turns included, so meme and non-meme turns differ in content rather than in which model produced them.

### A.5 Serving architecture

vLLM exposes an OpenAI-compatible server, so nothing in the orchestrator needs to know it is talking to a local process:

```
Kaggle / Colab notebook
  vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
      --tensor-parallel-size 2 --dtype half --port 8000
        |
        |  OpenAI-compatible HTTP on localhost:8000
        v
  SimulationOrchestrator  ->  ModelGateway  ->  litellm (hosted_vllm/...)
```

`_build_messages` (`model_gateway.py:139-151`) already emits the standard multimodal payload with a base64 `image_url` data URI, which is exactly what vLLM's OpenAI server accepts. **The vision path therefore needs no change to message construction**, only a model that accepts images.

### A.6 Vision restored, and RQ3 keeps its original claim

With a self-hosted VL model, meme-receiving agents genuinely see the image. The narrowing forced by hosted free tiers is reverted:

> RQ3 tests whether **visual meme content** accelerates convergence, as the HLD originally intended.

`vision_fallback.py` returns to being a genuine fallback rather than the permanent state, and `used_vision_fallback` should be **False** for essentially every meme encounter. That makes it a real monitoring signal: if it starts coming back True, something has silently degraded, and the field will say so. Assert it in the campaign's sanity checks rather than only logging it.

This is strictly better than the caption-only compromise and is the main reason self-hosting is worth the setup cost.

### A.7 Code changes required

**`ModelGateway` cannot currently reach a local server.** `_call_with_retry` calls `litellm.acompletion(model=..., messages=..., temperature=...)` with no `api_base`, so there is no way to point it at `localhost:8000`. This is the one blocking change:

```python
class ModelGateway:
    def __init__(self, model_backend_id, rate_limiter, cost_tracker,
                 api_base: str | None = None): ...
    # forwarded to litellm.acompletion(..., api_base=self._api_base)
```

Add a matching optional `api_base: str | None` to `ExperimentRun` so it is captured in `run_config.json` and therefore reproducible, rather than living in an environment variable that no record preserves.

**`pricing_table.yaml`** gains a self-hosted provider. `CostTracker._lookup_rate` (`cost_tracker.py:62-69`) raises on a missing entry, so this is mandatory:

```yaml
# Self-hosted via vLLM. Genuinely zero per-token cost, unlike the
# placeholder rates above: the compute is rented by the hour, not billed
# per token, so per-call cost accounting does not apply.
hosted_vllm:
  Qwen2.5-7B-Instruct:
    text: {prompt_per_1k: 0.0, completion_per_1k: 0.0}
  Qwen2.5-VL-7B-Instruct:
    text: {prompt_per_1k: 0.0, completion_per_1k: 0.0}
    vision: {prompt_per_1k: 0.0, completion_per_1k: 0.0}
```

The `vision:` modality is required: `generate()` selects `modality = "vision" if image is not None else "text"`, so a VL model without it raises on the first meme turn.

**`model_gateway.py:30-31`** gains the new ids:

```python
VISION_CAPABLE = {..., "Qwen2.5-VL-7B-Instruct", "Qwen2.5-VL-3B-Instruct"}
TEXT_ONLY = {..., "Qwen2.5-7B-Instruct", "gpt-oss-120b", "gpt-oss-20b"}
```

`_resolve_vision_support` raises on an unrecognized id at construction, so an unlisted model cannot start.

**Rate limiting is deliberately absent for self-hosted runs.** Leave `rate_limits` empty: `RateLimiter.acquire` early-returns for an unconfigured provider (`rate_limiter.py:28-29`), giving full concurrency, which is what vLLM's batching wants. Note in-file that this is intentional, not an omission, because the existing limiter serializes a provider to one request at a time and would destroy throughput here.

**Not needed:** `MultiHostGateway` and `QuotaExhaustedError`, both designed for the hosted-API path. Keep them specified in case the fallback is used, but neither is on the critical path any more.

### A.8 Session limits, and the thing that will bite

Kaggle sessions cap at 9 hours and Colab sessions end too. Phase 4's D8 fix means a resumed run is byte-identical to an uninterrupted one, so a campaign spanning sessions produces exactly the data a single continuous run would.

**But the notebook filesystem is ephemeral.** `runs/{run_id}/` holds `interactions.jsonl` and the checkpoint that resume depends on, and it vanishes with the session. Losing it means losing the campaign, silently, at the worst possible moment.

Mitigation, to be part of the notebook template rather than left to memory:
- Colab: mount Drive and point `--runs-dir` at it.
- Kaggle: write to `/kaggle/working/` and rely on output persistence, or push to a Dataset between sessions.
- Either: verify the checkpoint is readable from the persistent location **before** starting turn 1, not after the session dies.

### A.9 Throughput and campaign planning

Single-stream tok/s figures for a T4 badly understate this workload, because 100 concurrent requests per turn is the case continuous batching is designed for. Rather than guess, **measure it in the pilot**: run one M=100/K=1 turn, record wall-clock, and extrapolate. `--dry-run` (C.3) should report that estimate in GPU-hours against Kaggle's 30/week budget.

Order of magnitude: 20,000 calls at roughly 150 output tokens each is about 3M output tokens. That is a few GPU-hours at plausible batched throughput, comfortably inside one week's free Kaggle quota, with Colab units available if a run needs to finish in one sitting.

### A.10 The validity gate: check Hinglish before spending the campaign

Unchanged by the move to self-hosting, and still the largest non-code risk. **If the chosen model produces poor or inconsistent Hinglish, RQ1 measures model deficiency rather than a language effect, and no post-hoc analysis repairs that.**

This mirrors `sandbox_hld.md:439`, where RQ2's support is already conditional on validating the diagnostic tools against hand-labelled Hinglish. The same logic applies one level earlier, to generation.

**Gate, to run before the campaign:**

1. Generate ~50 responses per language condition with the real template, real temperature, a spread of personas and stances.
2. Hand-check register: is the Hinglish genuinely code-mixed, or English with Hindi nouns, or Hindi with English loanwords? Either degenerate case breaks the comparison.
3. Compute the code-mix index (Q5.5) and confirm `english` and `hinglish` distributions actually separate.
4. Confirm the `STANCE:` format holds in both. A higher parse-failure rate in one language is itself a confound, since `failed_logged_null` rows are dropped from analysis and would be dropped unevenly.

Self-hosting makes this gate cheaper, not less necessary: iterating on model choice costs GPU minutes rather than a day's API quota. Record the outcome either way. A negative result is a finding that changes the model, obtained for ~100 calls instead of 20,000.

**Also report:** Phase 4 measured the three language directives differing by a fixed 42 characters, 2.87% of a realistic prompt. State it alongside the `prompt_token_count` diagnostic (Q5.9) rather than leaving it implicit.

## Section B: Phase 5, post-hoc analysis (§3.13)

Six modules plus a shared loader, all offline, operating only on a completed run's `interactions.jsonl`. No dependency on the Orchestrator, only on its output format, which Phase 4 locked and `tests/test_integration_run.py` now pins. **Phase 5 is unblocked today and can proceed in parallel with Phase 6.**

### Files to create

Proposed as a subpackage, `sandbox/analysis/`:

- `sandbox/analysis/__init__.py` (exports `load_run_dataframe`)
- `sandbox/analysis/loader.py`, `stance_regression.py`, `bimodality.py`, `codemix.py`, `sentiment.py`, `embeddings.py`, `coherence.py`, `diagnostics.py`
- one test module each

**Deviation, stated deliberately:** §3.13 lists these as flat modules in `sandbox/`. A subpackage is proposed instead because `sandbox/` already holds 15 files and these eight are a different *kind* of thing: offline, pandas-shaped, and unable to import the orchestrator. `CLAUDE.md` already describes "Post-hoc analysis" as its own tier, so the package mirrors the documented architecture rather than departing from it.

### Q5.1: the shared loader

```python
def load_run_dataframe(run_dir: Path) -> pd.DataFrame
```

Every module starts here. `tests/test_integration_run.py::test_the_log_is_ready_for_the_phase_5_analysis_modules` already asserts the columns and dtypes it must produce, including that `neighbor_agent_ids` survives as real lists rather than stringified ones, so the loader's contract is pinned before it exists.

Add `run_id` validation and a clear error when the directory has no `interactions.jsonl`, rather than surfacing a pandas error.

### Q5.2: `neighbor_avg_stance`, and the trap inside it

This is the one item in Phase 5 that can **silently corrupt a headline result** rather than fail loudly, so it is specified first.

§3.13's sketch is `df["neighbor_avg_stance"] = df.apply(lambda row: _neighbor_avg(row, df), axis=1)`. Two problems, and the second is the dangerous one.

**Performance:** that is O(n^2). At 20,000 rows it is slow but survivable, and easily fixed with a precomputed index.

**Correctness:** *which turn's stance gets averaged?* A speaker at turn *t* saw its neighbours as they were **at the end of turn t-1**. That is Fix F, the frozen-snapshot guarantee the whole orchestrator is built around. Averaging neighbours' turn-*t* `stance_after` averages values the speaker could not possibly have seen, and produces wrong regression coefficients on RQ1's primary result with no error and no warning.

**Resolved:**

```python
# (turn, agent_id) -> stance_after, built once
stance_at = df.set_index(["turn", "speaker_agent_id"])["stance_after"].to_dict()

# a row at turn t averages its neighbours as of turn t-1
def neighbor_mean(row):
    prior = [stance_at.get((row.turn - 1, n)) for n in row.neighbor_agent_ids]
    prior = [s for s in prior if s is not None]
    return sum(prior) / len(prior) if prior else float("nan")
```

Turn 1 rows have no prior turn and therefore no neighbour average. They must be **excluded from the regression**, not filled with zero or with the neighbours' initial stances. Missing neighbours (an agent whose turn t-1 was `failed_logged_null`) are dropped from that row's mean, which is the Fix I exclusion policy applied consistently downstream.

Write a test with hand-computed expected values on a tiny fixture. This is not a function to trust by inspection.

### Q5.3: Fix N filtering, enforced rather than documented

`coherence.py` and `sentiment.py` must filter to `content_type == "generated_text"` **inside** the function, not rely on the caller. A meme row's `reason_text` is a dataset caption, not the agent's own reasoning, so scoring it measures the wrong thing entirely.

`sandbox_hld.md:417` (Fix N) states this as a requirement; Phase 4's experience is that requirements stated only in prose get missed. Filter internally, and return the filtered frame so the row count change is visible to the caller rather than silent.

### Q5.4: bimodality, RQ3's primary metric

**Resolved: the bimodality coefficient**, needing only numpy and scipy, both already available through pandas and statsmodels.

```
BC = (skewness^2 + 1) / kurtosis        BC > 0.555 suggests bimodality
```

(0.555 is the value for a uniform distribution; above it, the distribution is more bimodal than uniform.)

**`turns_to_bimodal`** is defined as the first turn where BC crosses 0.555 **and stays above it for every subsequent turn**. The "and stays" clause matters: a single-turn excursion is noise, and without the clause RQ3's headline metric becomes unstable at exactly the low agent counts where it is least reliable.

Return `None`, never a sentinel like `-1` or `K+1`, when a run never converges. A magic number will eventually be averaged into a result by accident.

Note in-file that **Hartigan's dip test** is the statistically stronger instrument and would need the `diptest` package. Recommend running it as a robustness check on the final distributions once the main result exists, rather than adding the dependency now.

### Q5.5: the code-mix index, and its honest error bar

CMI is RQ1's key covariate and also **the weakest link in the analysis chain**, because it depends on token-level language identification for *romanized* Hindi, which is genuinely hard. "Main office ja raha hoon" has no Devanagari to key on, and words like "main", "to" and "is" are valid in both languages.

Standard formula, for an utterance of `n` language-tagged tokens:

```
CMI = 100 * (1 - max(w_lang) / n)      0 = monolingual, higher = more mixed
```

**Resolved:** implement a lexicon-based tagger (an English wordlist plus a romanized-Hindi list, with ambiguous tokens tagged `unknown` and excluded from `n` rather than guessed), and compare it against `lingua` on the A.10 validation sample. Pick whichever separates the two conditions more cleanly on hand-labelled data.

Whichever is chosen, **report its measured accuracy on that hand-labelled sample in the paper.** Presenting CMI as an exact quantity when it rests on approximate token-level LID would be the sort of overclaim that is easy to avoid and awkward to defend.

### Q5.6: sentiment

Fix B (`sandbox_hld.md:409`) explicitly forbids VADER, which is English-only and would produce meaningless scores on Hinglish, systematically differing between RQ1's arms in exactly the way that manufactures a false result.

**Resolved:** `cardiffnlp/twitter-xlm-roberta-base-sentiment`, run locally. Multilingual, trained on social text, and handles romanized input more gracefully than most alternatives. Score all rows in one batched pass; per-row inference on 20,000 rows is needlessly slow.

### Q5.7: embeddings and clustering

**Resolved:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (fast, ~470 MB, good multilingual coverage). LaBSE is the stronger alternative at roughly 1.8 GB, which is worth weighing on an 8 GB machine.

Cluster with KMeans over a small range of k, selecting by silhouette score, and record the chosen k per run. Ohagi's method is cluster-based, so the clusters are a reported artifact, not an implementation detail.

### Q5.8: the coherence judge, and self-evaluation bias

**Resolved: judge with a different model family than the one being judged.**

If the simulation runs on Qwen, judge with something else (a Llama-family instruct model, or `gpt-oss-120b` on Groq's free tier, which is well suited here precisely because judging is a one-off offline pass rather than a 20,000-call campaign).

**Why a different family.** A model grading its own output rates its own style as more coherent. If RQ2's claim is "reasoning quality differs by language" and the judge shares the generator's blind spots, particularly in Hinglish where both are likely weakest, that bias lands directly on the headline result. A different family breaks the coupling. This still satisfies Fix B's requirement of a *fixed* judge, which is about consistency across the corpus, not about matching the generator.

**On throughput.** Judging touches only `content_type == "generated_text"` rows (Q5.3), so the volume is well below the run's total and lower again for RQ3 runs where a fraction of turns are memes. That makes it small enough to fit a hosted free tier comfortably, which is convenient: the judge does not compete with the self-hosted GPU the simulation is using, and the two never contend for the same resource.

**Hinglish caveat.** Whatever judge is chosen must itself be checked on Hinglish before use, exactly as A.10 checks the generator. A judge that scores code-mixed text poorly would manufacture RQ2's result rather than measure it. Fold this into the A.10 sample: score the same ~50 responses per condition and confirm the judge's ratings are not systematically depressed for Hinglish independent of quality.

**Route judge calls through `ModelGateway`.** §3.13 leaves this open and recommends it; the recommendation should be taken, because the judge needs exactly the same retry, rate-limit and cost protections as the simulation, and reimplementing them for an offline script is how the two drift apart.

### Q5.9: the Fix A diagnostic as a standard report

Not in §3.13, added because Phase 4's D7 work made it load-bearing.

`diagnostics.py` reports **mean `prompt_token_count` per language condition, with a significance test.** This turns "memory was capped at 5 turns" from an assertion into a measured result:

> Mean prompt length was X tokens for English and Y for Hinglish (difference not significant, p = Z), so there is no truncation asymmetry between conditions.

Fix A counts memory in *turns* rather than tokens precisely to prevent one language silently receiving less context. This is the evidence that it worked, computed from the run's own logged data at zero extra cost. Report the 42-character directive difference (A.10) alongside it.

The same diagnostic applies to RQ3, where meme turns and generated-text turns occupy equal memory slots but unequal token counts, which `sandbox_hld.md:394` flags as a covariate worth checking.

### Q5.10: dependencies

Adds `torch`, `transformers`, `sentence-transformers`: roughly 2-3 GB installed.

Worth stating plainly, because it sounds heavier than it is: these are **small encoder models**, not generative LLMs. A 470 MB MiniLM scoring 20,000 short texts on an M1 is a matter of minutes, and runs fully offline after the one-time download. **This is where HuggingFace genuinely fits this project**: not as an inference host for the simulation, which its free tier cannot support, but as the source of small local models for offline analysis, which is what it is good at.

The pandas and statsmodels modules (Q5.1, Q5.2, Q5.4, Q5.9) need **no new dependencies at all** and can ship first if the install weight is unwelcome.

---

## Section C: Phase 6, CLI and Python API (§6)

Fully blocked on Phase 4, which is now complete.

### C.1 `sandbox/api.py`

`launch_run`, `get_run_status`, `load_run_dataframe`, `list_runs`, per §6.2. One import surface for a researcher in a notebook, so nobody needs to know which of 20-plus internal modules to call. Thin wrappers; no logic of their own.

`launch_run` must not swallow errors. A notebook user needs `ConfigLoadError` and `CostCeilingExceeded` to surface directly.

### C.2 `sandbox/__main__.py`

```
python -m sandbox run --config <path> [--dry-run]
python -m sandbox status --run-id <id>
python -m sandbox resume --run-id <id>
python -m sandbox list [--rq-target ...] [--status ...]
```

Per §6.1: exit codes 0 (completed), 1 (config invalid, nothing launched), 2 (run failed, checkpoint may exist), 130 (SIGINT, last checkpoint preserved). Structured JSON progress on stdout, one event per completed turn; human-readable logs on stderr. Keeping the streams separate is what makes stdout pipeable into a monitoring script without filtering.

### C.3 `--dry-run` reports GPU-hours, not dollars

§6.1 specifies config validation plus a cost estimate. Self-hosted, **the binding constraint is neither dollars nor request caps but GPU-hours and session length**, so that is what the estimate should report:

```
$ python -m sandbox run --config configs/rq1_english.yaml --dry-run

Config valid.
  100 agents x 10 turns x 1 trial     = 1,000 calls
  backend: hosted_vllm @ localhost:8000 (Qwen2.5-7B-Instruct)
  measured: 2.1 s/turn-batch in pilot    -> ~0.6 GPU-h
  full campaign (20 runs)                -> ~12 GPU-h of 30/week

  WARNING: session limit is 9 h. This run spans a session boundary;
           confirm --runs-dir is persistent before starting.
```

The estimate should come from a measured pilot (A.9), not a guess, and the session-boundary warning is the one that actually prevents lost work.

### C.4 A campaign runner

Follows directly from the scope decision (A.4). Twenty runs over about seven days needs something that runs until quota is exhausted, checkpoints, waits, and resumes the next day.

Phase 4 made this safe rather than merely possible: defect D8's fix means **a resumed run is byte-identical to an uninterrupted one**, so a campaign stopped and restarted daily produces exactly the data a single continuous run would have. Without that fix, every daily boundary would have introduced a silent discontinuity in the neighbour-sampling stream.

Proposed: `python -m sandbox campaign --config-dir configs/rq1/ [--daily-cap N]`, iterating configs, running each until `QuotaExhaustedError` (A.6), and exiting cleanly with a resume hint.

### C.4a The notebook is the real deployment target

Worth stating because it is not what §6 assumes. The campaign runs inside a Kaggle or Colab notebook alongside the vLLM server, not from a terminal on a workstation. Two consequences for Phase 6:

- `sandbox/api.py` matters more than the CLI. `await launch_run(config_path)` in a cell is the primary interface; `python -m sandbox run` is for local testing.
- The campaign runner (C.4) must tolerate the **session** ending, not just quota exhausting. Same mechanism, different trigger: run, checkpoint, die, resume next session. A.8's persistence requirement is what makes that survivable.

### C.5 Fix the CWD-relative paths

`Path("data/memes")`, `Path("data/personas")`, `Path("runs")` and `Path("pricing_table.yaml")` are resolved against the process working directory in `config_loader.py`, `agent_manager.py`, `meme_pool_manager.py`, `cost_tracker.py` and `checkpoint_manager.py`. The system therefore only works when invoked from the repo root.

That was tolerable while the only entry point was pytest. It becomes untenable in a notebook, where the working directory is `/kaggle/working` or `/content` and the repo lives somewhere else entirely, so every asset lookup fails with a confusing missing-file error. Resolve paths against a project root discovered from the config file's location, and accept an explicit `--runs-dir` so output can be written to mounted Drive or persistent Kaggle output (A.8).

### C.6 Optional, explicitly non-load-bearing

- `runs_manifest.sqlite` (§5.4): a derived, rebuildable index, never the source of truth. Safe to delete and regenerate at any time.
- `status_server.py` (§6.3): a read-only JSON endpoint for monitoring a long run from a second terminal. Genuinely useful given multi-day campaigns, but strictly optional.

---

## Section D: Still open

Carried forward from `phase4.md`, not closed by this plan.

- **Credentials.** `HF_TOKEN` for gated model downloads (Llama and similar; Qwen is generally ungated). Free accounts for Groq and Cerebras only if the fallback path is used. Never commit tokens; read them from the environment. Note that `HF_TOKEN` authenticates *downloads*, so it is needed once per session when the weights are fetched, not per inference call.
- **A real persona pool.** Only `test_pool.jsonl` (4 entries) and `pool_20.jsonl` (20, test-only) exist. With M=100 and a 20-entry pool, each persona repeats across 5 agents, which is a real design choice about population diversity rather than a fixture detail, and should be made deliberately.
- **Meme `stance_label` rescaling.** `MemeContent.stance_label` is a bare float with no link to `stance_scale`, and `MemePoolManager` never receives the run, so it structurally cannot validate. The test fixture's 1-7 range is coincidental. Any real meme dataset (typically 0-1, -1..+1, or 1-5) silently produces garbage until a rescale step exists.
- **`pricing_table.yaml`'s legacy entries.** The openai/anthropic/google rates remain unvalidated placeholders, as the file's own header says. They are now unused but still loadable, so either validate or remove them rather than leaving a trap.

---

## Sequencing

1. **Section A.7 code changes.** `api_base` on `ModelGateway` and `ExperimentRun`, the `hosted_vllm` pricing entries, the vision/text model tables. Small, and blocks every real call. `MultiHostGateway` and `QuotaExhaustedError` are fallback-only and can wait.
2. **Stand up vLLM on Kaggle and smoke-test one turn.** M=5, K=1 against the real server. Confirms `api_base` plumbing, `--dtype half`, and the T4 x2 tensor-parallel setup before any of it matters. Also produces the throughput number A.9 and C.3 both need.
3. **The A.10 Hinglish validation gate.** About 100 calls, now GPU-minutes rather than a day's API quota. A negative result changes the model, so do it before anything expensive.
4. **Phase 5's dependency-free modules** (loader, regression, bimodality, diagnostics). Unblocked today, and they make the first real run's output interpretable immediately rather than weeks later.
5. **Phase 6's `api.py`, persistence, and campaign runner.** A.8's persistent `--runs-dir` is the part that prevents losing a campaign to a session timeout; do not defer it.
6. **The real campaign.** A few GPU-hours of the weekly Kaggle budget, spread across sessions.
7. **Phase 5's model-backed modules** (sentiment, embeddings, coherence) against real data rather than fixtures.

Steps 4 and 5 are independent. Step 2 should not be skipped: it is the first time any code in this project talks to a real model, and it is far cheaper to find the plumbing problems on five agents than on a hundred.
