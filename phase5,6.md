# Phase 5 and 6 Plan: Post-hoc Analysis, CLI, and a Zero-Cost Model Strategy

Status: **planning only, not started.** Phases 0 through 4 are merged: Tiers 0-3 built, 15 modules, 201 passing tests, eight defects found and fixed (`phase4.md` Section 0). The system runs a simulation end to end against a mocked gateway, and **has never made a real API call.**

Per the convention in `CLAUDE.md`, update this status line as the phase progresses rather than leaving it at its pre-implementation value. Both `plan.md` and `phase3.md` claimed "planning only, not started" long after shipping, for exactly that reason, until Phase 4 corrected them.

Scope: two phases in one document, because they share a constraint that neither architecture document anticipated. The project will run on **free inference only**, and that is not a deployment detail. It bounds the experiment's calendar, narrows one research claim, and adds the only genuinely new runtime component in either phase.

- **Section A** resolves the model and provider strategy. No design-doc counterpart exists.
- **Section B** is Phase 5: the six post-hoc analysis modules (`full_design_doc.md` §3.13).
- **Section C** is Phase 6: the CLI and Python API (§6).
- **Section D** carries forward what remains open.

---

## Section A: Model and provider strategy

### A.1 Open weights are not free inference

This distinction caused a wrong answer during planning and will cause more later if it is not written down.

Llama, Qwen, Kimi and GPT-OSS are **open-weight**: free to download, free of licence fees, all published on HuggingFace. That says nothing about the cost of *running* them. Inference needs GPUs, and somebody pays for the GPU.

Kimi K2 makes the point concretely. It is 1 trillion parameters (32B active, MoE), genuinely open under a Modified MIT licence, and about **630 GB of weights at INT4**. Serving it properly takes roughly 8xH200. The development machine here is an Apple M1 with 8 GB of unified memory, so it is short by a factor of about 1,700, and no free host serves it. The model is free and unusable at the same time, with no contradiction.

So "free model" splits into two questions that must be answered separately: *are the weights open* (usually yes) and *who is paying for the compute* (usually not free).

### A.2 What is actually free, measured

Checked during planning. Numbers move; re-verify before a real campaign rather than trusting this table.

| Host | Free chat models | Real free limit | 20,000 calls takes |
|---|---|---|---|
| **Groq** | `gpt-oss-120b`, `gpt-oss-20b`, `qwen3.8-27b` | 30 RPM, **1,000 RPD**, 8K TPM | ~20 days |
| **Cerebras** | `gpt-oss-120b`, `glm-4.7` | 5 RPM, 1M tokens/day (~1,800 calls) | ~11 days |
| HuggingFace Inference | anything on HF | **$0.10 per month** | not viable |
| OpenRouter `:free` | 28+ models | 20 RPM, 50 RPD (1,000 after a one-time $10) | not viable |
| Gemini free | Gemini only (not open-weight) | 10 RPM, 250-1,500 RPD, free-tier data used for training | not open-weight |
| Local (M1, 8 GB) | up to ~8B at Q4 | unlimited, ~10-15 tok/s, no concurrency | weeks |

Three corrections to an earlier, wrong reading of this landscape, recorded because each is an easy mistake to repeat:

1. **HuggingFace's own inference free tier is $0.10 per month.** It verifies that code works. It is not a way to run anything. The models are free; HF's hosting of them is not.
2. **Groq's advertised "14,400 requests/day" is for prompt-guard classifiers**, tiny safety models, not chat models. Every usable chat model is capped at **1,000 RPD**. The Llama chat models are not on the free plan at all.
3. **No free host currently serves a vision model.** An earlier claim that Groq's Llama 4 supports vision was stale: Llama 4 is not in Groq's catalog.

### A.3 Resolved: `gpt-oss-120b`, alternating Groq and Cerebras

`gpt-oss-120b` is the only model both hosts serve free. Alternating between them roughly doubles throughput to about 2,800 calls per day **with identical weights**, so this buys speed without a cross-model confound, which is normally exactly what would invalidate a comparison. It is also the largest free option, which matters most for the Hinglish generation RQ1 depends on.

The existing code already routes these ids correctly, and does so **because of the Phase 4 D1 fix**:

```
groq/openai/gpt-oss-120b   ->  provider=groq      bare=gpt-oss-120b
cerebras/gpt-oss-120b      ->  provider=cerebras  bare=gpt-oss-120b
```

Both resolve to the same pricing key while getting distinct provider names for rate limiting, which is precisely what per-host throttling requires. Before D1, `model_id` kept its prefix and the pricing lookup failed for every non-bare-OpenAI id, so this strategy would not have worked at all.

LiteLLM supports `groq`, `cerebras`, `huggingface` and `ollama` natively, so no new dispatch code is needed.

### A.4 Resolved: keep Ohagi's design, spend calendar time instead

M=100, N=5, K=10, 5 trials per cell, unchanged. About 20,000 calls, about 7 days unattended at ~2,800/day.

`sandbox_hld.md:78` is explicit that changing M/N/K "would confound 'language effect' with 'population-size effect' relative to the paper you are extending." Under free inference the alternative to calendar time is a methodological compromise, and calendar time is the cheaper currency: the runs are unattended, and Phase 4 made resume byte-identical to an uninterrupted run (defect D8), which is what makes multi-day campaigns safe rather than merely possible.

```
RQ1   2 languages x 5 trials x 1,000 calls  = 10,000
RQ3   2 meme conditions x 5 trials x 1,000  = 10,000
                                              ------
                                              20,000 calls, ~7 days
```

### A.5 Resolved: RQ3 runs caption-only, and says so

No free host serves vision, so every meme reaches its receiving agent as a caption. The architecture already handles this exactly: `vision_fallback.build_meme_prompt()` substitutes the caption and `Interaction.used_vision_fallback` records every instance, so the limitation lands in the logged data instead of being invisible.

`sandbox_hld.md:286` anticipates this case and warns that the fallback "degrades meme content to text-only and would otherwise make RQ3's meme-vs-text comparison partially moot." That warning is accepted rather than dismissed, and it narrows the claim:

> RQ3 tests whether **meme-sourced stance content** accelerates convergence, not whether **visual** meme content does.

That is a narrower claim, but an honest and defensible one, and `used_vision_fallback` being uniformly `True` is the audit trail that proves the scope of it. The alternative, introducing a second local vision model for meme turns only, would put two models in one run, which `sandbox_hld.md:346` rules out of scope precisely to avoid cross-model confounds.

### A.6 Code changes required before any real run

Small, but every one is mandatory: each of these modules raises rather than defaulting, by design, so an unlisted model fails loudly instead of running wrong.

**`pricing_table.yaml`** gains two providers. `CostTracker._lookup_rate` (`cost_tracker.py:62-69`) raises `ValueError` on a missing entry, so a run against an unpriced model dies after the first successful response.

```yaml
groq:
  gpt-oss-120b:
    text: {prompt_per_1k: 0.0, completion_per_1k: 0.0}
  gpt-oss-20b:
    text: {prompt_per_1k: 0.0, completion_per_1k: 0.0}
  qwen3.8-27b:
    text: {prompt_per_1k: 0.0, completion_per_1k: 0.0}
cerebras:
  gpt-oss-120b:
    text: {prompt_per_1k: 0.0, completion_per_1k: 0.0}
  glm-4.7:
    text: {prompt_per_1k: 0.0, completion_per_1k: 0.0}
```

These zeroes are *genuinely* zero, unlike the placeholder rates the file's header warns about. Say so in a comment, so a later reader does not "correct" them.

**`model_gateway.py:31`** gains the new ids in `TEXT_ONLY`:

```python
TEXT_ONLY = {"gpt-3.5-turbo", "llama-3.1-8b", "llama-3.1-70b",
             "gpt-oss-120b", "gpt-oss-20b", "qwen3.8-27b", "glm-4.7"}
```

`_resolve_vision_support` raises on an unrecognized id at construction time, so this is required to start at all. Add a note that **no free model belongs in `VISION_CAPABLE` today**, and that moving one there without checking the host actually accepts images will produce silent `FileNotFoundError`s from `vision_fallback._load_image` rather than a clean failure.

**New: `MultiHostGateway`** (Tier 1, wrapping `ModelGateway`). The only genuinely new runtime component in either phase.

```python
class MultiHostGateway:
    """Round-robins one model across several hosts.

    Satisfies ModelGateway's surface exactly, so the Orchestrator and
    PromptBuilder need no change: composition, not modification.
    """
    def __init__(self, gateways: list[ModelGateway]): ...
    async def generate(self, prompt_text, image=None, temperature=0.7) -> BackendResponse: ...
    @property
    def supports_vision(self) -> bool: ...   # AND across hosts, never OR
```

Two details worth pinning in tests:
- `supports_vision` must be the **conjunction** across hosts. If any host is text-only, the run must behave text-only throughout, or otherwise-identical agents would receive different stimuli depending on which host happened to serve them, which is a confound introduced purely by infrastructure.
- Every host must serve the **same** model. Guard the constructor against mismatched bare model ids and raise, rather than silently running a cross-model experiment.

**New: `QuotaExhaustedError`.** Today a 429 becomes `TransientGatewayError`, is retried three times with exponential backoff, and then propagates and aborts the run at `failed_partial`. That is already *correct* and resumable, which is why this is an efficiency item rather than a defect. But once the daily cap is hit, every remaining agent in the turn burns three doomed retries plus backoff. At M=100 that is 300 pointless calls and several minutes of sleeping per turn. Distinguish daily-quota exhaustion from transient throttling so the orchestrator stops at the next checkpoint immediately.

### A.7 The validity gate: check Hinglish before spending 20,000 calls

The single largest risk to RQ1 is not in the code. **If `gpt-oss-120b` produces poor or inconsistent Hinglish, RQ1 measures model deficiency rather than a language effect, and no amount of post-hoc analysis repairs that.**

This mirrors the contingency already stated in `sandbox_hld.md:439`, where RQ2's support is conditional on validating the diagnostic tools against a hand-labelled Hinglish sample before full-scale use. The same logic applies one level earlier, to generation itself.

**Proposed gate, to run before the campaign:**

1. Generate roughly 50 responses per language condition with the real prompt template, at the real temperature, using a spread of personas and stances.
2. Hand-check for register: is the Hinglish actually code-mixed, or is it English with occasional Hindi nouns, or Hindi with English loanwords? Either degenerate case breaks the comparison.
3. Compute the code-mix index (Section B, Q5.5) on those samples and confirm the distributions for `english` and `hinglish` genuinely separate.
4. Confirm the `STANCE:` format holds across conditions. A higher parse-failure rate in one language is itself a confound, because `failed_logged_null` rows are dropped from analysis and would be dropped unevenly.

Record the result either way. A negative result here is a finding that changes the model choice, and it costs about 100 calls to obtain instead of 20,000.

**Also report, from Phase 4's measurement:** the three language directives differ by a fixed 42 characters, which is 2.87% of a realistic full-size prompt. Report that alongside the `prompt_token_count` diagnostic (Q5.9) rather than leaving it implicit. It is a known, quantified, non-zero asymmetry between RQ1's arms, and stating it is far better than having a reviewer find it.

---

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

**Resolved:** implement a lexicon-based tagger (an English wordlist plus a romanized-Hindi list, with ambiguous tokens tagged `unknown` and excluded from `n` rather than guessed), and compare it against `lingua` on the Section A.7 validation sample. Pick whichever separates the two conditions more cleanly on hand-labelled data.

Whichever is chosen, **report its measured accuracy on that hand-labelled sample in the paper.** Presenting CMI as an exact quantity when it rests on approximate token-level LID would be the sort of overclaim that is easy to avoid and awkward to defend.

### Q5.6: sentiment

Fix B (`sandbox_hld.md:409`) explicitly forbids VADER, which is English-only and would produce meaningless scores on Hinglish, systematically differing between RQ1's arms in exactly the way that manufactures a false result.

**Resolved:** `cardiffnlp/twitter-xlm-roberta-base-sentiment`, run locally. Multilingual, trained on social text, and handles romanized input more gracefully than most alternatives. Score all rows in one batched pass; per-row inference on 20,000 rows is needlessly slow.

### Q5.7: embeddings and clustering

**Resolved:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (fast, ~470 MB, good multilingual coverage). LaBSE is the stronger alternative at roughly 1.8 GB, which is worth weighing on an 8 GB machine.

Cluster with KMeans over a small range of k, selecting by silhouette score, and record the chosen k per run. Ohagi's method is cluster-based, so the clusters are a reported artifact, not an implementation detail.

### Q5.8: the coherence judge, and self-evaluation bias

**Resolved: judge with `qwen3.8-27b` on Groq, not with `gpt-oss-120b`.**

Two independent reasons, either sufficient:

1. **Validity.** A model grading its own output rates its own style as more coherent. If RQ2's claim is "reasoning quality differs by language" and the judge shares the generator's blind spots, particularly in Hinglish where both may be weakest, the bias lands directly on the headline result. A different model family breaks that coupling. This still satisfies Fix B's requirement of a *fixed* judge, which is about consistency across the corpus, not about matching the generator.
2. **Throughput.** Groq's quotas are per-model. Judging with a different model draws on a **separate** 1,000/day budget rather than competing with simulation runs for the same one.

Judging happens only on `content_type == "generated_text"` rows (Q5.3), so the judged volume is smaller than the run's total, and smaller again for RQ3 runs where a fraction of turns are memes.

**Route judge calls through `ModelGateway`.** §3.13 leaves this open and recommends it; the recommendation should be taken, because the judge needs exactly the same retry, rate-limit and cost protections as the simulation, and reimplementing them for an offline script is how the two drift apart.

### Q5.9: the Fix A diagnostic as a standard report

Not in §3.13, added because Phase 4's D7 work made it load-bearing.

`diagnostics.py` reports **mean `prompt_token_count` per language condition, with a significance test.** This turns "memory was capped at 5 turns" from an assertion into a measured result:

> Mean prompt length was X tokens for English and Y for Hinglish (difference not significant, p = Z), so there is no truncation asymmetry between conditions.

Fix A counts memory in *turns* rather than tokens precisely to prevent one language silently receiving less context. This is the evidence that it worked, computed from the run's own logged data at zero extra cost. Report the 42-character directive difference (A.7) alongside it.

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

### C.3 `--dry-run` earns its keep under free tiers

§6.1 specifies config validation plus a cost estimate. Under free inference, **the binding constraint is not dollars but the daily request cap**, so the estimate should report calendar time:

```
$ python -m sandbox run --config configs/rq1_english.yaml --dry-run

Config valid.
  100 agents x 10 turns x 1 trial     = 1,000 calls
  hosts: groq (1,000/day), cerebras (~1,800/day)
  estimated: 1 day, $0.00

  WARNING: this run alone consumes Groq's entire daily free quota.
```

Catching that before launching beats discovering it four turns in.

### C.4 A campaign runner

Follows directly from the scope decision (A.4). Twenty runs over about seven days needs something that runs until quota is exhausted, checkpoints, waits, and resumes the next day.

Phase 4 made this safe rather than merely possible: defect D8's fix means **a resumed run is byte-identical to an uninterrupted one**, so a campaign stopped and restarted daily produces exactly the data a single continuous run would have. Without that fix, every daily boundary would have introduced a silent discontinuity in the neighbour-sampling stream.

Proposed: `python -m sandbox campaign --config-dir configs/rq1/ [--daily-cap N]`, iterating configs, running each until `QuotaExhaustedError` (A.6), and exiting cleanly with a resume hint.

### C.5 Fix the CWD-relative paths

`Path("data/memes")`, `Path("data/personas")`, `Path("runs")` and `Path("pricing_table.yaml")` are resolved against the process working directory in `config_loader.py`, `agent_manager.py`, `meme_pool_manager.py`, `cost_tracker.py` and `checkpoint_manager.py`. The system therefore only works when invoked from the repo root.

That was tolerable while the only entry point was pytest. A CLI makes it untenable: `python -m sandbox run` from any other directory fails with a confusing missing-asset error. Resolve paths against a project root discovered from the config file's location, or accept an explicit `--data-dir`.

### C.6 Optional, explicitly non-load-bearing

- `runs_manifest.sqlite` (§5.4): a derived, rebuildable index, never the source of truth. Safe to delete and regenerate at any time.
- `status_server.py` (§6.3): a read-only JSON endpoint for monitoring a long run from a second terminal. Genuinely useful given multi-day campaigns, but strictly optional.

---

## Section D: Still open

Carried forward from `phase4.md`, not closed by this plan.

- **API credentials.** Free accounts for Groq and Cerebras, no card required for either. Never commit keys; read them from the environment. The first real call the project ever makes will be through these.
- **A real persona pool.** Only `test_pool.jsonl` (4 entries) and `pool_20.jsonl` (20, test-only) exist. With M=100 and a 20-entry pool, each persona repeats across 5 agents, which is a real design choice about population diversity rather than a fixture detail, and should be made deliberately.
- **Meme `stance_label` rescaling.** `MemeContent.stance_label` is a bare float with no link to `stance_scale`, and `MemePoolManager` never receives the run, so it structurally cannot validate. The test fixture's 1-7 range is coincidental. Any real meme dataset (typically 0-1, -1..+1, or 1-5) silently produces garbage until a rescale step exists.
- **`pricing_table.yaml`'s legacy entries.** The openai/anthropic/google rates remain unvalidated placeholders, as the file's own header says. They are now unused but still loadable, so either validate or remove them rather than leaving a trap.

---

## Sequencing

1. **Section A code changes** (pricing table, `TEXT_ONLY`, `MultiHostGateway`, `QuotaExhaustedError`). Small, and blocks every real call.
2. **The A.7 Hinglish validation gate.** About 100 calls. Do this before anything expensive; a negative result changes the model choice.
3. **Phase 5's dependency-free modules** (loader, regression, bimodality, diagnostics). Unblocked now, and they make the first real run's output immediately interpretable.
4. **Phase 6's CLI and campaign runner.** Needed to actually launch a multi-day campaign comfortably.
5. **The real campaign.** About 7 days unattended.
6. **Phase 5's model-backed modules** (sentiment, embeddings, coherence) against real data rather than fixtures.

Steps 3 and 4 are independent and can proceed in either order.
