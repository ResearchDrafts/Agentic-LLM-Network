# Consultation prompt: experimental design and network analysis for Chorus

*Paste everything below to the model you are consulting.*

---

I am running a computational social science study using an LLM multi-agent
simulation, and I need advice on three things: how many runs to collect, which
topics to cover, and how to analyse the output as a social network using graph
theory and network mining. Full context follows so you can answer concretely
rather than generically.

## 1. What the system is

A simulation of opinion dynamics in populations of LLM-backed agents, extending
Ohagi's echo-chamber design. It is built, tested, and has produced its first
real dataset.

**Mechanism.** M agents each hold a persona, a language condition, and a numeric
stance on a topic (1-7 Likert). For each of K turns, every agent is shown a
homophily-weighted sample of N other agents' most recent posts, then updates its
stance and posts a short free-text justification. The homophily strength is a
single parameter, alpha:

```
weight(candidate) = alpha * 1/(1 + |stance_speaker - stance_candidate|)
                  + (1 - alpha) * 1.0
```

alpha = 0 is uniform random neighbour sampling; alpha = 1 is maximally
homophilous. Sampling is weighted-without-replacement (Efraimidis-Spirakis).

**Locked research questions.**

- **RQ1** Does Hindi-English code-mixed (Hinglish) discussion produce different
  polarization rates or intensity than English, under identical echo-chamber
  conditions?
- **RQ2** Do models show different quality or consistency of opinion-updating
  reasoning in Hinglish vs English, and does that instability correlate with
  polarization speed?
- **RQ3** Does injecting memes from a pre-existing labelled dataset into an
  ongoing discussion accelerate convergence to a polarized (bimodal)
  distribution, compared to text-only discussion? **Not yet run.**

## 2. Design guarantees that constrain the analysis

These are deliberate and should be treated as fixed.

- **Frozen-turn ordering.** Neighbour selection and meme assignment for turn t
  complete before any agent's turn-t generation begins. No agent's turn-t output
  can influence another agent's turn-t input. Turns are therefore clean discrete
  time steps, not a continuous asynchronous process.
- **Fixed 5-turn memory window, counted in turns rather than tokens.** This is
  specifically to avoid a language-driven truncation asymmetry: token budgets
  would silently give one language less context than the other.
- **Seed reuse across language arms.** For a given (topic, trial), the English
  and Hinglish runs share a seed, so neighbour sampling and initial stance
  assignment are identical between arms. Only the language directive differs. The
  prompt is otherwise byte-identical across conditions.
- **Failed generations are logged but excluded from agent state.** A parse
  failure produces a record with `api_call_status = failed_logged_null` and does
  not update the agent, so neighbours continue to see that agent's last valid
  post. Zero failures occurred in the data described below.
- **Reproducible by seed.** Per-turn RNG streams derive from (seed, purpose,
  turn), so a run resumed from a checkpoint is byte-identical to one that never
  stopped.

## 3. The data

One row per agent-turn. 8,000 rows from the first campaign. Fields:

| field | meaning |
|---|---|
| `run_id`, `topic_key`, `topic`, `language_condition`, `trial_number`, `seed` | run identity and condition |
| `turn` | 1..K discrete time step |
| `speaker_agent_id` | the agent acting |
| `neighbor_agent_ids` | pipe-separated list of the N agents whose posts it was shown |
| `stance_before`, `stance_after`, `stance_shift`, `abs_stance_shift` | stance on the 1-7 scale |
| `reason_text` | the agent's free-text justification, mean ~60 tokens |
| `content_type` | `generated_text` or `meme` |
| `prompt_token_count`, `completion_token_count`, `latency_ms` | per-call telemetry |
| `api_call_status` | `success`, `retried_success`, or `failed_logged_null` |
| `alpha`, `M`, `N`, `K` | run parameters |

Also produced per run: `agents_final.jsonl` (each agent's full stance history)
and `run_config.json` (exact config plus git commit hash).

**`neighbor_agent_ids` is a directed edge list.** Each row contributes N directed
edges (speaker <- each neighbour it was shown), timestamped by turn. The first
campaign yields **40,000 directed edges** across 8 runs, 5,000 per run, over 100
nodes and 10 time steps per run. The graph is *induced by the sampling
mechanism*, not specified in advance: there is no fixed social graph, the
neighbourhood is redrawn every turn from stance similarity.

## 4. What has actually been run

One pilot campaign, complete and clean:

- **8 runs** = 4 topics x 2 languages x 1 trial
- Topics: politics (immigration), social (social media and teenagers),
  economic (minimum wage), and a deliberately low-stakes **control** (urban cycle
  lanes), included so that polarization on a bland topic would indicate the
  mechanism rather than the topic is driving it
- M=100, N=5, K=10, **alpha=0.5 fixed**, temperature 0.7
- Model: Qwen2.5-7B-Instruct, self-hosted with vLLM on a Colab A100
- 8,000 interactions, **zero parse failures**, no memes
- Roughly 3 GPU-minutes per run at concurrency 32, so about 25 minutes for the
  whole campaign. **Inference is free and runs are cheap.** This matters for
  question 1: compute is not the binding constraint.

**Headline results so far** (single trial, so indicative only):

```
mean |stance shift|      english  hinglish        final-turn stance std   english  hinglish
control                    0.224     0.202        control                   0.902     0.793
economic                   0.175     0.183        economic                  0.863     0.848
politics                   0.120     0.124        politics                  0.756     0.748
social                     0.172     0.163        social                    0.677     0.634
```

Two things stand out. English and Hinglish are very close on every topic. And
**politics moves least while the neutral control moves most**, which is the
opposite of the intuitive prediction.

**One measured asymmetry that needs handling:** mean prompt length is 693 tokens
for English and 809 for Hinglish, a 17% gap. The prompt scaffold is identical by
construction, so this comes from code-mixed `reason_text` tokenizing less
efficiently and then feeding back into memory and neighbour blocks each turn. The
memory window counts turns, not tokens, so the turn-count guarantee holds, but
the information content per turn differs between arms.

## 5. What exists and what does not

Built and tested: config loading, agent population, neighbour sampling, meme pool
management, model gateway, prompt construction, stance parsing, logging,
checkpoint and resume, the per-turn orchestrator, and CSV export.

**Not built:** the entire post-hoc analysis layer. No regression, no bimodality
measure, no code-mix index, no sentiment scoring, no embedding clustering, and
**no graph or network analysis of any kind**.

Relevant history for question 3: the original design named `graph_export.py` and
`network_metrics.py` for structural analysis using "the 12 Impiccichè & Viviani
network metrics", then **explicitly descoped them**, on the reasoning that the
research questions concern convergence speed and shape rather than interaction
graph structure. The descoping note adds that the interaction log retains
everything needed to build them later, since `neighbor_agent_ids` remains a valid
edge list. I am now revisiting that decision, which is why question 3 matters.

## 6. What I am asking

### Question 1: how many runs?

The locked design is 5 trials per cell. With 4 topics and 2 languages that is 40
runs for RQ1/RQ2, plus 40 more for RQ3's meme conditions. I have run 1 trial.

Given that compute is free and a run costs about 3 GPU-minutes:

- Is 5 trials per cell adequate for the claims RQ1 and RQ3 make, or is that
  number inherited from cost-constrained designs that no longer apply here?
- What statistical power do I actually need to detect a language effect of the
  size suggested by the pilot, where the arms differ by roughly 0.02 on mean
  absolute stance shift?
- **alpha is currently fixed at 0.5 and never swept.** Should it be a swept
  variable? The echo-chamber strength is arguably the most theoretically
  important parameter in the whole design, and I have one value of it.
- Does M=100 with N=5 give a neighbourhood-to-population ratio that produces
  meaningful network structure, or should M or N vary too?
- How should trials, topics, alpha values, and languages be allocated if I can
  afford a few hundred runs rather than 80?

### Question 2: which topics and domains?

I currently have politics, social, economic, and one neutral control, each a
single issue phrased as a 1-7 agreement scale with topic-specific anchors.

- Is one issue per domain enough to claim a domain-level effect, or am I
  measuring issue-specific idiosyncrasy and calling it a domain?
- Does the control topic actually function as a control, given the pilot shows it
  producing the *most* movement?
- Which topic properties should be varied deliberately: prior polarization in the
  real world, moral versus empirical framing, cultural specificity to the
  Hindi-English context, the existence of a factual answer?
- For RQ1 specifically, should topics be chosen for their natural code-mixing
  patterns? Some subjects are discussed in Hinglish far more naturally than
  others in Indian discourse, and a topic nobody would ever discuss in Hinglish
  may make the language arm artificial.
- Does the control need to be one topic or a class of topics?

### Question 3: network and graph analysis

This is where I have the least developed plan and where I most want structure.

The data gives, per run, a **temporal directed multigraph**: 100 nodes, 10 time
steps, 500 directed edges per step, where an edge means "the speaker was shown
this neighbour's post this turn". Nodes carry a time-varying stance (1-7) and
free text. Edges are induced by stance-similarity sampling rather than given.

- **Which graph representation is right?** Per-turn snapshots, a cumulative
  weighted graph where edge weight counts co-appearances, a temporal or dynamic
  network, or a bipartite agent-turn structure? What does each let me claim that
  the others do not?
- **Which metrics actually bear on polarization?** Modularity and community
  detection, assortativity by stance, clustering coefficient, path length,
  centrality distributions, network density over time? The original design
  referenced 12 Impiccichè & Viviani metrics. Which of those are appropriate for
  a *sampling-induced* graph rather than a persistent social network?
- **How should community structure relate to stance clusters?** If communities
  detected purely from interaction structure coincide with stance groupings, that
  seems like the core echo-chamber finding. What is the right way to measure that
  correspondence, and what is the right null model given that edges are generated
  from stance similarity by construction, so *some* correspondence is guaranteed?
- **What is the correct null or baseline?** An alpha=0 run gives uniform random
  sampling, which seems the natural comparison. Is that sufficient, or do I need
  configuration models or rewiring nulls as well?
- **Can network measures serve as the polarization metric itself**, rather than
  stance-distribution bimodality? Would a network-structural definition of
  polarization be more defensible, or does it risk measuring the sampling
  mechanism rather than an emergent property?
- **How should the language comparison be made at the network level?** English
  and Hinglish arms share a seed, so with identical stance trajectories they would
  produce identical graphs. Any structural divergence is therefore downstream of
  differing stance updates. Does that make network differences a clean measure of
  the language effect, or a confounded one?
- **Which tools and literature** should I be working from? I am in Python with
  pandas available; networkx, igraph and similar are open to me. Which papers
  should I read on temporal network analysis of opinion dynamics, and on
  echo-chamber quantification specifically?

Please be concrete, flag anywhere my framing is wrong or my proposed measures
would not support the claims I want to make, and say clearly when something I am
asking for is not answerable from this data.
