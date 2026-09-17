"""Feature 2: Seed Management.

Provides three independently-seeded random.Random instances (neighbor
sampling, meme injection, persona assignment), deterministically offset
from one top-level seed, per Fix M. Reusing the same top-level seed across
two ExperimentRun configs (e.g. an English-condition run and a Hinglish-
condition run in a combined sweep) reproduces identical meme-injection and
neighbor-sampling draw sequences across both runs.

This feature makes no claim about LLM sampling determinism -- provider
APIs do not guarantee deterministic output even at temperature=0.
"""

from __future__ import annotations

import random


class SeedManager:
    """One instance per run. Each property returns the SAME long-lived
    random.Random instance on every call -- not a fresh one -- since each
    stream advances as consuming features draw from it.

    Note (Question B2, per the plan's Phase 6 checkpoint): neighbor-sampling
    and meme-injection resolution are assumed to happen strictly sequentially
    within a turn (never concurrently with each other), so these
    random.Random instances are never accessed concurrently. If a future
    change ever parallelizes those two resolution steps against each other,
    this assumption breaks and per-call locking would be needed here.
    """

    def __init__(self, seed: int):
        self._base_seed = seed
        self._neighbor_rng = random.Random(seed)
        self._meme_rng = random.Random(seed + 1)
        self._persona_rng = random.Random(seed + 2)

    @property
    def neighbor_sampling_rng(self) -> random.Random:
        return self._neighbor_rng

    @property
    def meme_injection_rng(self) -> random.Random:
        return self._meme_rng

    @property
    def persona_assignment_rng(self) -> random.Random:
        return self._persona_rng
