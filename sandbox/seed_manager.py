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

    # Purpose offsets for turn_rng, kept distinct from the three long-lived
    # stream offsets above so no derived stream can collide with them.
    _TURN_PURPOSES = {"neighbor": 1_000_003, "meme": 2_000_003}

    def turn_rng(self, purpose: str, turn: int) -> random.Random:
        """A fresh stream for one (purpose, turn), derived from the base seed.

        The long-lived streams above cannot survive a resume. Their state
        advances as turns consume them, but CheckpointState stores only agent
        state, so a resumed run rebuilds them at their *initial* position and
        replays earlier turns' draws: resuming at turn 2 gives turn 2 the
        numbers turn 1 already used. The run still completes and is still
        internally deterministic, which is precisely why this was invisible
        until a test compared a resumed run against an uninterrupted one
        (phase4.md D8).

        Deriving per turn makes each turn's draws a pure function of
        (seed, purpose, turn), so a resumed turn is identical to the same turn
        in a run that never stopped, and no state has to be serialized.

        Fix M is preserved: two runs sharing a seed still draw identically,
        which is what holds neighbour sampling and meme injection constant
        while language varies.
        """
        if purpose not in self._TURN_PURPOSES:
            raise ValueError(f"unknown rng purpose: {purpose!r}")
        if turn < 1:
            raise ValueError(f"turn must be >= 1, got {turn}")
        return random.Random(self._base_seed + self._TURN_PURPOSES[purpose] * turn)
