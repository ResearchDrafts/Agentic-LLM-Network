"""Neighbor selection (full_design_doc.md Sec 3.3).

Ohagi's alpha-sampling mechanism: alpha in [0,1] blends uniform-random
neighbor selection (alpha=0) with maximally homophilous, stance-distance-
weighted selection (alpha=1). ValueError on invalid alpha/N at
construction, and on an insufficient candidate population at call time --
never silently clamped, per the HLD's Critique Pass 3: silently clamping
alpha or N would itself be exactly the kind of implementation-level
confound the design warns against.
"""

from __future__ import annotations

import math
import random
from typing import Protocol

from sandbox.models import Agent


class RecommendationStrategy(Protocol):
    def select_neighbors(
        self,
        speaker: Agent,
        all_agents: list[Agent],
        turn: int,
        rng: random.Random,
    ) -> list[Agent]:
        """Must return exactly N distinct agents, never including speaker itself."""
        ...


class AlphaSampling:
    """Ohagi's mechanism. alpha in [0,1]: 0 = uniform random sampling,
    1 = maximally homophilous (always prefers closest-stance candidates)."""

    def __init__(self, alpha: float, N: int):
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0,1], got {alpha}")
        if N < 1:
            raise ValueError(f"N must be >= 1, got {N}")
        self.alpha = alpha
        self.N = N

    def select_neighbors(
        self,
        speaker: Agent,
        all_agents: list[Agent],
        turn: int,
        rng: random.Random,
    ) -> list[Agent]:
        candidates = [a for a in all_agents if a.agent_id != speaker.agent_id]
        if len(candidates) < self.N:
            raise ValueError(f"population too small: need {self.N} candidates, have {len(candidates)}")

        current_stance = _current_stance(speaker, turn)
        weights = []
        for c in candidates:
            c_stance = _current_stance(c, turn)
            distance = abs(current_stance - c_stance)
            similarity_weight = 1.0 / (1.0 + distance)  # closer stance -> higher weight
            uniform_weight = 1.0  # equal weight for every candidate
            blended = self.alpha * similarity_weight + (1 - self.alpha) * uniform_weight
            weights.append(blended)

        # weighted sample without replacement, using rng (never the global
        # random module -- reproducibility depends on this)
        return _weighted_sample_without_replacement(candidates, weights, self.N, rng)


def _current_stance(agent: Agent, turn: int) -> float:
    """Reads the agent's stance as of the end of turn-1 (the frozen
    snapshot, per Fix F). turn=1 reads initial_stance; turn>1 reads the
    last entry in stance_history.

    Deliberately duplicates Agent.current_stance()'s logic rather than
    calling it (per phase3.md's resolution of this fidelity-vs-consolidation
    question): the design doc's own sample code never routes through
    Agent.current_stance(), so this stays a standalone port rather than a
    silent deviation from the spec.
    """
    if turn == 1 or not agent.stance_history:
        return agent.initial_stance
    return agent.stance_history[-1].stance_value


def _weighted_sample_without_replacement(
    items: list[Agent], weights: list[float], k: int, rng: random.Random
) -> list[Agent]:
    """Efraimidis-Spirakis weighted reservoir sampling -- O(n log k),
    deterministic given rng's state.

    Weights must be strictly positive and finite. The key formula divides by
    w, so w == 0 raises ZeroDivisionError, and w < 0 is worse than an error:
    random() ** negative > 1, and keys sort descending, so a negative-weight
    item is *guaranteed* to be selected first. That is a silent wrong answer
    in the module whose whole contract is "never silently clamp", so it is
    checked rather than assumed. AlphaSampling cannot currently produce a
    non-positive weight (blended is strictly positive for any alpha in
    [0,1]), but a NaN stance would propagate through to NaN weights and make
    the sort order meaningless.
    """
    if len(weights) != len(items):
        raise ValueError(f"weights/items length mismatch: {len(weights)} vs {len(items)}")
    for w in weights:
        if not math.isfinite(w) or w <= 0.0:
            raise ValueError(f"sampling weights must be finite and > 0, got {w}")

    keyed = [(rng.random() ** (1.0 / w), item) for w, item in zip(weights, items, strict=True)]
    keyed.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in keyed[:k]]
