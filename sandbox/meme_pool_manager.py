"""Feature 6: Meme Pool Loading & Injection Decisioning.

Loads a pre-existing meme dataset and decides, per scheduled speaker per
turn, whether that speaker's post is a meme. Direct port of
full_design_doc.md Sec 3.4.
"""

from __future__ import annotations

import random
from pathlib import Path

from sandbox.models import Agent, MemeContent, MemeInjectionConfig


class MemePoolManager:
    def __init__(self, config: MemeInjectionConfig, rng: random.Random):
        self._config = config
        self._rng = rng
        self._pool: list[MemeContent] = []
        self._by_id: dict[str, MemeContent] = {}
        if config.enabled:
            self._pool = self._load_pool()
            self._by_id = {meme.meme_id: meme for meme in self._pool}

    def _load_pool(self) -> list[MemeContent]:
        path = Path("data/memes") / f"{self._config.meme_pool_id}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"meme pool not found: {path}")
        items = [
            MemeContent.model_validate_json(line)
            for line in path.read_text().splitlines()
            if line.strip()
        ]
        if not items:
            raise ValueError(f"meme pool at {path} is empty")
        return items

    def get_meme(self, meme_id: str) -> MemeContent:
        """Resolves a meme_id back to its MemeContent.

        Needed because Interaction stores only meme_id, so anything rendering
        a past meme post (prompt_builder.py, per phase4.md Q4.7) has to look
        the content back up. Raises KeyError on an unknown id rather than
        returning None: an Interaction carrying a meme_id that is not in the
        pool means the run's logged data and its meme pool disagree, which is
        a data-integrity failure, not a missing-value case.
        """
        return self._by_id[meme_id]

    def resolve_injections_for_turn(
        self,
        scheduled_speakers: list[Agent],
        turn: int,
        rng: random.Random | None = None,
    ) -> dict[str, MemeContent | None]:
        """
        For each scheduled speaker, decides (per Fix L: uniformly, at the
        configured injection_rate, applied independently per agent-turn
        under injection_schedule='random'; or only at the configured
        turn(s) under injection_schedule='fixed_turn') whether that
        speaker's post this turn is a meme. Returns a dict mapping
        agent_id -> MemeContent or None (None = generate text normally).

        Must be called as part of the frozen-snapshot phase of the
        Orchestrator's per-turn sequence, before any dispatch begins.

        `rng` overrides the construction-time stream for this call. The
        Orchestrator passes SeedManager.turn_rng("meme", turn), because the
        long-lived stream cannot survive a resume: its position is not
        checkpointed, so a resumed run would replay an earlier turn's draws
        (phase4.md D8). Omitting it keeps the original behaviour for callers
        that run a single turn or do not resume.
        """
        if not self._config.enabled:
            return {a.agent_id: None for a in scheduled_speakers}

        draw = rng if rng is not None else self._rng
        eligible = self._is_eligible_turn(turn)
        result: dict[str, MemeContent | None] = {}
        for agent in scheduled_speakers:
            if eligible and draw.random() < self._config.injection_rate:
                result[agent.agent_id] = self._sample_meme(draw)
            else:
                result[agent.agent_id] = None
        return result

    def _is_eligible_turn(self, turn: int) -> bool:
        if self._config.injection_schedule == "random":
            return True
        if self._config.injection_schedule == "fixed_turn":
            return turn in self._config.fixed_turns
        raise ValueError(f"unknown injection_schedule: {self._config.injection_schedule}")

    def _sample_meme(self, rng: random.Random | None = None) -> MemeContent:
        return (rng if rng is not None else self._rng).choice(self._pool)
