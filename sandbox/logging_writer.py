"""Append-only persistence (full_design_doc.md Sec 3.8).

One run = one directory, containing run_config.json, interactions.jsonl,
and agents_final.jsonl. Disk-write failures (OSError, e.g. disk full)
propagate uncaught -- treated as fatal, since silently losing an
Interaction record would corrupt the run's data integrity in a way no
downstream analysis could detect.

run_dir creation is NOT this module's responsibility: per the design doc's
own sample code, LoggingWriter never calls run_dir.mkdir() (unlike
checkpoint_manager.py, which does create its own checkpoints/ directory).
The Orchestrator is expected to create run_dir before constructing a
LoggingWriter; calling any write method against a run_dir that doesn't
exist yet propagates FileNotFoundError uncaught, exactly like any other
disk-write failure.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from sandbox.models import Agent, ExperimentRun, Interaction


class LoggingWriter:
    def __init__(self, run_dir: Path):
        self._run_dir = run_dir
        self._interactions_path = run_dir / "interactions.jsonl"
        self._lock = asyncio.Lock()  # serializes concurrent appends from dispatch tasks

    async def write_interaction(self, interaction: Interaction) -> None:
        """
        Appends one JSON line. Uses an asyncio.Lock rather than relying on
        OS-level atomic append, because Python's async file writes are not
        guaranteed atomic across concurrent tasks writing to the same file
        handle within one process (this is a single-process design -- no
        cross-process write contention to handle).
        """
        line = interaction.model_dump_json() + "\n"
        async with self._lock:
            with open(self._interactions_path, "a", encoding="utf-8") as f:
                f.write(line)

    def write_run_config(self, run: ExperimentRun) -> None:
        """Called once at run start, before any concurrent activity -- no
        locking needed."""
        path = self._run_dir / "run_config.json"
        path.write_text(run.model_dump_json(indent=2))

    def write_agents_final(self, agents: list[Agent]) -> None:
        """Called once at run completion, after all turns finish and no
        further concurrent writes are possible -- no locking needed.
        Overwrites (open mode "w"), not appends: a second call with a
        different agent list leaves only that second list's contents."""
        path = self._run_dir / "agents_final.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for agent in agents:
                f.write(agent.model_dump_json() + "\n")
