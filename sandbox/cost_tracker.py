"""Feature 3: Pricing Table & Cost Tracking.

Tracks $ cost per API call using a per-(provider, model_id, modality)
pricing table, and enforces a run's cost ceiling. Never falls back to an
estimated/default rate on a missing pricing entry -- fails loudly instead,
consistent with this system's "fail before spending budget" philosophy.
"""

from __future__ import annotations

import threading
from pathlib import Path

import yaml

from sandbox.models import ExperimentRun


class CostCeilingExceeded(Exception):
    def __init__(self, current: float, ceiling: float):
        self.current = current
        self.ceiling = ceiling
        super().__init__(f"cost ${current:.2f} reached/exceeded ceiling ${ceiling:.2f}")


class CostTracker:
    def __init__(self, run: ExperimentRun, pricing_table_path: Path = Path("pricing_table.yaml")):
        self._run = run
        if not pricing_table_path.exists():
            raise FileNotFoundError(f"pricing table not found: {pricing_table_path}")
        self._pricing = yaml.safe_load(pricing_table_path.read_text())
        self._total_usd = 0.0
        # A plain threading.Lock, not asyncio.Lock: record() is synchronous
        # (per Feature 3's Interface/Contract) and contains no `await`, so
        # under asyncio's cooperative scheduling a sync method with no
        # internal await point is already atomic with respect to other
        # coroutines -- this lock is defense-in-depth against a future
        # refactor that adds an await inside record(), not a requirement
        # of the current control flow.
        self._lock = threading.Lock()

    def record(
        self,
        provider: str,
        model_id: str,
        modality: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        """Adds the computed cost to total_usd BEFORE checking the ceiling,
        and raises CostCeilingExceeded AFTER that addition -- the call that
        pushes the total over the ceiling is always counted. This is a
        deliberate choice (full_design_doc.md §7.3), not a bug."""
        rate = self._lookup_rate(provider, model_id, modality)
        cost = (prompt_tokens / 1000) * rate["prompt_per_1k"] + (completion_tokens / 1000) * rate["completion_per_1k"]
        with self._lock:
            self._total_usd += cost
            total = self._total_usd
        if self._run.max_cost_usd is not None and total >= self._run.max_cost_usd:
            raise CostCeilingExceeded(total, self._run.max_cost_usd)

    def _lookup_rate(self, provider: str, model_id: str, modality: str) -> dict:
        try:
            return self._pricing[provider][model_id][modality]
        except KeyError as e:
            raise ValueError(
                f"no pricing entry for {provider}/{model_id}/{modality} -- "
                f"update pricing_table.yaml before running"
            ) from e

    @property
    def total_usd(self) -> float:
        return self._total_usd
