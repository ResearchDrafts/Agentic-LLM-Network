"""Feature 4: Rate Limiting.

Per-provider token-bucket throttling, enforced across all concurrent
dispatch tasks in a run. An unconfigured provider (e.g. a `local` backend
with no entry in rate_limits) is treated as having no limit, not an error.
"""

from __future__ import annotations

import asyncio
import time


class RateLimiter:
    """One instance shared across all concurrent dispatch tasks for a run."""

    def __init__(self, limits: dict[str, int]):
        self._limits = limits
        self._last_request_time: dict[str, float] = {p: 0.0 for p in limits}
        # A limit of 0 is treated as "no limit" (not "no requests allowed"),
        # per Feature 4's edge-case guard -- a 0 requests/minute limit would
        # make the provider entirely unusable and is almost certainly a
        # config mistake, not an intended setting.
        self._min_interval = {p: (60.0 / n if n > 0 else 0.0) for p, n in limits.items()}
        self._locks = {p: asyncio.Lock() for p in limits}

    async def acquire(self, provider: str) -> None:
        if provider not in self._limits:
            return
        async with self._locks[provider]:
            elapsed = time.monotonic() - self._last_request_time[provider]
            wait = self._min_interval[provider] - elapsed
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_request_time[provider] = time.monotonic()

    def release(self, provider: str) -> None:
        """No-op by design: a token-bucket-by-interval strategy needs no
        explicit release step. Kept for interface symmetry with the Model
        Gateway's call site (acquire/release pair around every dispatch)."""
        pass
