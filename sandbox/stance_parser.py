"""Stance parsing (full_design_doc.md Sec 3.7).

Extracts a structured (stance_value, reason_text) pair from raw model
output. Pure, stateless functions -- no I/O, no retry logic. Retry
orchestration (re-dispatching to the Model Gateway with an amended prompt)
belongs to the Simulation Orchestrator (Tier 3), which calls parse_stance()
up to 3 times per agent-turn; this module has no access to the Gateway and
does not attempt that loop itself.
"""

from __future__ import annotations

import re

_STANCE_PATTERN = re.compile(r"STANCE:\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)


class StanceParseFailure(Exception):
    """Raised when parsing fails, or the parsed value falls outside the
    configured stance_scale. Caught by the Orchestrator, never by callers
    of parse_stance()/clamp_and_validate_scale() directly."""


def parse_stance(raw_text: str) -> tuple[float, str]:
    """
    Expects the model to have followed the prompt's required output format
    (a line "STANCE: <number>" followed by free-text reasoning). Returns
    (stance_value, reason_text). Raises StanceParseFailure if no STANCE
    line is found or the value doesn't parse as a float.

    reason_text is raw_text with every "STANCE: <number>" occurrence
    stripped (re.sub with no count replaces all matches, not just the one
    parse_stance() reads the value from) and surrounding whitespace
    trimmed. When multiple STANCE-like substrings are present, the FIRST
    match (leftmost) is the one whose value is returned.
    """
    match = _STANCE_PATTERN.search(raw_text)
    if match is None:
        raise StanceParseFailure(f"no STANCE line found in output: {raw_text[:200]!r}")
    try:
        value = float(match.group(1))
    except ValueError as e:
        raise StanceParseFailure(f"STANCE value not numeric: {match.group(1)!r}") from e

    reason = _STANCE_PATTERN.sub("", raw_text).strip()
    return value, reason


def clamp_and_validate_scale(value: float, stance_scale: list[float]) -> float:
    """Raises StanceParseFailure if value is outside [min(scale), max(scale)]
    -- an out-of-range value is treated identically to an unparseable one,
    never silently clamped, per Fix I's 'never fabricate a placeholder
    stance' principle."""
    lo, hi = min(stance_scale), max(stance_scale)
    if not (lo <= value <= hi):
        raise StanceParseFailure(f"stance {value} outside configured scale [{lo}, {hi}]")
    return value
