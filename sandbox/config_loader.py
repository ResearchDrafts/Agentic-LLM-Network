"""Feature 1: Configuration Loading & Validation.

Parses and validates a run config into an ExperimentRun before any other
feature is permitted to act. Fails loudly and immediately, before a single
dollar is spent, per sandbox_feature_specs.md's Feature 1 Problem Statement.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import yaml
from pydantic import ValidationError

from sandbox.models import ExperimentRun


class ConfigLoadError(Exception):
    """Raised for any config problem; wraps the underlying cause."""

    def __init__(self, path: Path, errors: list[str]):
        self.path = path
        self.errors = errors
        super().__init__(f"Invalid config at {path}: {'; '.join(errors)}")


def load_run_config(path: Path) -> ExperimentRun:
    """
    Reads a YAML file, validates it against the ExperimentRun schema, and
    resolves referenced assets (meme_pool_id, persona_pool_id) to confirm
    they exist on disk -- without loading their full contents.

    Raises ConfigLoadError on any failure. Never returns a partially valid
    ExperimentRun.
    """
    if not path.exists():
        raise ConfigLoadError(path, [f"file not found: {path}"])

    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        raise ConfigLoadError(path, [f"malformed YAML: {e}"]) from e

    if raw is None:
        raise ConfigLoadError(path, ["config file is empty"])

    # A syntactically valid YAML document need not be a mapping: "- a\n- b",
    # "just_a_string", and "42" all parse cleanly into a list/str/int. Without
    # this guard the raw.pop() below raises TypeError/AttributeError, breaking
    # this function's documented contract that every failure surfaces as a
    # ConfigLoadError.
    if not isinstance(raw, dict):
        raise ConfigLoadError(
            path, [f"config root must be a mapping, got {type(raw).__name__}"]
        )

    # git_commit_hash is populated here, never user-supplied in the YAML
    # (Feature 1 FR7). Resolution to a plain-text repo command is closed
    # under this build's own decision: if the working directory isn't a
    # git repo, or has zero commits, this is treated as a fatal
    # config-load failure -- consistent with every other "fail before
    # spending budget" decision in this system, rather than silently
    # falling back to an empty string that would make git_commit_hash
    # useless for reproducing which code version produced a run.
    raw.pop("git_commit_hash", None)
    try:
        raw["git_commit_hash"] = _resolve_git_commit_hash()
    except ConfigLoadError as e:
        # Re-wrap so the error carries the caller's real config path.
        # _resolve_git_commit_hash() has no idea which file is being loaded
        # and raises with a placeholder path, so letting it propagate
        # unchanged tells the user their config is invalid without telling
        # them which one.
        raise ConfigLoadError(path, e.errors) from e

    # Pydantic errors and filesystem-asset errors are accumulated together,
    # not fail-fast on whichever stage runs first: Feature 1's own
    # Acceptance Criteria require "a config with two independent errors
    # (e.g., bad alpha and missing persona pool)" to produce ONE
    # ConfigLoadError listing both. This deliberately deviates from
    # full_design_doc.md §3.1's sample implementation, which raises
    # immediately on the first Pydantic ValidationError and never reaches
    # the filesystem check at all -- that sample code cannot actually
    # satisfy this feature's own stated acceptance criterion, so the
    # acceptance criterion (the higher-authority contract) is what's
    # implemented here. Asset existence is checked directly against the
    # raw dict so it runs regardless of whether Pydantic validation itself
    # succeeded.
    errors: list[str] = []
    run: ExperimentRun | None = None
    try:
        run = ExperimentRun(**raw)
    except ValidationError as e:
        errors.extend(str(err) for err in e.errors())

    errors.extend(_referenced_asset_errors(raw))

    if errors:
        raise ConfigLoadError(path, errors)

    assert run is not None
    return run


def _resolve_git_commit_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise ConfigLoadError(
            Path("<config>"),
            [f"could not resolve git_commit_hash (not a git repo, or no commits yet): {e}"],
        ) from e
    return result.stdout.strip()


def _referenced_asset_errors(raw: dict) -> list[str]:
    """
    Checks that meme_pool_id (if meme_injection.enabled) resolves to an
    actual dataset file, and that the persona pool referenced by the run
    exists. Returns ALL missing-asset errors found, not just the first.

    Reads directly from the raw (possibly Pydantic-invalid) dict rather
    than a validated ExperimentRun, so this check runs independently of
    whether schema validation itself succeeded -- see the accumulation
    note in load_run_config().
    """
    errors: list[str] = []

    meme_injection = raw.get("meme_injection")
    if isinstance(meme_injection, dict) and meme_injection.get("enabled") is True:
        meme_pool_id = meme_injection.get("meme_pool_id")
        if isinstance(meme_pool_id, str) and meme_pool_id:
            pool_path = Path("data/memes") / f"{meme_pool_id}.jsonl"
            if not pool_path.exists():
                errors.append(f"meme_pool_id '{meme_pool_id}' not found at {pool_path}")

    persona_pool_id = raw.get("persona_pool_id")
    if isinstance(persona_pool_id, str) and persona_pool_id:
        persona_path = Path("data/personas") / f"{persona_pool_id}.jsonl"
        if not persona_path.exists():
            errors.append(f"persona pool '{persona_pool_id}' not found at {persona_path}")

    return errors
