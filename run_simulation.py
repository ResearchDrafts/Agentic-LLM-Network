#!/usr/bin/env python3
"""Launch one simulation run from a config file.

    python run_simulation.py configs/smoke_test.yaml
    python run_simulation.py configs/smoke_test.yaml --dry-run

A stopgap until Phase 6 builds the real CLI (`python -m sandbox run`) and
`sandbox/api.py`. Deliberately thin: it wires the four objects the
orchestrator needs and gets out of the way, so there is no behaviour here
that the eventual CLI would have to reimplement differently.

Re-running the same config resumes from the last checkpoint rather than
starting over, and a resumed run produces output byte-identical to an
uninterrupted one. To start fresh, delete runs/{run_id}/ or change run_id.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from sandbox.config_loader import ConfigLoadError, load_run_config
from sandbox.cost_tracker import CostTracker
from sandbox.model_gateway import ModelGateway
from sandbox.models import ExperimentRun
from sandbox.rate_limiter import RateLimiter
from sandbox.simulation_orchestrator import build_orchestrator


def describe(run: ExperimentRun, runs_dir: Path) -> None:
    """What this run will do, before it does it."""
    generated = run.M * run.K
    print(f"  run_id        {run.run_id}")
    print(f"  topic         {run.topic}")
    print(f"  population    M={run.M}  N={run.N}  K={run.K}  alpha={run.alpha}")
    print(f"  language      {run.language_condition}")
    print(f"  model         {run.model_backend_id}")
    print(f"  api_base      {run.api_base or '(hosted provider)'}")
    print(f"  memes         {'enabled' if run.meme_injection.enabled else 'disabled'}")
    print(f"  seed          {run.seed}")
    print(f"  output        {runs_dir / run.run_id}/")
    print()
    print(f"  at most {generated} LLM calls (M x K), fewer if memes replace turns")

    if run.api_base is None and run.model_backend_id.startswith("hosted_vllm/"):
        print()
        print("  WARNING: model_backend_id is hosted_vllm/* but api_base is unset.")
        print("           litellm will route on the prefix and never reach your server.")

    checkpoint = runs_dir / run.run_id / "checkpoints" / "latest.json"
    if checkpoint.exists():
        print()
        print("  NOTE: a checkpoint exists, so this RESUMES rather than starting over.")
        print(f"        Delete {runs_dir / run.run_id}/ or change run_id to start fresh.")


def check_server(run: ExperimentRun) -> str | None:
    """Confirm the endpoint is up before dispatching anything.

    Without this, a server that is not running costs a long wait for nothing:
    the connection error is classed as transient, so every agent retries three
    times with exponential backoff, concurrently, and the real cause ends up
    buried in litellm's output. Forgetting to start the server is the most
    likely first failure, so it should be the fastest and clearest one.

    Returns an error string, or None if reachable.
    """
    if not run.api_base:
        return None  # hosted provider: nothing local to check

    import urllib.error
    import urllib.request

    # Ollama and vLLM expose different listing endpoints, and litellm's
    # ollama provider talks to the native API rather than the /v1 shim.
    is_ollama = run.model_backend_id.startswith("ollama/")
    base = run.api_base.rstrip("/")
    url = f"{base}/api/tags" if is_ollama else f"{base}/models"
    wanted = run.model_backend_id.split("/", 1)[-1]

    if is_ollama:
        start_hint = f"    ollama serve        # then: ollama pull {wanted}"
    else:
        start_hint = f"    vllm serve {wanted} --dtype half --port 8000"

    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            body = resp.read().decode("utf-8", "replace")
    except urllib.error.URLError as e:
        return (
            f"cannot reach {url}\n"
            f"  {e.reason}\n\n"
            f"  Start the server first:\n"
            f"{start_hint}\n"
            f"  then confirm: curl {url}"
        )
    except Exception as e:  # noqa: BLE001 - any failure here is a failed check
        return f"cannot reach {url}: {type(e).__name__}: {e}"

    if wanted not in body:
        pulled = "  Pull it with: ollama pull " + wanted if is_ollama else ""
        return (
            f"{url} is up, but does not report serving {wanted!r}.\n"
            f"  It returned: {body[:300]}\n\n"
            f"  model_backend_id must match the served name exactly, after the\n"
            f"  provider prefix, including any :tag suffix.\n{pulled}"
        )
    return None


async def launch(run: ExperimentRun, runs_dir: Path) -> None:
    tracker = CostTracker(run)
    gateway = ModelGateway(
        run.model_backend_id,
        RateLimiter(run.rate_limits),
        tracker,
        api_base=run.api_base,
    )
    await build_orchestrator(
        run, gateway, cost_tracker=tracker, runs_dir=runs_dir
    ).run()

    run_dir = runs_dir / run.run_id
    rows = (run_dir / "interactions.jsonl").read_text().splitlines()
    print(f"\ndone. {len(rows)} interactions -> {run_dir}/")
    print("  interactions.jsonl   the dataset every analysis reads")
    print("  agents_final.jsonl   per-agent state, including stance_history")
    print("  run_config.json      the exact config used, with git_commit_hash")
    print("  checkpoints/         resume point")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("config", type=Path, help="path to a run config YAML")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="validate the config and print the plan without calling any model",
    )
    parser.add_argument(
        "--runs-dir", type=Path, default=Path("runs"),
        help="where run output goes (default: runs/). Point this at persistent "
             "storage when running in a notebook, whose filesystem is ephemeral.",
    )
    args = parser.parse_args()

    try:
        run = load_run_config(args.config)
    except ConfigLoadError as e:
        # Every config problem surfaces here, with all errors at once rather
        # than one per attempt.
        print(f"config invalid: {e}", file=sys.stderr)
        return 1

    print(f"\n{args.config}")
    describe(run, args.runs_dir)

    if args.dry_run:
        print("\ndry run, nothing dispatched.")
        return 0

    problem = check_server(run)
    if problem is not None:
        print(f"\n{problem}", file=sys.stderr)
        return 1

    print()
    try:
        asyncio.run(launch(run, args.runs_dir))
    except KeyboardInterrupt:
        print("\ninterrupted. the last completed turn is checkpointed; "
              "re-run the same config to resume.", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"\nrun failed: {type(e).__name__}: {e}", file=sys.stderr)
        print("the last completed turn is checkpointed; re-run to resume.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
