#!/usr/bin/env python3
"""Flatten a campaign's runs into analysis-ready CSVs.

    python export_csv.py outputs/chorus_run1
    python export_csv.py outputs/chorus_run1 --out-dir analysis --keep-duplicates

Produces three files:

  interactions.csv  one row per agent-turn, with run-level config joined on.
                    The main dataset.
  agents.csv        one row per agent per run: final stance, total movement,
                    and how far it drifted from where it started.
  runs.csv          one row per run: the config, plus summary statistics you
                    would otherwise recompute every time.

**Deduplication.** A run interrupted mid-turn can log the same
(turn, speaker_agent_id) twice: the orchestrator writes interactions one at a
time and checkpoints only after the whole turn, so a crash part-way through
leaves rows on disk with no checkpoint, and the resumed pass re-runs that turn
and appends again. The later row is the authoritative one, verified against
agents_final.jsonl, whose stance_history references the post-resume
interaction_id in every observed case. Earlier copies are orphans: their
effect on agent state was lost with the un-checkpointed memory.

Duplicates are dropped by default and reported. Pass --keep-duplicates to
inspect them, with an `is_duplicate` column marking the ones that would be
removed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Joined onto every interaction row so the CSV is groupable without a second file.
RUN_FIELDS = (
    "run_id", "topic", "language_condition", "trial_number", "seed", "alpha",
    "M", "N", "K", "rq_target", "model_backend_id", "stance_low_label",
    "stance_high_label", "status", "git_commit_hash",
)


def load_run(run_dir: Path) -> pd.DataFrame | None:
    """One run's interactions with its config joined on, or None if unusable."""
    cfg_path, rows_path = run_dir / "run_config.json", run_dir / "interactions.jsonl"
    if not cfg_path.exists() or not rows_path.exists():
        return None

    cfg = json.loads(cfg_path.read_text())
    df = pd.read_json(rows_path, lines=True)
    if df.empty:
        return None

    for field in RUN_FIELDS:
        df[field] = cfg.get(field)

    # topic_key is the grouping dimension: run_ids look like politics_english_t1.
    df["topic_key"] = run_dir.name.split("_")[0]
    df["meme_injection_enabled"] = cfg.get("meme_injection", {}).get("enabled", False)

    # Row order within a run is write order, which is what makes "keep last"
    # mean "keep the post-resume copy".
    df["row_index"] = range(len(df))
    return df


def mark_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Flags all but the final write of each (run_id, turn, speaker_agent_id)."""
    keys = ["run_id", "turn", "speaker_agent_id"]
    df = df.sort_values(["run_id", "row_index"])
    df["is_duplicate"] = df.duplicated(subset=keys, keep="last")
    return df


def derive(df: pd.DataFrame) -> pd.DataFrame:
    """Columns every analysis recomputes, added once here."""
    df["stance_shift"] = df.stance_after - df.stance_before
    df["abs_stance_shift"] = df.stance_shift.abs()
    df["stance_moved"] = df.stance_shift != 0
    df["is_valid_for_analysis"] = df.api_call_status != "failed_logged_null"
    df["n_neighbors"] = df.neighbor_agent_ids.map(len)
    # Lists do not survive CSV round-trips as lists; make it explicit and
    # re-splittable rather than letting str() of a Python list leak through.
    df["neighbor_agent_ids"] = df.neighbor_agent_ids.map(lambda v: "|".join(v))
    df["reason_char_count"] = df.reason_text.fillna("").str.len()
    df["reason_word_count"] = df.reason_text.fillna("").str.split().map(len)
    return df


def agents_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per agent per run: trajectory summary."""
    valid = df[df.is_valid_for_analysis]
    first = valid.sort_values("turn").groupby(["run_id", "speaker_agent_id"]).first()
    last = valid.sort_values("turn").groupby(["run_id", "speaker_agent_id"]).last()
    agg = valid.groupby(["run_id", "speaker_agent_id"]).agg(
        topic_key=("topic_key", "first"),
        language_condition=("language_condition", "first"),
        turns_completed=("turn", "nunique"),
        total_abs_movement=("abs_stance_shift", "sum"),
        mean_abs_shift=("abs_stance_shift", "mean"),
        turns_moved=("stance_moved", "sum"),
        mean_prompt_tokens=("prompt_token_count", "mean"),
        mean_completion_tokens=("completion_token_count", "mean"),
    )
    agg["initial_stance"] = first.stance_before
    agg["final_stance"] = last.stance_after
    # Net drift, as distinct from total movement: an agent can move a lot and
    # end where it began.
    agg["net_drift"] = agg.final_stance - agg.initial_stance
    return agg.reset_index()


def runs_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per run: config plus the summaries worth precomputing."""
    valid = df[df.is_valid_for_analysis]
    final_turn = valid[valid.turn == valid.groupby("run_id").turn.transform("max")]

    runs = df.groupby("run_id").agg(
        topic_key=("topic_key", "first"),
        topic=("topic", "first"),
        language_condition=("language_condition", "first"),
        trial_number=("trial_number", "first"),
        seed=("seed", "first"),
        alpha=("alpha", "first"),
        M=("M", "first"), N=("N", "first"), K=("K", "first"),
        model_backend_id=("model_backend_id", "first"),
        status=("status", "first"),
        n_interactions=("interaction_id", "count"),
        n_failed=("is_valid_for_analysis", lambda s: (~s).sum()),
        mean_latency_ms=("latency_ms", "mean"),
        mean_prompt_tokens=("prompt_token_count", "mean"),
    )
    runs["parse_failure_rate"] = runs.n_failed / runs.n_interactions
    runs["mean_abs_shift"] = valid.groupby("run_id").abs_stance_shift.mean()
    # Spread of final stances: the headline polarization measure. Higher means
    # agents ended further apart.
    runs["final_stance_std"] = final_turn.groupby("run_id").stance_after.std()
    runs["final_stance_mean"] = final_turn.groupby("run_id").stance_after.mean()
    runs["initial_stance_std"] = (
        valid[valid.turn == 1].groupby("run_id").stance_before.std()
    )
    runs["polarization_change"] = runs.final_stance_std - runs.initial_stance_std
    return runs.reset_index()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("campaign_dir", type=Path, help="directory holding one subdir per run")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="where to write the CSVs (default: alongside the runs)")
    ap.add_argument("--keep-duplicates", action="store_true",
                    help="keep re-logged rows, marked with is_duplicate")
    ap.add_argument("--exclude", nargs="*", default=["colab_smoke"],
                    help="run_ids to skip (default: colab_smoke)")
    args = ap.parse_args()

    if not args.campaign_dir.is_dir():
        print(f"not a directory: {args.campaign_dir}", file=sys.stderr)
        return 1

    frames, skipped = [], []
    for d in sorted(p for p in args.campaign_dir.iterdir() if p.is_dir()):
        if d.name in args.exclude:
            skipped.append(f"{d.name} (excluded)")
            continue
        got = load_run(d)
        if got is None:
            skipped.append(f"{d.name} (no usable output)")
        else:
            frames.append(got)

    if not frames:
        print("no runs loaded", file=sys.stderr)
        return 1

    df = derive(mark_duplicates(pd.concat(frames, ignore_index=True)))

    n_dupes = int(df.is_duplicate.sum())
    if n_dupes and not args.keep_duplicates:
        affected = sorted(df[df.is_duplicate].run_id.unique())
        df = df[~df.is_duplicate].drop(columns=["is_duplicate"])
        print(f"dropped {n_dupes} re-logged rows from interrupted runs: {', '.join(affected)}")
        print("  (a run that crashed mid-turn re-logs that turn on resume; the later")
        print("   write is authoritative, per agents_final.jsonl's stance_history)")
    elif n_dupes:
        print(f"kept {n_dupes} re-logged rows, marked in is_duplicate")

    out = args.out_dir or args.campaign_dir
    out.mkdir(parents=True, exist_ok=True)
    df = df.drop(columns=["row_index"]).sort_values(["run_id", "turn", "speaker_agent_id"])

    for name, table in (("interactions", df),
                        ("agents", agents_table(df)),
                        ("runs", runs_table(df))):
        path = out / f"{name}.csv"
        table.to_csv(path, index=False)
        print(f"  {path}  {len(table):>5} rows x {len(table.columns):>2} cols "
              f"({path.stat().st_size/1e6:.1f} MB)")

    for s in skipped:
        print(f"  skipped {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
