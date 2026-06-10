from __future__ import annotations

import argparse

from deepswe_rank_stability.analysis.resampling import bootstrap_rank_stability, filter_trials
from deepswe_rank_stability.data.deepswe import DEFAULT_CACHE_DIR, load_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DeepSWE rank-stability side experiment.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    summarize = subparsers.add_parser("summarize", help="Print artifact and trial summary.")
    summarize.add_argument("--refresh", action="store_true", help="Refresh cached DeepSWE artifacts.")

    bootstrap = subparsers.add_parser("bootstrap", help="Run task-bootstrap rank stability.")
    bootstrap.add_argument("--refresh", action="store_true", help="Refresh cached DeepSWE artifacts.")
    bootstrap.add_argument("--draws", type=int, default=2_000)
    bootstrap.add_argument("--seed", type=int, default=0)
    bootstrap.add_argument("--source", default="deep-swe")
    bootstrap.add_argument("--eval-scope", default="full")
    bootstrap.add_argument("--include-excluded", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    dataset = load_dataset(cache_dir=DEFAULT_CACHE_DIR, refresh=args.refresh)

    if args.command == "summarize":
        print(f"trials: {len(dataset.trials):,}")
        print(f"tasks: {len(dataset.tasks):,}")
        print(f"release: {dataset.release.get('release_id', 'unknown')}")
        print()
        print(
            dataset.trials.groupby(["source", "eval_scope", "included_in_score"], dropna=False)
            .size()
            .reset_index(name="n")
            .sort_values(["source", "eval_scope", "included_in_score"])
            .to_string(index=False)
        )
        return

    if args.command == "bootstrap":
        trials = filter_trials(
            dataset.trials,
            source=args.source,
            eval_scope=args.eval_scope,
            included_in_score=None if args.include_excluded else True,
        )
        result = bootstrap_rank_stability(trials, draws=args.draws, seed=args.seed)
        columns = [
            "model_key",
            "observed_rank",
            "observed_score",
            "rank_mean",
            "top1_probability",
            "top3_probability",
            "score_p05",
            "score_p95",
        ]
        print(result.leaderboard[columns].to_string(index=False))
        return

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
