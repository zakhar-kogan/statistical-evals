from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

DEFAULT_DRAWS = 2_000
RANK_QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)
SCORE_QUANTILES = (0.05, 0.50, 0.95)
DEFAULT_MIN_SLICE_TASKS = 10
DEFAULT_MIN_SLICE_MODELS = 2


@dataclass(frozen=True)
class BootstrapResult:
    leaderboard: pd.DataFrame
    rank_distribution: pd.DataFrame
    pairwise_win_probability: pd.DataFrame
    boot_scores: pd.DataFrame
    boot_ranks: pd.DataFrame


@dataclass(frozen=True)
class DimensionBootstrapResult:
    summaries: pd.DataFrame
    skipped_slices: pd.DataFrame


def filter_trials(
    trials: pd.DataFrame,
    *,
    source: str | None = None,
    eval_scope: str | None = None,
    included_in_score: bool | None = None,
    outcome: str | None = None,
    language: str | None = None,
    repository: str | None = None,
    model_keys: Iterable[str] | None = None,
) -> pd.DataFrame:
    frame = trials.copy()
    if source is not None:
        frame = frame[frame["source"] == source]
    if eval_scope is not None:
        frame = frame[frame["eval_scope"] == eval_scope]
    if included_in_score is not None:
        frame = frame[frame["included_in_score"] == included_in_score]
    if outcome is not None:
        frame = frame[frame["outcome"] == outcome]
    if language is not None and "language" in frame.columns:
        frame = frame[frame["language"] == language]
    if repository is not None and "repository" in frame.columns:
        frame = frame[frame["repository"] == repository]
    if model_keys is not None:
        frame = frame[frame["model_key"].isin(set(model_keys))]
    return frame


def aggregate_task_model_scores(trials: pd.DataFrame) -> pd.DataFrame:
    required = {"task_name", "model_key", "source", "eval_scope", "score_value", "passed"}
    missing = sorted(required - set(trials.columns))
    if missing:
        raise ValueError(f"trials frame is missing required columns: {missing}")

    trial_count = ("trial_name", "count") if "trial_name" in trials.columns else ("score_value", "count")
    return (
        trials.groupby(["task_name", "model_key", "source", "eval_scope"], dropna=False)
        .agg(
            score_value=("score_value", "mean"),
            pass_rate=("passed", "mean"),
            n_trials=trial_count,
        )
        .reset_index()
    )


def score_matrix(aggregated: pd.DataFrame) -> pd.DataFrame:
    matrix = aggregated.pivot_table(
        index="task_name",
        columns="model_key",
        values="score_value",
        aggfunc="mean",
        observed=False,
    )
    return matrix.sort_index().sort_index(axis=1)


def rank_scores(scores: pd.Series) -> pd.Series:
    return scores.rank(method="min", ascending=False, na_option="bottom").astype("Int64")


def observed_leaderboard(matrix: pd.DataFrame) -> pd.DataFrame:
    scores = matrix.mean(axis=0, skipna=True)
    ranks = rank_scores(scores)
    return (
        pd.DataFrame({"model_key": scores.index, "observed_score": scores.values, "observed_rank": ranks.values})
        .sort_values(["observed_rank", "observed_score", "model_key"], ascending=[True, False, True])
        .reset_index(drop=True)
    )


def bootstrap_rank_stability(
    trials: pd.DataFrame,
    *,
    draws: int = DEFAULT_DRAWS,
    seed: int = 0,
) -> BootstrapResult:
    if draws <= 0:
        raise ValueError("draws must be positive")

    matrix = score_matrix(aggregate_task_model_scores(trials))
    if matrix.empty:
        raise ValueError("no task-model scores are available after filtering")

    values = matrix.to_numpy(dtype=float)
    task_count, model_count = values.shape
    rng = np.random.default_rng(seed)
    sampled_task_indices = rng.integers(0, task_count, size=(draws, task_count))

    boot_scores = np.empty((draws, model_count), dtype=float)
    for draw_index, task_indices in enumerate(sampled_task_indices):
        sampled = values[task_indices, :]
        valid_counts = np.sum(~np.isnan(sampled), axis=0)
        score_sums = np.nansum(sampled, axis=0)
        boot_scores[draw_index] = np.divide(
            score_sums,
            valid_counts,
            out=np.full(model_count, np.nan, dtype=float),
            where=valid_counts > 0,
        )

    score_frame = pd.DataFrame(boot_scores, columns=matrix.columns)
    rank_frame = score_frame.rank(axis=1, method="min", ascending=False, na_option="bottom").astype("Int64")

    return BootstrapResult(
        leaderboard=_summarize_leaderboard(matrix, score_frame, rank_frame),
        rank_distribution=_rank_distribution(rank_frame),
        pairwise_win_probability=_pairwise_win_probability(score_frame),
        boot_scores=score_frame,
        boot_ranks=rank_frame,
    )


def bootstrap_by_dimension(
    trials: pd.DataFrame,
    *,
    dimension: str,
    draws: int = DEFAULT_DRAWS,
    seed: int = 0,
    min_tasks: int = DEFAULT_MIN_SLICE_TASKS,
    min_models: int = DEFAULT_MIN_SLICE_MODELS,
) -> DimensionBootstrapResult:
    if dimension not in trials.columns:
        raise ValueError(f"dimension {dimension!r} is not present in trials")
    if min_tasks <= 0:
        raise ValueError("min_tasks must be positive")
    if min_models <= 0:
        raise ValueError("min_models must be positive")

    summaries: list[pd.DataFrame] = []
    skipped: list[dict[str, object]] = []
    grouped = trials.dropna(subset=[dimension]).groupby(dimension, sort=True, dropna=False, observed=False)
    for slice_index, (slice_value, slice_trials) in enumerate(grouped):
        task_count = int(slice_trials["task_name"].nunique())
        model_count = int(slice_trials["model_key"].nunique())
        trial_count = int(len(slice_trials))
        reason = None
        if task_count < min_tasks:
            reason = f"fewer than {min_tasks} tasks"
        elif model_count < min_models:
            reason = f"fewer than {min_models} models"
        if reason is not None:
            skipped.append(
                {
                    "dimension": dimension,
                    "slice_value": str(slice_value),
                    "n_trials": trial_count,
                    "n_tasks": task_count,
                    "n_models": model_count,
                    "reason": reason,
                }
            )
            continue

        result = bootstrap_rank_stability(slice_trials, draws=draws, seed=seed + slice_index)
        summary = result.leaderboard.copy()
        summary.insert(0, "dimension", dimension)
        summary.insert(1, "slice_value", str(slice_value))
        summary.insert(2, "n_trials", trial_count)
        summary.insert(3, "n_tasks", task_count)
        summary.insert(4, "n_models", model_count)
        summaries.append(summary)

    summary_frame = pd.concat(summaries, ignore_index=True) if summaries else _empty_slice_summary()
    skipped_frame = pd.DataFrame(
        skipped,
        columns=["dimension", "slice_value", "n_trials", "n_tasks", "n_models", "reason"],
    )
    return DimensionBootstrapResult(summaries=summary_frame, skipped_slices=skipped_frame)


def model_task_coverage_by_dimension(trials: pd.DataFrame, *, dimension: str) -> pd.DataFrame:
    if dimension not in trials.columns:
        raise ValueError(f"dimension {dimension!r} is not present in trials")

    rows: list[dict[str, object]] = []
    for slice_value, slice_trials in trials.dropna(subset=[dimension]).groupby(
        dimension, sort=True, dropna=False, observed=False
    ):
        matrix = score_matrix(aggregate_task_model_scores(slice_trials))
        task_count = int(matrix.shape[0])
        for model_key in matrix.columns:
            observed_tasks = int(matrix[model_key].notna().sum())
            rows.append(
                {
                    "dimension": dimension,
                    "slice_value": str(slice_value),
                    "model_key": str(model_key),
                    "n_tasks": task_count,
                    "observed_tasks": observed_tasks,
                    "coverage": float(observed_tasks / task_count) if task_count else np.nan,
                }
            )
    return pd.DataFrame(rows)


def cost_diagnostics_by_dimension(trials: pd.DataFrame, *, dimension: str) -> pd.DataFrame:
    if dimension not in trials.columns:
        raise ValueError(f"dimension {dimension!r} is not present in trials")
    available = [
        column
        for column in [
            "cost_usd",
            "agent_duration_seconds",
            "trial_duration_seconds",
            "n_agent_steps",
            "n_input_tokens",
            "n_output_tokens",
            "peak_context_tokens",
        ]
        if column in trials.columns
    ]
    if not available:
        return pd.DataFrame(columns=["dimension", "slice_value", "model_key", "n_trials"])

    grouped = trials.dropna(subset=[dimension]).groupby([dimension, "model_key"], sort=True, dropna=False, observed=False)
    summary = grouped.agg(n_trials=("score_value", "count")).reset_index()
    summary = summary.rename(columns={dimension: "slice_value"})
    summary.insert(0, "dimension", dimension)
    summary["slice_value"] = summary["slice_value"].astype(str)
    for column in available:
        means = grouped[column].mean().reset_index(name=f"{column}_mean")
        means = means.rename(columns={dimension: "slice_value"})
        means["slice_value"] = means["slice_value"].astype(str)
        summary = summary.merge(means[["slice_value", "model_key", f"{column}_mean"]], on=["slice_value", "model_key"])
    return summary


def swing_tasks_by_dimension(trials: pd.DataFrame, *, dimension: str, limit: int = 30) -> pd.DataFrame:
    if dimension not in trials.columns:
        raise ValueError(f"dimension {dimension!r} is not present in trials")
    rows: list[pd.DataFrame] = []
    for slice_value, slice_trials in trials.dropna(subset=[dimension]).groupby(
        dimension, sort=True, dropna=False, observed=False
    ):
        matrix = score_matrix(aggregate_task_model_scores(slice_trials))
        if matrix.empty:
            continue
        spread = matrix.max(axis=1, skipna=True) - matrix.min(axis=1, skipna=True)
        coverage = matrix.notna().sum(axis=1)
        metadata_columns = [column for column in ["task_name", "language", "repository", "problem_title"] if column in slice_trials]
        metadata = slice_trials[metadata_columns].drop_duplicates("task_name").set_index("task_name")
        table = pd.DataFrame(
            {
                "dimension": dimension,
                "slice_value": str(slice_value),
                "task_name": spread.index,
                "score_spread": spread.values,
                "models_with_result": coverage.values,
            }
        ).join(metadata, on="task_name")
        rows.append(table)
    if not rows:
        return pd.DataFrame()
    return (
        pd.concat(rows, ignore_index=True)
        .sort_values(["score_spread", "models_with_result"], ascending=[False, False])
        .head(limit)
    )


def task_influence_table(
    trials: pd.DataFrame,
    *,
    contender_models: Iterable[str],
    limit: int = 30,
) -> pd.DataFrame:
    contenders = list(dict.fromkeys(str(model) for model in contender_models))
    if not contenders:
        return pd.DataFrame()
    matrix = score_matrix(aggregate_task_model_scores(trials))
    if matrix.empty:
        return pd.DataFrame()
    available_contenders = [model for model in contenders if model in matrix.columns]
    if not available_contenders:
        return pd.DataFrame()

    contender_matrix = matrix[available_contenders]
    spread = contender_matrix.max(axis=1, skipna=True) - contender_matrix.min(axis=1, skipna=True)
    coverage = contender_matrix.notna().sum(axis=1)
    best_model = contender_matrix.idxmax(axis=1, skipna=True)
    worst_model = contender_matrix.idxmin(axis=1, skipna=True)
    metadata_columns = [column for column in ["task_name", "language", "repository", "problem_title"] if column in trials]
    metadata = trials[metadata_columns].drop_duplicates("task_name").set_index("task_name")
    table = pd.DataFrame(
        {
            "task_name": spread.index,
            "contender_score_spread": spread.values,
            "contenders_with_result": coverage.values,
            "best_contender": best_model.values,
            "worst_contender": worst_model.values,
        }
    ).join(metadata, on="task_name")
    return table.sort_values(
        ["contender_score_spread", "contenders_with_result"],
        ascending=[False, False],
    ).head(limit)


def _empty_slice_summary() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "dimension",
            "slice_value",
            "n_trials",
            "n_tasks",
            "n_models",
            "model_key",
            "observed_score",
            "observed_rank",
            "score_mean",
            "score_p05",
            "score_p50",
            "score_p95",
            "rank_mean",
            "rank_p05",
            "rank_p25",
            "rank_p50",
            "rank_p75",
            "rank_p95",
            "rank_interval_width",
            "top1_probability",
            "top3_probability",
        ]
    )


def _summarize_leaderboard(
    matrix: pd.DataFrame,
    score_frame: pd.DataFrame,
    rank_frame: pd.DataFrame,
) -> pd.DataFrame:
    observed = observed_leaderboard(matrix).set_index("model_key")
    rank_float = rank_frame.astype(float)
    summary = pd.DataFrame(index=score_frame.columns)
    summary["score_mean"] = score_frame.mean(axis=0)
    for quantile in SCORE_QUANTILES:
        summary[f"score_p{int(quantile * 100):02d}"] = score_frame.quantile(quantile, axis=0)
    summary["rank_mean"] = rank_float.mean(axis=0)
    for quantile in RANK_QUANTILES:
        summary[f"rank_p{int(quantile * 100):02d}"] = rank_float.quantile(quantile, axis=0)
    summary["rank_interval_width"] = summary["rank_p95"] - summary["rank_p05"]
    summary["top1_probability"] = (rank_frame == 1).mean(axis=0)
    summary["top3_probability"] = (rank_frame <= 3).mean(axis=0)
    summary = observed.join(summary, how="right")
    return (
        summary.reset_index(names="model_key")
        .sort_values(["observed_rank", "observed_score", "model_key"], ascending=[True, False, True])
        .reset_index(drop=True)
    )


def _rank_distribution(rank_frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, float | int | str]] = []
    for model_key in rank_frame.columns:
        counts = rank_frame[model_key].value_counts(dropna=False).sort_index()
        for rank, count in counts.items():
            records.append(
                {
                    "model_key": model_key,
                    "rank": int(rank),
                    "n": int(count),
                    "probability": float(count / len(rank_frame)),
                }
            )
    return pd.DataFrame.from_records(records)


def _pairwise_win_probability(score_frame: pd.DataFrame) -> pd.DataFrame:
    models = list(score_frame.columns)
    result = pd.DataFrame(np.eye(len(models)), index=models, columns=models, dtype=float)
    values = score_frame.to_numpy(dtype=float)

    for left_index, left_model in enumerate(models):
        for right_index, right_model in enumerate(models):
            if left_index == right_index:
                continue
            left = values[:, left_index]
            right = values[:, right_index]
            valid = ~np.isnan(left) & ~np.isnan(right)
            if not valid.any():
                probability = np.nan
            else:
                wins = left[valid] > right[valid]
                ties = left[valid] == right[valid]
                probability = float((wins.sum() + 0.5 * ties.sum()) / valid.sum())
            result.loc[left_model, right_model] = probability
    return result
