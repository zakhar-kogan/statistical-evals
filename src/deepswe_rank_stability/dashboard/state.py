from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


def friendly_empty_message(filtered: pd.DataFrame) -> str | None:
    if filtered.empty:
        return "No trial rows match the current filters. Relax source, scope, outcome, or model filters."
    if filtered["task_name"].nunique() == 0:
        return "The current filters contain no benchmark tasks."
    if filtered["model_key"].nunique() < 2:
        return "Select at least two model+effort rows to compare rank stability."
    return None


def eligible_variance_dimensions(
    trials: pd.DataFrame,
    *,
    candidates: tuple[str, ...] = ("language", "source", "eval_scope"),
    min_tasks: int = 10,
    min_models: int = 2,
) -> list[str]:
    eligible: list[str] = []
    for dimension in candidates:
        if dimension == "repository" or dimension not in trials.columns:
            continue
        values = trials[dimension].dropna().unique()
        if len(values) <= 1:
            continue
        groups = trials.dropna(subset=[dimension]).groupby(dimension, dropna=False, observed=False)
        has_eligible_slice = any(
            slice_trials["task_name"].nunique() >= min_tasks and slice_trials["model_key"].nunique() >= min_models
            for _, slice_trials in groups
        )
        if has_eligible_slice:
            eligible.append(dimension)
    return eligible


def variance_empty_message(eligible_dimensions: list[str]) -> str | None:
    if eligible_dimensions:
        return None
    return (
        "No bootstrap variance dimensions are eligible for the current filters. "
        "Try relaxing language, source, scope, repository, or model filters."
    )


@dataclass(frozen=True)
class DashboardSelection:
    source: str
    eval_scope: str
    included_in_score: bool | None
    outcome: str
    language: str
    repository: str
    model_keys: tuple[str, ...]
    draws: int
    seed: int
    include_cross_benchmark: bool = False

    def label(self) -> str:
        included = (
            "all"
            if self.included_in_score is None
            else "included"
            if self.included_in_score
            else "excluded"
        )
        pieces = [
            f"source={self.source}",
            f"eval_scope={self.eval_scope}",
            f"inclusion={included}",
            f"draws={self.draws:,}",
            f"seed={self.seed}",
        ]
        if self.outcome != "All":
            pieces.append(f"outcome={self.outcome}")
        if self.language != "All":
            pieces.append(f"language={self.language}")
        if self.repository != "All":
            pieces.append(f"repository={self.repository}")
        if self.model_keys:
            pieces.append(f"models={len(self.model_keys)} selected")
        return " · ".join(pieces)


def source_options(all_sources: list[str], *, include_cross_benchmark: bool) -> list[str]:
    if include_cross_benchmark:
        return ["All", *sorted(set(all_sources))]
    return ["All", *sorted(source for source in set(all_sources) if source != "swebenchpro")]


def coerce_source_for_mode(source: str, all_sources: list[str], *, include_cross_benchmark: bool) -> str:
    options = source_options(all_sources, include_cross_benchmark=include_cross_benchmark)
    if source in options:
        return source
    if "deep-swe" in options:
        return "deep-swe"
    return options[0]


def submit_selection(
    *,
    current: DashboardSelection | None,
    trigger_count: int,
    last_trigger_count: int,
    pending: DashboardSelection,
) -> tuple[DashboardSelection, int]:
    if current is None or trigger_count != last_trigger_count:
        return pending, trigger_count
    return current, last_trigger_count


def rank_model_order(leaderboard: pd.DataFrame) -> list[str]:
    return (
        leaderboard.sort_values(["observed_rank", "observed_score", "model_key"], ascending=[True, False, True])[
            "model_key"
        ]
        .astype(str)
        .tolist()
    )


def contender_model_order(leaderboard: pd.DataFrame, *, top_n: int = 10) -> list[str]:
    ordered = rank_model_order(leaderboard)
    top = ordered[:top_n]
    nonzero_top3 = (
        leaderboard[leaderboard["top3_probability"] > 0]["model_key"]
        .astype(str)
        .tolist()
        if "top3_probability" in leaderboard.columns
        else []
    )
    contenders = list(dict.fromkeys([*top, *nonzero_top3]))
    return [model_key for model_key in ordered if model_key in contenders]


def slice_values_with_summaries(summary: pd.DataFrame) -> list[str]:
    if summary.empty:
        return []
    return (
        summary[["slice_value", "n_tasks"]]
        .drop_duplicates()
        .sort_values(["n_tasks", "slice_value"], ascending=[False, True])["slice_value"]
        .astype(str)
        .tolist()
    )


def plotly_top_first_categoryarray(order: list[str]) -> list[str]:
    return order


def top_model_summary(leaderboard: pd.DataFrame) -> dict[str, object]:
    if leaderboard.empty:
        return {
            "model_key": "None",
            "observed_rank": None,
            "observed_score": None,
            "top1_probability": None,
            "rank_p50": None,
            "rank_p05": None,
            "rank_p95": None,
        }
    top_row = leaderboard.sort_values(
        ["observed_rank", "observed_score", "model_key"],
        ascending=[True, False, True],
    ).iloc[0]
    return {
        "model_key": str(top_row["model_key"]),
        "observed_rank": int(top_row["observed_rank"]),
        "observed_score": float(top_row["observed_score"]),
        "top1_probability": float(top_row["top1_probability"]),
        "rank_p50": float(top_row["rank_p50"]),
        "rank_p05": float(top_row["rank_p05"]),
        "rank_p95": float(top_row["rank_p95"]),
    }


def rank_axis_range(leaderboard: pd.DataFrame) -> tuple[float, float]:
    max_rank = float(max(leaderboard["rank_p95"].max(), leaderboard["observed_rank"].max()))
    return 0.5, max_rank + 0.8


def pairwise_strength(pairwise: pd.DataFrame) -> pd.DataFrame:
    strengths: list[dict[str, Any]] = []
    for model in pairwise.index:
        values = pairwise.loc[model].drop(labels=[model], errors="ignore").astype(float)
        strengths.append(
            {
                "model_key": model,
                "pairwise_strength": float(values.mean(skipna=True)) if len(values) else np.nan,
            }
        )
    return (
        pd.DataFrame(strengths)
        .sort_values(["pairwise_strength", "model_key"], ascending=[False, True])
        .reset_index(drop=True)
    )


def order_pairwise_by_strength(pairwise: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    strengths = pairwise_strength(pairwise)
    order = strengths["model_key"].tolist()
    return pairwise.reindex(index=order, columns=order), strengths
