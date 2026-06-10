from __future__ import annotations

import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from deepswe_rank_stability.data.deepswe import DEFAULT_CACHE_DIR, load_dataset


@dataclass(frozen=True)
class MetricSpec:
    name: str
    label: str
    column: str
    higher_is_better: bool = True


@dataclass(frozen=True)
class DimensionSpec:
    name: str
    label: str
    column: str
    bootstrap: bool = True


@dataclass(frozen=True)
class EvalDataset:
    eval_id: str
    label: str
    trials: pd.DataFrame
    tasks: pd.DataFrame
    metrics: tuple[MetricSpec, ...]
    dimensions: tuple[DimensionSpec, ...]
    default_filters: dict[str, Any]
    default_metric: str

    def metric(self, metric_name: str | None = None) -> MetricSpec:
        selected = metric_name or self.default_metric
        for metric in self.metrics:
            if metric.name == selected:
                return metric
        raise ValueError(f"Unknown metric {selected!r} for eval {self.eval_id!r}")


EvalLoader = Callable[[], EvalDataset]


def normalize_eval_trials(
    trials: pd.DataFrame,
    *,
    eval_id: str,
    column_map: dict[str, str],
    score_column: str,
) -> pd.DataFrame:
    frame = trials.copy()
    for target, source in column_map.items():
        if source not in frame.columns:
            raise ValueError(f"source column {source!r} for {target!r} is not present")
        frame[target] = frame[source]

    required = {"trial_id", "task_id", "system_id", score_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"normalized eval rows are missing required columns: {missing}")

    frame["eval_id"] = eval_id
    frame["score"] = pd.to_numeric(frame[score_column], errors="coerce")
    frame[score_column] = pd.to_numeric(frame[score_column], errors="coerce")

    # Keep legacy aliases while the dashboard and some tests migrate.
    frame["trial_name"] = frame["trial_id"]
    frame["task_name"] = frame["task_id"]
    frame["model_key"] = frame["system_id"]
    frame["score_value"] = frame[score_column]
    if "passed" not in frame.columns:
        frame["passed"] = frame["score"].fillna(0).astype(float) >= 1.0
    return frame


def normalize_eval_tasks(tasks: pd.DataFrame, *, eval_id: str) -> pd.DataFrame:
    frame = tasks.copy()
    if "task_id" not in frame.columns and "task_name" in frame.columns:
        frame["task_id"] = frame["task_name"]
    if "task_name" not in frame.columns and "task_id" in frame.columns:
        frame["task_name"] = frame["task_id"]
    if "task_id" in frame.columns:
        frame["eval_id"] = eval_id
    return frame


def _declared_dimensions(trials: pd.DataFrame) -> tuple[DimensionSpec, ...]:
    candidates = [
        ("source", "Source", True),
        ("eval_scope", "Eval scope", True),
        ("outcome", "Outcome", False),
        ("language", "Language", True),
        ("repository", "Repository", False),
    ]
    return tuple(
        DimensionSpec(name=column, label=label, column=column, bootstrap=bootstrap)
        for column, label, bootstrap in candidates
        if column in trials.columns and trials[column].notna().any()
    )


@lru_cache(maxsize=1)
def _raw_deepswe_dataset():
    return load_dataset(cache_dir=DEFAULT_CACHE_DIR)


def _load_deepswe_source_eval(
    *,
    eval_id: str,
    label: str,
    source: str,
    default_eval_scope: str,
) -> EvalDataset:
    raw = _raw_deepswe_dataset()
    trials = raw.trials[raw.trials["source"] == source].copy()
    normalized = normalize_eval_trials(
        trials,
        eval_id=eval_id,
        column_map={
            "trial_id": "trial_name",
            "task_id": "task_name",
            "system_id": "model_key",
            "score": "score_value",
        },
        score_column="score",
    )
    tasks = normalize_eval_tasks(raw.tasks, eval_id=eval_id)
    metrics = (
        MetricSpec(name="score", label="Score", column="score"),
        MetricSpec(name="score_value", label="Score value", column="score_value"),
    )
    return EvalDataset(
        eval_id=eval_id,
        label=label,
        trials=normalized,
        tasks=tasks,
        metrics=metrics,
        dimensions=_declared_dimensions(normalized),
        default_filters={
            "source": source,
            "eval_scope": default_eval_scope,
            "included_in_score": True,
            "outcome": "All",
            "language": "All",
            "repository": "All",
        },
        default_metric="score",
    )


def load_deep_swe_eval() -> EvalDataset:
    return _load_deepswe_source_eval(
        eval_id="deep_swe",
        label="DeepSWE",
        source="deep-swe",
        default_eval_scope="full",
    )


def load_swebench_pro_eval() -> EvalDataset:
    return _load_deepswe_source_eval(
        eval_id="swebench_pro",
        label="SWE-Bench Pro",
        source="swebenchpro",
        default_eval_scope="cross-bench",
    )


BUILTIN_EVALS: dict[str, EvalLoader] = {
    "deep_swe": load_deep_swe_eval,
    "swebench_pro": load_swebench_pro_eval,
}


def list_eval_ids() -> list[str]:
    return list(BUILTIN_EVALS)


def load_eval(eval_id: str) -> EvalDataset:
    try:
        loader = BUILTIN_EVALS[eval_id]
    except KeyError as exc:
        known = ", ".join(sorted(BUILTIN_EVALS))
        raise ValueError(f"Unknown eval {eval_id!r}; expected one of: {known}") from exc
    return loader()


def load_table_eval_from_toml(path: str | Path) -> EvalDataset:
    config_path = Path(path)
    config = tomllib.loads(config_path.read_text())
    eval_config = config["eval"]
    data_config = config["data"]
    columns = config["columns"]
    metrics_config = config.get("metrics", [])
    dimensions_config = config.get("dimensions", [])

    uri = data_config["uri"]
    fmt = data_config.get("format") or Path(uri).suffix.lstrip(".").lower()
    if fmt == "csv":
        raw = pd.read_csv(uri)
    elif fmt == "json":
        raw = pd.read_json(uri)
    elif fmt == "parquet":
        raw = pd.read_parquet(uri)
    else:
        raise ValueError(f"Unsupported table eval format {fmt!r}")

    metric_specs = tuple(
        MetricSpec(
            name=str(metric["name"]),
            label=str(metric.get("label", metric["name"])),
            column=str(metric["column"]),
            higher_is_better=bool(metric.get("higher_is_better", True)),
        )
        for metric in metrics_config
    )
    if not metric_specs:
        metric_specs = (MetricSpec(name="score", label="Score", column="score"),)

    score_column = metric_specs[0].column
    trials = normalize_eval_trials(
        raw,
        eval_id=str(eval_config["id"]),
        column_map=dict(columns),
        score_column=score_column,
    )
    dimension_specs = tuple(
        DimensionSpec(
            name=str(dimension["name"]),
            label=str(dimension.get("label", dimension["name"])),
            column=str(dimension["column"]),
            bootstrap=bool(dimension.get("bootstrap", True)),
        )
        for dimension in dimensions_config
    )
    return EvalDataset(
        eval_id=str(eval_config["id"]),
        label=str(eval_config.get("label", eval_config["id"])),
        trials=trials,
        tasks=normalize_eval_tasks(pd.DataFrame({"task_id": sorted(trials["task_id"].dropna().unique())}), eval_id=str(eval_config["id"])),
        metrics=metric_specs,
        dimensions=dimension_specs,
        default_filters=dict(config.get("default_filters", {})),
        default_metric=str(eval_config.get("default_metric", metric_specs[0].name)),
    )
