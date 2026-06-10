from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests

ARTIFACT_BASE_URL = "https://deepswe.datacurve.ai/artifacts"
ARTIFACTS = {
    "trials": f"{ARTIFACT_BASE_URL}/trials.json",
    "tasks": f"{ARTIFACT_BASE_URL}/tasks.json",
    "release": f"{ARTIFACT_BASE_URL}/release.json",
}
DEFAULT_CACHE_DIR = Path(".cache/deepswe_rank_stability")


@dataclass(frozen=True)
class DeepSWEDataset:
    trials: pd.DataFrame
    tasks: pd.DataFrame
    release: dict[str, Any]


def model_effort_key(model: Any, reasoning_effort: Any) -> str:
    if pd.isna(reasoning_effort) or reasoning_effort in ("", "null", None):
        return str(model)
    return f"{model} [{reasoning_effort}]"


def _series_or_default(frame: pd.DataFrame, column: str, default: Any) -> pd.Series:
    if column in frame.columns:
        return frame[column]
    return pd.Series([default] * len(frame), index=frame.index)


def artifact_path(name: str, cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    if name not in ARTIFACTS:
        known = ", ".join(sorted(ARTIFACTS))
        raise ValueError(f"Unknown artifact {name!r}; expected one of: {known}")
    return cache_dir / f"{name}.json"


def download_artifact(
    name: str,
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    refresh: bool = False,
    timeout: float = 60.0,
) -> dict[str, Any]:
    path = artifact_path(name, cache_dir)
    if path.exists() and not refresh:
        return json.loads(path.read_text())

    response = requests.get(ARTIFACTS[name], timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def normalize_trials(payload: dict[str, Any]) -> pd.DataFrame:
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("trials artifact must contain a list under 'rows'")

    trials = pd.DataFrame(rows)
    if trials.empty:
        return trials

    required = {"trial_name", "task_name", "source", "eval_scope", "model", "score_value"}
    missing = sorted(required - set(trials.columns))
    if missing:
        raise ValueError(f"trials artifact is missing required columns: {missing}")

    efforts = trials["reasoning_effort"] if "reasoning_effort" in trials.columns else [None] * len(trials)
    trials["model_key"] = [
        model_effort_key(model, effort) for model, effort in zip(trials["model"], efforts, strict=False)
    ]
    trials["score_value"] = pd.to_numeric(trials["score_value"], errors="coerce")
    trials["passed"] = _series_or_default(trials, "passed", False).fillna(False).astype(bool)
    trials["errored"] = _series_or_default(trials, "errored", False).fillna(False).astype(bool)
    trials["included_in_score"] = (
        _series_or_default(trials, "included_in_score", False).fillna(False).astype(bool)
    )

    for column in [
        "cost_usd",
        "n_agent_steps",
        "n_input_tokens",
        "n_output_tokens",
        "peak_context_tokens",
        "agent_duration_seconds",
        "trial_duration_seconds",
    ]:
        if column in trials.columns:
            trials[column] = pd.to_numeric(trials[column], errors="coerce")

    return trials


def normalize_tasks(payload: dict[str, Any]) -> pd.DataFrame:
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("tasks artifact must contain a list under 'rows'")

    tasks = pd.DataFrame(rows)
    if tasks.empty:
        return tasks
    if "id" not in tasks.columns:
        raise ValueError("tasks artifact is missing required column: id")
    return tasks.rename(columns={"id": "task_name"})


def join_task_metadata(trials: pd.DataFrame, tasks: pd.DataFrame) -> pd.DataFrame:
    if tasks.empty:
        return trials.copy()

    metadata_columns = [
        column
        for column in [
            "task_name",
            "language",
            "repository",
            "problem_title",
            "display_description",
            "prompt_characters",
        ]
        if column in tasks.columns
    ]
    return trials.merge(tasks[metadata_columns], on="task_name", how="left", validate="many_to_one")


def load_dataset(
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    refresh: bool = False,
) -> DeepSWEDataset:
    trials_payload = download_artifact("trials", cache_dir=cache_dir, refresh=refresh)
    tasks_payload = download_artifact("tasks", cache_dir=cache_dir, refresh=refresh)
    release_payload = download_artifact("release", cache_dir=cache_dir, refresh=refresh)

    tasks = normalize_tasks(tasks_payload)
    trials = join_task_metadata(normalize_trials(trials_payload), tasks)
    return DeepSWEDataset(trials=trials, tasks=tasks, release=release_payload)
