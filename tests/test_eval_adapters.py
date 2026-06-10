from __future__ import annotations

import pandas as pd

from deepswe_rank_stability.analysis.resampling import bootstrap_rank_stability
from deepswe_rank_stability.data.evals import (
    load_eval,
    load_table_eval_from_toml,
    normalize_eval_trials,
)


def test_normalize_eval_trials_adds_generic_columns_and_legacy_aliases() -> None:
    raw = pd.DataFrame(
        [
            {
                "trial_name": "trial-1",
                "task_name": "task-1",
                "model_key": "model-a",
                "score_value": "0.75",
                "source": "custom",
            }
        ]
    )

    trials = normalize_eval_trials(
        raw,
        eval_id="custom_eval",
        column_map={
            "trial_id": "trial_name",
            "task_id": "task_name",
            "system_id": "model_key",
            "score": "score_value",
        },
        score_column="score",
    )

    assert trials.loc[0, "eval_id"] == "custom_eval"
    assert trials.loc[0, "trial_id"] == "trial-1"
    assert trials.loc[0, "task_id"] == "task-1"
    assert trials.loc[0, "system_id"] == "model-a"
    assert trials.loc[0, "score"] == 0.75
    assert trials.loc[0, "trial_name"] == "trial-1"
    assert trials.loc[0, "task_name"] == "task-1"
    assert trials.loc[0, "model_key"] == "model-a"
    assert trials.loc[0, "score_value"] == 0.75


def test_builtin_eval_registry_loads_deepswe_and_swebenchpro() -> None:
    deep_swe = load_eval("deep_swe")
    swebench_pro = load_eval("swebench_pro")

    assert deep_swe.eval_id == "deep_swe"
    assert swebench_pro.eval_id == "swebench_pro"
    assert set(deep_swe.trials["source"]) == {"deep-swe"}
    assert set(swebench_pro.trials["source"]) == {"swebenchpro"}
    assert deep_swe.default_filters["eval_scope"] == "full"
    assert swebench_pro.default_filters["eval_scope"] == "cross-bench"


def test_bootstrap_uses_explicit_metric_column() -> None:
    trials = pd.DataFrame(
        [
            {"trial_id": "a1", "task_id": "t1", "system_id": "a", "score": 0.0, "alt_score": 1.0},
            {"trial_id": "b1", "task_id": "t1", "system_id": "b", "score": 1.0, "alt_score": 0.0},
            {"trial_id": "a2", "task_id": "t2", "system_id": "a", "score": 0.0, "alt_score": 1.0},
            {"trial_id": "b2", "task_id": "t2", "system_id": "b", "score": 1.0, "alt_score": 0.0},
        ]
    )

    normalized = normalize_eval_trials(
        trials,
        eval_id="metric_eval",
        column_map={"trial_id": "trial_id", "task_id": "task_id", "system_id": "system_id"},
        score_column="score",
    )
    result = bootstrap_rank_stability(normalized, draws=20, seed=0, score_column="alt_score")

    assert result.leaderboard.iloc[0]["model_key"] == "a"
    assert result.leaderboard.iloc[0]["observed_score"] == 1.0


def test_table_eval_from_toml_maps_columns(tmp_path) -> None:
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "run,task,model,score,lang\n"
        "r1,t1,m1,1,python\n"
        "r2,t1,m2,0,python\n",
        encoding="utf-8",
    )
    toml_path = tmp_path / "eval.toml"
    toml_path.write_text(
        f"""
[eval]
id = "toy"
label = "Toy Eval"
default_metric = "score"

[data]
uri = "{csv_path}"
format = "csv"

[columns]
trial_id = "run"
task_id = "task"
system_id = "model"
score = "score"

[[metrics]]
name = "score"
label = "Score"
column = "score"

[[dimensions]]
name = "language"
label = "Language"
column = "lang"
""",
        encoding="utf-8",
    )

    dataset = load_table_eval_from_toml(toml_path)

    assert dataset.eval_id == "toy"
    assert dataset.label == "Toy Eval"
    assert dataset.trials["system_id"].tolist() == ["m1", "m2"]
    assert dataset.metrics[0].column == "score"
    assert dataset.dimensions[0].column == "lang"
