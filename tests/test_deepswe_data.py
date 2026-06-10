from __future__ import annotations

from deepswe_rank_stability.data.deepswe import (
    join_task_metadata,
    model_effort_key,
    normalize_tasks,
    normalize_trials,
)


def test_model_effort_key_omits_missing_effort() -> None:
    assert model_effort_key("model-a", None) == "model-a"
    assert model_effort_key("model-a", "high") == "model-a [high]"


def test_normalize_trials_adds_model_key_and_types() -> None:
    trials = normalize_trials(
        {
            "rows": [
                {
                    "trial_name": "trial-1",
                    "task_name": "task-1",
                    "source": "deep-swe",
                    "eval_scope": "full",
                    "model": "model-a",
                    "reasoning_effort": "high",
                    "score_value": "1",
                    "passed": 1,
                    "errored": 0,
                    "included_in_score": True,
                    "outcome": "pass",
                }
            ]
        }
    )

    assert trials.loc[0, "model_key"] == "model-a [high]"
    assert trials.loc[0, "score_value"] == 1
    assert bool(trials.loc[0, "passed"]) is True
    assert bool(trials.loc[0, "errored"]) is False


def test_join_task_metadata_keeps_all_trials() -> None:
    trials = normalize_trials(
        {
            "rows": [
                {
                    "trial_name": "trial-1",
                    "task_name": "task-1",
                    "source": "deep-swe",
                    "eval_scope": "full",
                    "model": "model-a",
                    "score_value": 1,
                    "passed": True,
                    "errored": False,
                    "included_in_score": True,
                    "outcome": "pass",
                }
            ]
        }
    )
    tasks = normalize_tasks(
        {
            "rows": [
                {
                    "id": "task-1",
                    "language": "python",
                    "repository": "owner/repo",
                    "problem_title": "Fix bug",
                }
            ]
        }
    )

    joined = join_task_metadata(trials, tasks)

    assert len(joined) == 1
    assert joined.loc[0, "language"] == "python"
    assert joined.loc[0, "repository"] == "owner/repo"

