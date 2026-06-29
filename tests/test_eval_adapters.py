from __future__ import annotations

import pandas as pd

from deepswe_rank_stability.analysis.resampling import bootstrap_rank_stability
from deepswe_rank_stability.data.evals import (
    load_eval,
    load_table_eval_from_toml,
    normalize_eval_trials,
)
from deepswe_rank_stability.data.tau_bench import load_dataset as load_tau_bench_dataset


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


def test_tau_bench_adapter_normalizes_results_json(tmp_path) -> None:
    results_path = tmp_path / "tau_results.json"
    results_path.write_text(
        """
{
  "info": {
    "git_commit": "abc123",
    "num_trials": 2,
    "seed": 42,
    "environment_info": {"domain_name": "airline"},
    "agent_info": {"implementation": "llm_agent", "llm": "openai/gpt-4.1"},
    "user_info": {"implementation": "user_simulator", "llm": "openai/gpt-4.1-mini"}
  },
  "tasks": [
    {
      "id": "1",
      "evaluation_criteria": {
        "actions": [{"action_id": "1_0", "requestor": "assistant", "name": "get_user", "arguments": {}}],
        "communicate_info": ["refund denied"],
        "reward_basis": ["DB", "COMMUNICATE"]
      }
    },
    {
      "id": "2",
      "evaluation_criteria": {
        "actions": [{"action_id": "2_0", "requestor": "assistant", "name": "write_case", "arguments": {}}],
        "reward_basis": ["DB", "COMMUNICATE"]
      }
    }
  ],
  "simulations": [
    {
      "id": "sim-1",
      "task_id": "1",
      "trial": 0,
      "seed": 42,
      "duration": 12.5,
      "termination_reason": "agent_stop",
      "reward_info": {
        "reward": 1.0,
        "reward_basis": ["DB", "COMMUNICATE"],
        "db_check": {"db_match": true, "db_reward": 1.0},
        "reward_breakdown": {"DB": 1.0, "COMMUNICATE": 1.0},
        "action_checks": [{"action_match": false, "action_reward": 0.0}]
      },
      "messages": [{"role": "assistant", "tool_calls": [{"name": "different_tool"}]}]
    },
    {
      "id": "sim-2",
      "task_id": "1",
      "trial": 1,
      "seed": 43,
      "duration": 15.0,
      "termination_reason": "agent_stop",
      "reward_info": {
        "reward": 0.0,
        "reward_basis": ["DB", "COMMUNICATE"],
        "db_check": {"db_match": false, "db_reward": 0.0},
        "reward_breakdown": {"DB": 0.0, "COMMUNICATE": 1.0},
        "action_checks": [{"action_match": true, "action_reward": 1.0}]
      },
      "messages": [{"role": "assistant", "tool_calls": [{"name": "get_user"}]}]
    }
  ]
}
""",
        encoding="utf-8",
    )

    dataset = load_tau_bench_dataset(paths=[results_path])

    assert dataset.trials["eval_id"].unique().tolist() == ["tau_bench"]
    assert dataset.trials["task_id"].tolist() == ["airline:1", "airline:1"]
    assert dataset.trials["system_id"].unique().tolist() == ["openai/gpt-4.1"]
    assert dataset.trials["score"].tolist() == [1.0, 0.0]
    assert dataset.trials["communicate_reward"].tolist() == [1.0, 1.0]
    assert dataset.trials["partial_action_reward"].tolist() == [0.0, 1.0]
    assert dataset.trials["tool_call_count"].tolist() == [1, 1]
    assert not bool(dataset.tasks.set_index("task_id").loc["airline:1", "action_required"])


def test_tau_bench_adapter_keeps_actions_diagnostic_without_action_basis(tmp_path) -> None:
    results_path = tmp_path / "tau_results.json"
    results_path.write_text(
        """
{
  "info": {
    "environment_info": {"domain_name": "retail"},
    "agent_info": {"implementation": "llm_agent", "llm": "model-a"},
    "user_info": {"implementation": "user_simulator", "llm": "user-model"}
  },
  "tasks": [
    {
      "id": "10",
      "evaluation_criteria": {
        "actions": [{"action_id": "a", "requestor": "assistant", "name": "reference_path", "arguments": {}}],
        "reward_basis": ["DB", "COMMUNICATE"]
      }
    },
    {
      "id": "11",
      "evaluation_criteria": {
        "actions": [{"action_id": "b", "requestor": "assistant", "name": "required_path", "arguments": {}}],
        "reward_basis": ["DB", "ACTION"]
      }
    }
  ],
  "simulations": [
    {
      "id": "sim-10",
      "task_id": "10",
      "trial": 0,
      "termination_reason": "agent_stop",
      "reward_info": {
        "reward": 1.0,
        "reward_basis": ["DB", "COMMUNICATE"],
        "db_check": {"db_match": true, "db_reward": 1.0},
        "reward_breakdown": {"DB": 1.0, "COMMUNICATE": 1.0},
        "action_checks": [{"action_match": false, "action_reward": 0.0}]
      }
    },
    {
      "id": "sim-11",
      "task_id": "11",
      "trial": 0,
      "termination_reason": "agent_stop",
      "reward_info": {
        "reward": 0.0,
        "reward_basis": ["DB", "ACTION"],
        "db_check": {"db_match": true, "db_reward": 1.0},
        "reward_breakdown": {"DB": 1.0, "ACTION": 0.0},
        "action_checks": [{"action_match": false, "action_reward": 0.0}]
      }
    }
  ]
}
""",
        encoding="utf-8",
    )

    dataset = load_tau_bench_dataset(paths=[results_path])
    rows = dataset.trials.set_index("task_id")
    tasks = dataset.tasks.set_index("task_id")

    assert rows.loc["retail:10", "score"] == 1.0
    assert rows.loc["retail:10", "partial_action_reward"] == 0.0
    assert not bool(rows.loc["retail:10", "action_required"])
    assert not bool(tasks.loc["retail:10", "action_required"])
    assert bool(rows.loc["retail:11", "action_required"])
    assert bool(tasks.loc["retail:11", "action_required"])


def test_tau_bench_adapter_supports_bootstrap_with_repeated_trials(tmp_path) -> None:
    csv_path = tmp_path / "tau_rows.csv"
    csv_path.write_text(
        "simulation_id,task_id,domain,agent_llm,info_agent_implementation,reward,reward_basis,trial\n"
        "a1,1,airline,model-a,llm_agent,1,DB+COMMUNICATE,0\n"
        "a2,1,airline,model-a,llm_agent,0,DB+COMMUNICATE,1\n"
        "b1,1,airline,model-b,llm_agent,1,DB+COMMUNICATE,0\n"
        "a3,2,airline,model-a,llm_agent,0,DB+COMMUNICATE,0\n"
        "b2,2,airline,model-b,llm_agent,1,DB+COMMUNICATE,0\n",
        encoding="utf-8",
    )
    dataset = load_tau_bench_dataset(paths=[csv_path])

    left = bootstrap_rank_stability(dataset.trials, draws=20, seed=3, score_column="reward")
    right = bootstrap_rank_stability(dataset.trials, draws=20, seed=3, score_column="reward")

    assert left.leaderboard.iloc[0]["model_key"] == "model-b"
    pd.testing.assert_frame_equal(left.leaderboard, right.leaderboard)


def test_tau_bench_adapter_loads_dir_format_simulation_files(tmp_path) -> None:
    result_dir = tmp_path / "tau_dir"
    simulations_dir = result_dir / "simulations"
    simulations_dir.mkdir(parents=True)
    (result_dir / "results.json").write_text(
        """
{
  "info": {
    "environment_info": {"domain_name": "telecom"},
    "agent_info": {"implementation": "llm_agent", "llm": "model-a"},
    "user_info": {"implementation": "user_simulator", "llm": "user-model"}
  },
  "tasks": [
    {
      "id": "20",
      "evaluation_criteria": {
        "actions": [],
        "reward_basis": ["DB", "COMMUNICATE"]
      }
    }
  ],
  "simulation_index": [
    {"id": "sim-20", "task_id": "20", "trial": 0, "reward": 1.0}
  ]
}
""",
        encoding="utf-8",
    )
    (simulations_dir / "sim-20.json").write_text(
        """
{
  "id": "sim-20",
  "task_id": "20",
  "trial": 0,
  "termination_reason": "agent_stop",
  "reward_info": {
    "reward": 1.0,
    "reward_basis": ["DB", "COMMUNICATE"],
    "db_check": {"db_match": true, "db_reward": 1.0},
    "reward_breakdown": {"DB": 1.0, "COMMUNICATE": 1.0}
  }
}
""",
        encoding="utf-8",
    )

    dataset = load_tau_bench_dataset(paths=[result_dir])

    assert dataset.trials["trial_id"].tolist() == ["sim-20"]
    assert dataset.trials["task_id"].tolist() == ["telecom:20"]
    assert dataset.trials["db_reward"].tolist() == [1.0]
