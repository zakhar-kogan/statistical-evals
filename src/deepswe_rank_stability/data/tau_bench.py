from __future__ import annotations

import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from deepswe_rank_stability.data.deepswe import DEFAULT_CACHE_DIR

TAU_BENCH_EVAL_ID = "tau_bench"
TAU_BENCH_SOURCE = "tau2-bench"
TAU_BENCH_EVAL_SCOPE = "tau3-current"
TAU_BENCH_RESULT_ENV = "TAU_BENCH_RESULTS"
TEXT_MODE_DOMAINS = {"airline", "retail", "telecom", "telecom-workflow"}

BASE_COLUMNS = [
    "eval_id",
    "trial_id",
    "task_id",
    "system_id",
    "score",
    "reward",
    "db_reward",
    "communicate_reward",
    "env_assertion_reward",
    "action_reward",
    "partial_action_reward",
    "pass_at_k",
    "source",
    "eval_scope",
    "included_in_score",
    "outcome",
    "domain",
    "agent_llm",
    "user_llm",
    "agent_strategy",
    "user_strategy",
    "trial_index",
    "task_split",
    "reward_basis",
    "action_required",
    "communication_mode",
    "termination_reason",
    "duration",
    "agent_cost",
    "user_cost",
    "num_messages",
    "tool_call_count",
    "result_file",
    "source_version",
]

TASK_COLUMNS = [
    "eval_id",
    "task_id",
    "task_name",
    "domain",
    "task_split",
    "reward_basis",
    "action_required",
    "task_num_agent_actions",
    "task_num_user_actions",
    "task_num_actions",
    "task_num_env_assertions",
    "task_num_nl_assertions",
]


@dataclass(frozen=True)
class TauBenchDataset:
    trials: pd.DataFrame
    tasks: pd.DataFrame
    result_paths: tuple[Path, ...]


def discover_result_paths(cache_dir: Path = DEFAULT_CACHE_DIR) -> tuple[Path, ...]:
    env_value = os.environ.get(TAU_BENCH_RESULT_ENV)
    if env_value:
        candidates = [Path(part).expanduser() for part in env_value.split(os.pathsep) if part]
    else:
        candidates = [
            cache_dir / "tau_bench",
            cache_dir / "tau2" / "results" / "final",
            Path("data") / "tau_bench",
            Path("data") / "tau2" / "results" / "final",
        ]
    return tuple(path for path in candidates if path.exists())


def load_dataset(
    *,
    paths: Iterable[str | Path] | None = None,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> TauBenchDataset:
    result_paths = tuple(Path(path) for path in paths) if paths is not None else discover_result_paths(cache_dir)
    rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    for path in result_paths:
        loaded_rows, loaded_tasks = _load_path(path)
        rows.extend(loaded_rows)
        task_rows.extend(loaded_tasks)

    trials = pd.DataFrame(rows, columns=BASE_COLUMNS) if rows else pd.DataFrame(columns=BASE_COLUMNS)
    tasks = pd.DataFrame(task_rows, columns=TASK_COLUMNS) if task_rows else pd.DataFrame(columns=TASK_COLUMNS)
    if not trials.empty:
        trials = _finalize_trials(trials)
    if not tasks.empty:
        tasks = _finalize_tasks(tasks)
    return TauBenchDataset(trials=trials, tasks=tasks, result_paths=result_paths)


def _load_path(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if path.is_dir():
        result_file = path / "results.json"
        if result_file.exists():
            return _load_results_json(result_file)
        rows: list[dict[str, Any]] = []
        tasks: list[dict[str, Any]] = []
        for child in sorted(path.iterdir()):
            if child.is_dir() or child.suffix.lower() in {".json", ".jsonl", ".csv"}:
                child_rows, child_tasks = _load_path(child)
                rows.extend(child_rows)
                tasks.extend(child_tasks)
        return rows, tasks
    if path.suffix.lower() == ".csv":
        return _load_flat_table(pd.read_csv(path), path), []
    if path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        return _load_records(records, path), []
    if path.suffix.lower() == ".json":
        return _load_json_file(path)
    return [], []


def _load_json_file(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return _load_records(payload, path), []
    if not isinstance(payload, dict):
        return [], []
    if "simulations" in payload or "simulation_index" in payload:
        return _load_results_payload(payload, path)
    for key in ("rows", "data", "results"):
        if isinstance(payload.get(key), list):
            return _load_records(payload[key], path), []
    return _load_records([payload], path), []


def _load_results_json(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (path.parent / "simulations").is_dir() and "simulations" not in payload:
        info = payload.get("info") or {}
        tasks = [task for task in payload.get("tasks", []) if isinstance(task, dict)]
        task_lookup = {str(task.get("id")): task for task in tasks}
        sim_rows = []
        for sim_file in sorted((path.parent / "simulations").glob("*.json")):
            sim = json.loads(sim_file.read_text(encoding="utf-8"))
            sim_rows.append(_simulation_to_row(sim, info, task_lookup.get(str(sim.get("task_id"))), sim_file))
        return sim_rows, [_task_to_row(task, info, path) for task in tasks]

    rows, tasks = _load_results_payload(payload, path)
    return rows, tasks


def _load_results_payload(payload: dict[str, Any], path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    info = payload.get("info") or {}
    tasks = [task for task in payload.get("tasks", []) if isinstance(task, dict)]
    task_lookup = {str(task.get("id")): task for task in tasks}
    simulations = [sim for sim in payload.get("simulations", []) if isinstance(sim, dict)]
    if not simulations and isinstance(payload.get("simulation_index"), list):
        simulations = [sim for sim in payload["simulation_index"] if isinstance(sim, dict)]
    rows = [_simulation_to_row(sim, info, task_lookup.get(str(sim.get("task_id"))), path) for sim in simulations]
    task_rows = [_task_to_row(task, info, path) for task in tasks]
    return rows, task_rows


def _load_records(records: list[dict[str, Any]], path: Path) -> list[dict[str, Any]]:
    return [_flat_record_to_row(record, path) for record in records if isinstance(record, dict)]


def _load_flat_table(frame: pd.DataFrame, path: Path) -> list[dict[str, Any]]:
    return [_flat_record_to_row(record, path) for record in frame.to_dict(orient="records")]


def _simulation_to_row(
    sim: dict[str, Any],
    info: dict[str, Any],
    task: dict[str, Any] | None,
    result_file: Path,
) -> dict[str, Any]:
    reward_info = sim.get("reward_info") or {}
    reward_basis = _reward_basis(reward_info, task)
    domain = _domain(info, task)
    agent_llm = _nested(info, "agent_info", "llm")
    agent_strategy = _nested(info, "agent_info", "implementation")
    user_llm = _nested(info, "user_info", "llm")
    user_strategy = _nested(info, "user_info", "implementation")
    reward = _number(reward_info.get("reward", sim.get("reward")))
    action_required = "ACTION" in reward_basis
    return {
        "eval_id": TAU_BENCH_EVAL_ID,
        "trial_id": str(sim.get("id") or _trial_id(domain, sim.get("task_id"), agent_llm, agent_strategy, sim.get("trial"))),
        "task_id": _qualified_task_id(domain, sim.get("task_id")),
        "system_id": _system_id(agent_llm, agent_strategy),
        "score": reward,
        "reward": reward,
        "db_reward": _db_reward(reward_info),
        "communicate_reward": _component_reward(reward_info, "COMMUNICATE"),
        "env_assertion_reward": _env_assertion_reward(reward_info),
        "action_reward": _action_reward(reward_info),
        "partial_action_reward": _partial_action_reward(reward_info),
        "pass_at_k": None,
        "source": TAU_BENCH_SOURCE,
        "eval_scope": TAU_BENCH_EVAL_SCOPE,
        "included_in_score": reward is not None and domain in TEXT_MODE_DOMAINS,
        "outcome": _outcome(reward),
        "domain": domain,
        "agent_llm": agent_llm,
        "user_llm": user_llm,
        "agent_strategy": agent_strategy,
        "user_strategy": user_strategy,
        "trial_index": sim.get("trial"),
        "task_split": _task_split(info),
        "reward_basis": "+".join(reward_basis),
        "action_required": action_required,
        "communication_mode": sim.get("mode") or _communication_mode(info),
        "termination_reason": _enum_value(sim.get("termination_reason")),
        "duration": _number(sim.get("duration")),
        "agent_cost": _number(sim.get("agent_cost")),
        "user_cost": _number(sim.get("user_cost")),
        "num_messages": _message_count(sim),
        "tool_call_count": _tool_call_count(sim),
        "result_file": str(result_file),
        "source_version": info.get("git_commit"),
    }


def _flat_record_to_row(record: dict[str, Any], result_file: Path) -> dict[str, Any]:
    reward_basis = _listify(record.get("reward_basis"))
    domain = record.get("domain") or record.get("info_domain")
    agent_llm = record.get("agent_llm") or record.get("info_agent_llm")
    agent_strategy = record.get("agent_strategy") or record.get("info_agent_implementation")
    reward = _number(record.get("reward", record.get("score")))
    row = {column: record.get(column) for column in BASE_COLUMNS}
    row.update(
        {
            "eval_id": TAU_BENCH_EVAL_ID,
            "trial_id": str(record.get("trial_id") or record.get("simulation_id") or _trial_id(domain, record.get("task_id"), agent_llm, agent_strategy, record.get("trial"))),
            "task_id": _qualified_task_id(domain, record.get("task_id")),
            "system_id": record.get("system_id") or _system_id(agent_llm, agent_strategy),
            "score": reward,
            "reward": reward,
            "source": record.get("source", TAU_BENCH_SOURCE),
            "eval_scope": record.get("eval_scope", TAU_BENCH_EVAL_SCOPE),
            "included_in_score": bool(record.get("included_in_score", reward is not None and domain in TEXT_MODE_DOMAINS)),
            "outcome": record.get("outcome") or _outcome(reward),
            "domain": domain,
            "agent_llm": agent_llm,
            "user_llm": record.get("user_llm") or record.get("info_user_llm"),
            "agent_strategy": agent_strategy,
            "user_strategy": record.get("user_strategy") or record.get("info_user_implementation"),
            "trial_index": record.get("trial_index", record.get("trial")),
            "task_split": record.get("task_split", record.get("task_split_name")),
            "reward_basis": "+".join(reward_basis),
            "action_required": "ACTION" in reward_basis,
            "communication_mode": record.get("communication_mode", record.get("mode")),
            "result_file": str(result_file),
            "source_version": record.get("source_version", record.get("info_git_commit")),
        }
    )
    return row


def _task_to_row(task: dict[str, Any], info: dict[str, Any], result_file: Path) -> dict[str, Any]:
    criteria = task.get("evaluation_criteria") or {}
    reward_basis = _listify(criteria.get("reward_basis")) or ["DB", "COMMUNICATE"]
    actions = criteria.get("actions") or []
    env_assertions = criteria.get("env_assertions") or []
    nl_assertions = criteria.get("nl_assertions") or []
    agent_actions = [action for action in actions if (action.get("requestor") or "assistant") == "assistant"]
    user_actions = [action for action in actions if action.get("requestor") == "user"]
    domain = _domain(info, task)
    return {
        "eval_id": TAU_BENCH_EVAL_ID,
        "task_id": _qualified_task_id(domain, task.get("id")),
        "task_name": _qualified_task_id(domain, task.get("id")),
        "domain": domain,
        "task_split": _task_split(info),
        "reward_basis": "+".join(reward_basis),
        "action_required": "ACTION" in reward_basis,
        "task_num_agent_actions": len(agent_actions),
        "task_num_user_actions": len(user_actions),
        "task_num_actions": len(actions),
        "task_num_env_assertions": len(env_assertions),
        "task_num_nl_assertions": len(nl_assertions),
        "result_file": str(result_file),
    }


def _finalize_trials(trials: pd.DataFrame) -> pd.DataFrame:
    for column in [
        "score",
        "reward",
        "db_reward",
        "communicate_reward",
        "env_assertion_reward",
        "action_reward",
        "partial_action_reward",
        "pass_at_k",
        "duration",
        "agent_cost",
        "user_cost",
        "num_messages",
        "tool_call_count",
    ]:
        if column in trials.columns:
            trials[column] = pd.to_numeric(trials[column], errors="coerce")
    trials["included_in_score"] = trials["included_in_score"].fillna(False).astype(bool)
    trials["passed"] = trials["score"].fillna(0).astype(float) >= 1.0
    trials["trial_name"] = trials["trial_id"]
    trials["task_name"] = trials["task_id"]
    trials["model_key"] = trials["system_id"]
    trials["score_value"] = trials["score"]
    return trials


def _finalize_tasks(tasks: pd.DataFrame) -> pd.DataFrame:
    tasks = tasks.drop_duplicates(subset=["task_id"], keep="first").copy()
    return tasks


def _nested(data: dict[str, Any], *keys: str) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return _enum_value(current)


def _enum_value(value: Any) -> Any:
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _listify(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.replace(",", "+").split("+") if part.strip()]
    if isinstance(value, list):
        return [str(_enum_value(item)) for item in value]
    return [str(_enum_value(value))]


def _reward_basis(reward_info: dict[str, Any], task: dict[str, Any] | None) -> list[str]:
    basis = _listify(reward_info.get("reward_basis"))
    if basis:
        return basis
    if task:
        criteria = task.get("evaluation_criteria") or {}
        basis = _listify(criteria.get("reward_basis"))
        if basis:
            return basis
    return ["DB", "COMMUNICATE"]


def _component_reward(reward_info: dict[str, Any], name: str) -> float | None:
    breakdown = reward_info.get("reward_breakdown") or {}
    for key, value in breakdown.items():
        if str(_enum_value(key)) == name:
            return _number(value)
    return None


def _db_reward(reward_info: dict[str, Any]) -> float | None:
    check = reward_info.get("db_check") or {}
    reward = _number(check.get("db_reward"))
    if reward is not None:
        return reward
    return _component_reward(reward_info, "DB")


def _env_assertion_reward(reward_info: dict[str, Any]) -> float | None:
    component = _component_reward(reward_info, "ENV_ASSERTION")
    if component is not None:
        return component
    rewards = [_number(check.get("reward")) for check in reward_info.get("env_assertions") or []]
    rewards = [reward for reward in rewards if reward is not None]
    if not rewards:
        return None
    result = 1.0
    for reward in rewards:
        result *= reward
    return result


def _action_reward(reward_info: dict[str, Any]) -> float | None:
    component = _component_reward(reward_info, "ACTION")
    if component is not None:
        return component
    rewards = [_number(check.get("action_reward")) for check in reward_info.get("action_checks") or []]
    rewards = [reward for reward in rewards if reward is not None]
    if not rewards:
        return None
    result = 1.0
    for reward in rewards:
        result *= reward
    return result


def _partial_action_reward(reward_info: dict[str, Any]) -> float | None:
    checks = [check for check in reward_info.get("action_checks") or [] if isinstance(check, dict)]
    if not checks:
        return None
    matches = [bool(check.get("action_match")) for check in checks]
    return sum(matches) / len(matches)


def _domain(info: dict[str, Any], task: dict[str, Any] | None = None) -> str | None:
    return _nested(info, "environment_info", "domain_name") or (task or {}).get("domain")


def _task_split(info: dict[str, Any]) -> Any:
    return info.get("task_split_name") or info.get("task_split") or "base"


def _communication_mode(info: dict[str, Any]) -> str:
    return "full_duplex" if info.get("audio_native_config") else "half_duplex"


def _qualified_task_id(domain: Any, task_id: Any) -> str:
    if domain is None or task_id is None:
        return str(task_id)
    task = str(task_id)
    prefix = f"{domain}:"
    return task if task.startswith(prefix) else f"{prefix}{task}"


def _system_id(agent_llm: Any, agent_strategy: Any) -> str:
    parts = [str(part) for part in [agent_llm, agent_strategy] if part not in (None, "", "llm_agent")]
    return " | ".join(parts) if parts else "unknown"


def _trial_id(domain: Any, task_id: Any, agent_llm: Any, agent_strategy: Any, trial: Any) -> str:
    return "::".join(str(part) for part in [_qualified_task_id(domain, task_id), _system_id(agent_llm, agent_strategy), trial] if part is not None)


def _outcome(reward: float | None) -> str:
    if reward is None:
        return "missing"
    return "pass" if reward >= 1.0 else "fail"


def _message_count(sim: dict[str, Any]) -> int | None:
    messages = sim.get("messages")
    if isinstance(messages, list):
        return len(messages)
    ticks = sim.get("ticks")
    if isinstance(ticks, list):
        return len(ticks)
    return None


def _tool_call_count(sim: dict[str, Any]) -> int | None:
    messages = sim.get("messages")
    if not isinstance(messages, list):
        return None
    count = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            count += len(tool_calls)
    return count
