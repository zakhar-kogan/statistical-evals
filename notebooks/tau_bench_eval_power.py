import marimo

__generated_with = "0.23.11"
app = marimo.App(width="full")


@app.cell
def _():
    from dataclasses import dataclass
    from pathlib import Path
    import json
    import os

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import requests
    import seaborn as sns

    ROOT = Path(__file__).resolve().parents[1]
    CACHE_ROOT = ROOT / ".cache" / "tau_bench_eval_power"
    GITHUB_API = "https://api.github.com/repos/sierra-research/tau2-bench/contents/data/tau2/results/final"
    GITHUB_RAW_BASE = "https://raw.githubusercontent.com/sierra-research/tau2-bench/main/data/tau2/results/final"
    RNG_SEED = 42
    N_BOOT = 800
    PRACTICAL_EQUIVALENCE_PP = 2.0
    MIN_SLICE_TASKS = 2

    def _enum_value(value):
        if isinstance(value, dict) and "value" in value:
            return value["value"]
        return value

    def _number(value):
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _listify(value):
        if value in (None, ""):
            return []
        if isinstance(value, str):
            return [part.strip() for part in value.replace(",", "+").split("+") if part.strip()]
        if isinstance(value, list):
            return [str(_enum_value(item)) for item in value]
        return [str(_enum_value(value))]

    def _nested(data, *keys):
        _current = data
        for key in keys:
            if not isinstance(_current, dict):
                return None
            _current = _current.get(key)
        return _enum_value(_current)

    def _qualified_task_id(domain, task_id):
        if domain is None or task_id is None:
            return str(task_id)
        _task = str(task_id)
        _prefix = f"{domain}:"
        return _task if _task.startswith(_prefix) else f"{_prefix}{_task}"

    def _system_id(agent_llm, agent_strategy):
        _parts = [str(part) for part in [agent_llm, agent_strategy] if part not in (None, "", "llm_agent")]
        return " | ".join(_parts) if _parts else "unknown"

    def _reward_basis(reward_info, task):
        _basis = _listify((reward_info or {}).get("reward_basis"))
        if _basis:
            return _basis
        _criteria = (task or {}).get("evaluation_criteria") or {}
        _basis = _listify(_criteria.get("reward_basis"))
        return _basis or ["DB", "COMMUNICATE"]

    def _component_reward(reward_info, name):
        for _key, _value in ((reward_info or {}).get("reward_breakdown") or {}).items():
            if str(_enum_value(_key)) == name:
                return _number(_value)
        return None

    def _db_reward(reward_info):
        _check = (reward_info or {}).get("db_check") or {}
        _reward = _number(_check.get("db_reward"))
        return _reward if _reward is not None else _component_reward(reward_info, "DB")

    def _product(values):
        _present = [value for value in values if value is not None]
        if not _present:
            return None
        _out = 1.0
        for value in _present:
            _out *= value
        return _out

    def _env_assertion_reward(reward_info):
        _component = _component_reward(reward_info, "ENV_ASSERTION")
        if _component is not None:
            return _component
        return _product([_number(check.get("reward")) for check in (reward_info or {}).get("env_assertions") or []])

    def _action_reward(reward_info):
        _component = _component_reward(reward_info, "ACTION")
        if _component is not None:
            return _component
        return _product([_number(check.get("action_reward")) for check in (reward_info or {}).get("action_checks") or []])

    def _partial_action_reward(reward_info):
        _checks = [check for check in (reward_info or {}).get("action_checks") or [] if isinstance(check, dict)]
        if not _checks:
            return None
        return sum(bool(check.get("action_match")) for check in _checks) / len(_checks)

    def _task_to_row(task, info, result_file):
        _criteria = task.get("evaluation_criteria") or {}
        _basis = _listify(_criteria.get("reward_basis")) or ["DB", "COMMUNICATE"]
        _actions = _criteria.get("actions") or []
        _domain = _nested(info, "environment_info", "domain_name") or task.get("domain")
        return {
            "task_id": _qualified_task_id(_domain, task.get("id")),
            "task_name": _qualified_task_id(_domain, task.get("id")),
            "domain": _domain,
            "task_split": info.get("task_split_name") or info.get("task_split") or "base",
            "reward_basis": "+".join(_basis),
            "action_required": "ACTION" in _basis,
            "task_num_actions": len(_actions),
            "task_num_agent_actions": len([action for action in _actions if (action.get("requestor") or "assistant") == "assistant"]),
            "task_num_user_actions": len([action for action in _actions if action.get("requestor") == "user"]),
            "task_num_env_assertions": len(_criteria.get("env_assertions") or []),
            "task_num_nl_assertions": len(_criteria.get("nl_assertions") or []),
            "result_file": str(result_file),
        }

    def _message_count(sim):
        _messages = sim.get("messages")
        if isinstance(_messages, list):
            return len(_messages)
        _ticks = sim.get("ticks")
        if isinstance(_ticks, list):
            return len(_ticks)
        return None

    def _tool_call_count(sim):
        _messages = sim.get("messages")
        if not isinstance(_messages, list):
            return None
        _count = 0
        for _message in _messages:
            if isinstance(_message, dict) and isinstance(_message.get("tool_calls"), list):
                _count += len(_message["tool_calls"])
        return _count

    def _sim_to_row(sim, info, task, result_file):
        _reward_info = sim.get("reward_info") or {}
        _basis = _reward_basis(_reward_info, task)
        _domain = _nested(info, "environment_info", "domain_name") or (task or {}).get("domain")
        _agent_llm = _nested(info, "agent_info", "llm")
        _agent_strategy = _nested(info, "agent_info", "implementation")
        _user_llm = _nested(info, "user_info", "llm")
        _user_strategy = _nested(info, "user_info", "implementation")
        _reward = _number(_reward_info.get("reward", sim.get("reward")))
        _run_id = str(sim.get("id") or f"{_domain}:{sim.get('task_id')}:{_agent_llm}:{sim.get('trial')}")
        return {
            "task_id": _qualified_task_id(_domain, sim.get("task_id")),
            "task_name": _qualified_task_id(_domain, sim.get("task_id")),
            "trial_id": _run_id,
            "run_id": _run_id,
            "system_id": _system_id(_agent_llm, _agent_strategy),
            "score": _reward,
            "reward": _reward,
            "score_value": _reward,
            "passed": bool(_reward is not None and _reward >= 1.0),
            "outcome": "missing" if _reward is None else ("pass" if _reward >= 1.0 else "fail"),
            "domain": _domain,
            "agent_llm": _agent_llm,
            "user_llm": _user_llm,
            "agent_strategy": _agent_strategy,
            "user_strategy": _user_strategy,
            "trial_index": sim.get("trial"),
            "seed": sim.get("seed"),
            "task_split": info.get("task_split_name") or info.get("task_split") or "base",
            "reward_basis": "+".join(_basis),
            "action_required": "ACTION" in _basis,
            "db_reward": _db_reward(_reward_info),
            "communicate_reward": _component_reward(_reward_info, "COMMUNICATE"),
            "env_assertion_reward": _env_assertion_reward(_reward_info),
            "action_reward": _action_reward(_reward_info),
            "partial_action_reward": _partial_action_reward(_reward_info),
            "communication_mode": sim.get("mode") or ("full_duplex" if info.get("audio_native_config") else "half_duplex"),
            "termination_reason": _enum_value(sim.get("termination_reason")),
            "duration": _number(sim.get("duration")),
            "agent_cost": _number(sim.get("agent_cost")),
            "user_cost": _number(sim.get("user_cost")),
            "num_messages": _message_count(sim),
            "tool_call_count": _tool_call_count(sim),
            "source": "tau2-bench",
            "eval_scope": "tau3-current",
            "included_in_score": _domain in {"airline", "retail", "telecom", "telecom-workflow"} and _reward is not None,
            "source_version": info.get("git_commit"),
            "result_file": str(result_file),
        }

    def _flat_record_to_row(record, result_file):
        _basis = _listify(record.get("reward_basis"))
        _domain = record.get("domain") or record.get("info_domain")
        _agent_llm = record.get("agent_llm") or record.get("info_agent_llm")
        _agent_strategy = record.get("agent_strategy") or record.get("info_agent_implementation")
        _reward = _number(record.get("reward", record.get("score")))
        _run_id = str(record.get("run_id") or record.get("trial_id") or record.get("simulation_id") or f"{_domain}:{record.get('task_id')}:{_agent_llm}:{record.get('trial')}")
        _row = dict(record)
        _row.update(
            {
                "task_id": _qualified_task_id(_domain, record.get("task_id")),
                "task_name": _qualified_task_id(_domain, record.get("task_id")),
                "trial_id": _run_id,
                "run_id": _run_id,
                "system_id": record.get("system_id") or _system_id(_agent_llm, _agent_strategy),
                "score": _reward,
                "reward": _reward,
                "score_value": _reward,
                "passed": bool(_reward is not None and _reward >= 1.0),
                "outcome": record.get("outcome") or ("missing" if _reward is None else ("pass" if _reward >= 1.0 else "fail")),
                "domain": _domain,
                "agent_llm": _agent_llm,
                "user_llm": record.get("user_llm") or record.get("info_user_llm"),
                "agent_strategy": _agent_strategy,
                "user_strategy": record.get("user_strategy") or record.get("info_user_implementation"),
                "trial_index": record.get("trial_index", record.get("trial")),
                "task_split": record.get("task_split", record.get("task_split_name", "base")),
                "reward_basis": "+".join(_basis),
                "action_required": "ACTION" in _basis,
                "source": record.get("source", "tau2-bench"),
                "eval_scope": record.get("eval_scope", "tau3-current"),
                "included_in_score": bool(record.get("included_in_score", _reward is not None and _domain in {"airline", "retail", "telecom", "telecom-workflow"})),
                "result_file": str(result_file),
            }
        )
        return _row

    def _load_results_payload(payload, result_file):
        _info = payload.get("info") or {}
        _tasks = [task for task in payload.get("tasks", []) if isinstance(task, dict)]
        _task_lookup = {str(task.get("id")): task for task in _tasks}
        _simulations = [sim for sim in payload.get("simulations", []) if isinstance(sim, dict)]
        if not _simulations and isinstance(payload.get("simulation_index"), list):
            _simulations = [sim for sim in payload["simulation_index"] if isinstance(sim, dict)]
        _rows = [_sim_to_row(sim, _info, _task_lookup.get(str(sim.get("task_id"))), result_file) for sim in _simulations]
        _task_rows = [_task_to_row(task, _info, result_file) for task in _tasks]
        return _rows, _task_rows

    def _load_one_path(path):
        _path = path.expanduser()
        if _path.is_dir():
            _result_file = _path / "results.json"
            if _result_file.exists() and (_path / "simulations").is_dir():
                _payload = json.loads(_result_file.read_text(encoding="utf-8"))
                _info = _payload.get("info") or {}
                _tasks = [task for task in _payload.get("tasks", []) if isinstance(task, dict)]
                _task_lookup = {str(task.get("id")): task for task in _tasks}
                _rows = []
                for _sim_file in sorted((_path / "simulations").glob("*.json")):
                    _sim = json.loads(_sim_file.read_text(encoding="utf-8"))
                    _rows.append(_sim_to_row(_sim, _info, _task_lookup.get(str(_sim.get("task_id"))), _sim_file))
                return _rows, [_task_to_row(task, _info, _result_file) for task in _tasks]
            if _result_file.exists():
                return _load_one_path(_result_file)
            _rows, _tasks = [], []
            for _child in sorted(_path.iterdir()):
                if _child.is_dir() or _child.suffix.lower() in {".json", ".jsonl", ".csv"}:
                    _child_rows, _child_tasks = _load_one_path(_child)
                    _rows.extend(_child_rows)
                    _tasks.extend(_child_tasks)
            return _rows, _tasks
        if _path.suffix.lower() == ".csv":
            _frame = pd.read_csv(_path)
            return [_flat_record_to_row(row, _path) for row in _frame.to_dict(orient="records")], []
        if _path.suffix.lower() == ".jsonl":
            _records = [json.loads(line) for line in _path.read_text(encoding="utf-8").splitlines() if line.strip()]
            return [_flat_record_to_row(row, _path) for row in _records if isinstance(row, dict)], []
        if _path.suffix.lower() == ".json":
            _payload = json.loads(_path.read_text(encoding="utf-8"))
            if isinstance(_payload, list):
                return [_flat_record_to_row(row, _path) for row in _payload if isinstance(row, dict)], []
            if isinstance(_payload, dict) and ("simulations" in _payload or "simulation_index" in _payload):
                return _load_results_payload(_payload, _path)
            if isinstance(_payload, dict):
                for _key in ("rows", "data", "results"):
                    if isinstance(_payload.get(_key), list):
                        return [_flat_record_to_row(row, _path) for row in _payload[_key] if isinstance(row, dict)], []
                return [_flat_record_to_row(_payload, _path)], []
        return [], []

    def load_tau_bench_paths(paths):
        _rows, _task_rows = [], []
        for _path in paths:
            _loaded_rows, _loaded_tasks = _load_one_path(_path)
            _rows.extend(_loaded_rows)
            _task_rows.extend(_loaded_tasks)
        _trials = pd.DataFrame(_rows)
        _tasks = pd.DataFrame(_task_rows)
        if not _trials.empty:
            for _column in [
                "score",
                "reward",
                "score_value",
                "db_reward",
                "communicate_reward",
                "env_assertion_reward",
                "action_reward",
                "partial_action_reward",
                "duration",
                "agent_cost",
                "user_cost",
                "num_messages",
                "tool_call_count",
            ]:
                if _column in _trials.columns:
                    _trials[_column] = pd.to_numeric(_trials[_column], errors="coerce")
        if not _tasks.empty and "task_id" in _tasks.columns:
            _tasks = _tasks.drop_duplicates("task_id", keep="first")
        return _trials, _tasks

    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["figure.dpi"] = 130
    return (
        CACHE_ROOT,
        GITHUB_API,
        GITHUB_RAW_BASE,
        N_BOOT,
        PRACTICAL_EQUIVALENCE_PP,
        Path,
        RNG_SEED,
        ROOT,
        dataclass,
        load_tau_bench_paths,
        mo,
        np,
        os,
        pd,
        plt,
        requests,
        sns,
    )


@app.cell
def _(mo):
    mo.md("""
    # τ-Bench Eval Power / Rank Resolution Audit

    This notebook treats τ²/τ³-bench as a **stateful agent reliability benchmark**, not just a point leaderboard.

    The default ranked unit is the evaluated agent system: agent LLM + agent implementation over a domain-specific user simulator and environment.

    ## How to read this notebook

    1. **Define the measured system:** model, agent scaffold, user simulator, tools/environment, and task domain.
    2. **Respect τ-bench scoring:** official reward is gated by `reward_basis`; `actions` are diagnostic unless `ACTION` is explicitly included.
    3. **Run only supported methods:** task bootstrap, paired resolution, slice analysis, repeated-trial diagnostics, influence, and operational profile.
    4. **Report caveats:** adjacent ranks are often over-resolved; simulator and domain effects are part of the measured system.
    """)
    return


@app.cell
def _(mo):
    data_source = mo.ui.dropdown(options=["auto", "fixture"], value="auto", label="Data source")
    rank_unit = mo.ui.dropdown(options=["system_id", "agent_llm", "agent_strategy"], value="system_id", label="Ranked unit")
    slice_dimension = mo.ui.dropdown(options=["domain", "reward_basis", "agent_strategy", "user_llm", "communication_mode"], value="domain", label="Slice dimension")
    slice_min_tasks = mo.ui.slider(1, 10, value=2, step=1, show_value=True, label="Min tasks per slice")
    top_n = mo.ui.slider(5, 25, value=15, step=1, show_value=True, label="Top N")
    top_k = mo.ui.slider(3, 15, value=10, step=1, show_value=True, label="Top K")
    mo.vstack(
        [
            mo.md("## Setup And Data / Eval Universe"),
            mo.hstack([data_source, rank_unit, slice_dimension, slice_min_tasks, top_n, top_k], justify="start", gap=1),
        ]
    )
    return (
        data_source,
        rank_unit,
        slice_dimension,
        slice_min_tasks,
        top_k,
        top_n,
    )


@app.cell
def _(dataclass):
    @dataclass(frozen=True)
    class EvalSpec:
        name: str
        task_col: str
        system_col: str
        run_col: str
        score_col: str
        system_factor_cols: tuple[str, ...] = ()
        dimension_cols: tuple[str, ...] = ()
        reliability_cols: tuple[str, ...] = ()
        simulator_cols: tuple[str, ...] = ()
        trajectory_cols: tuple[str, ...] = ()

        @property
        def required_cols(self):
            return (self.task_col, self.system_col, self.run_col, self.score_col)

    TAU_SPEC = EvalSpec(
        name="τ-Bench / τ² current",
        task_col="task_id",
        system_col="system_id",
        run_col="run_id",
        score_col="score",
        system_factor_cols=("system_id", "agent_llm", "agent_strategy"),
        dimension_cols=("domain", "reward_basis", "agent_strategy", "user_llm", "communication_mode"),
        reliability_cols=("duration", "agent_cost", "user_cost", "num_messages", "tool_call_count"),
        simulator_cols=("user_llm", "user_strategy"),
        trajectory_cols=("partial_action_reward", "action_reward", "tool_call_count"),
    )
    return (TAU_SPEC,)


@app.cell
def _(
    CACHE_ROOT,
    GITHUB_API,
    GITHUB_RAW_BASE,
    Path,
    ROOT,
    data_source,
    load_tau_bench_paths,
    mo,
    os,
    pd,
    requests,
):
    def _env_paths():
        _value = os.environ.get("TAU_BENCH_RESULTS")
        if not _value:
            return []
        return [ROOT.joinpath(part).resolve() if not os.path.isabs(part) else Path(part).expanduser() for part in _value.split(os.pathsep) if part]

    from pathlib import Path as _Path

    def _fixture_paths():
        return [ROOT / "tests" / "fixtures" / "tau_bench_results.csv"]

    def _fetch_public_result_paths():
        _cache_dir = CACHE_ROOT / "results" / "final"
        _cache_dir.mkdir(parents=True, exist_ok=True)
        _response = requests.get(GITHUB_API, timeout=60)
        _response.raise_for_status()
        _paths = []
        for _entry in _response.json():
            _name = _entry.get("name", "")
            if not _name.endswith(".json"):
                continue
            _path = _cache_dir / _name
            if not _path.exists():
                _url = _entry.get("download_url") or f"{GITHUB_RAW_BASE}/{_name}"
                _payload = requests.get(_url, timeout=120)
                _payload.raise_for_status()
                _path.write_text(_payload.text, encoding="utf-8")
            _paths.append(_path)
        return _paths

    try:
        if data_source.value == "fixture":
            selected_paths = _fixture_paths()
            status_kind = "fixture"
        else:
            selected_paths = _env_paths()
            status_kind = "env"
            if not selected_paths:
                selected_paths = _fetch_public_result_paths()
                status_kind = "public-github-cache"
        trials, tasks = load_tau_bench_paths(selected_paths)
        data_error = None
    except Exception as _exc:
        selected_paths = _env_paths() if data_source.value == "auto" else _fixture_paths()
        trials, tasks = pd.DataFrame(), pd.DataFrame()
        data_error = f"{type(_exc).__name__}: {_exc}"
        status_kind = "failed"

    data_status = (
        mo.callout(f"Could not load τ-bench results: `{data_error}`", kind="danger")
        if data_error
        else mo.md(f"Loaded τ-bench results from `{status_kind}`: `{len(selected_paths)}` file/path(s).")
    )
    data_status
    return data_error, selected_paths, status_kind, tasks, trials


@app.cell
def _(mo, pd, selected_paths, status_kind, tasks, trials):
    data_summary = pd.DataFrame(
        [
            {
                "rows": len(trials),
                "tasks": trials["task_id"].nunique() if "task_id" in trials else 0,
                "systems": trials["system_id"].nunique() if "system_id" in trials else 0,
                "domains": trials["domain"].nunique() if "domain" in trials else 0,
                "agent_llms": trials["agent_llm"].nunique() if "agent_llm" in trials else 0,
                "user_llms": trials["user_llm"].nunique() if "user_llm" in trials else 0,
                "task_metadata_rows": len(tasks),
                "source_kind": status_kind,
                "paths": len(selected_paths),
            }
        ]
    )
    path_status = pd.DataFrame({"path": [str(path) for path in selected_paths]})
    mo.vstack([mo.ui.table(data_summary, pagination=False), mo.accordion({"Loaded paths": mo.ui.table(path_status, pagination=True, page_size=8)})])
    return (data_summary,)


@app.cell
def _(TAU_SPEC, data_error, mo, np, pd, plt, sns, trials):
    def _has_cols(*columns):
        return (not data_error) and (not trials.empty) and all(column in trials.columns and trials[column].notna().any() for column in columns)

    def _availability(status):
        return {"yes": "available", "partial": "partial", "no": "unavailable"}[status]

    n_tasks = int(trials["task_id"].nunique()) if _has_cols("task_id") else 0
    n_systems = int(trials["system_id"].nunique()) if _has_cols("system_id") else 0
    n_runs = int(trials[TAU_SPEC.run_col].nunique()) if _has_cols(TAU_SPEC.run_col) else 0
    _repeated_cell_count = 0
    _coverage_pct = np.nan
    if _has_cols("task_id", "system_id", TAU_SPEC.run_col):
        _cell_counts = trials.groupby(["task_id", "system_id"], dropna=False)[TAU_SPEC.run_col].size()
        _repeated_cell_count = int((_cell_counts > 1).sum())
        _possible_cells = n_tasks * n_systems
        _coverage_pct = 100 * _cell_counts.size / _possible_cells if _possible_cells else np.nan
    _dimensions = [column for column in TAU_SPEC.dimension_cols if _has_cols(column)]
    _reliability = [column for column in TAU_SPEC.reliability_cols if _has_cols(column)]
    _simulator = [column for column in TAU_SPEC.simulator_cols if _has_cols(column)]
    _trajectory = [column for column in TAU_SPEC.trajectory_cols if _has_cols(column)]
    _action_required_tasks = int(trials.loc[trials.get("action_required", False).fillna(False).astype(bool), "task_id"].nunique()) if "action_required" in trials else 0

    evidence_counts = pd.DataFrame(
        [
            ("tasks", n_tasks),
            ("systems", n_systems),
            ("runs", n_runs),
            ("repeated system-task cells", _repeated_cell_count),
            ("system-task coverage %", _coverage_pct),
            ("available dimensions", len(_dimensions)),
            ("available reliability fields", len(_reliability)),
            ("available simulator fields", len(_simulator)),
            ("available trajectory diagnostics", len(_trajectory)),
            ("tasks where ACTION gates reward", _action_required_tasks),
        ],
        columns=["evidence", "value"],
    )
    taxonomy = pd.DataFrame(
        [
            ("S", "evaluated system", "agent LLM + agent implementation under τ-bench orchestration"),
            ("M", "model", "agent_llm"),
            ("H", "harness/scaffold", "agent_strategy plus τ-bench orchestrator"),
            ("E", "environment", "domain environment, tools, DB, user simulator"),
            ("B", "benchmark", "τ²/τ³ current"),
            ("C", "domain/cluster", "airline, retail, telecom, telecom-workflow"),
            ("T", "task", "task_id"),
            ("R", "rollout/run", "trial_id / trial_index"),
            ("trajectory", "tool/message path", "messages/tool_call_count/action diagnostics when present"),
            ("e", "event", "tool calls/messages; only coarse counts parsed here"),
        ],
        columns=["Symbol", "Meaning", "τ-bench mapping"],
    )
    capability_matrix = pd.DataFrame(
        [
            ("Observed leaderboard", "observed reward by ranked system", f"{n_tasks} tasks; {n_systems} systems", _availability("yes" if _has_cols("task_id", "system_id", "score") else "no"), "run"),
            ("Nearby-rank resolution", "paired task deltas and bootstrap CIs", f"{_coverage_pct:.1f}% system-task coverage", _availability("yes" if _has_cols("task_id", "system_id", "score") else "no"), "run"),
            ("Rank stability", "task bootstrap rank intervals and top-K probabilities", f"{n_tasks} task units", _availability("yes" if _has_cols("task_id", "system_id", "score") else "no"), "run"),
            ("Domain heterogeneity", "slice leaderboards by domain/reward basis", ", ".join(_dimensions), _availability("yes" if _dimensions else "no"), "run" if _dimensions else "roadmap"),
            ("Repeated-trial reliability", "within-task repeated trial spread and pass@k approximation", f"{_repeated_cell_count} repeated cells", _availability("yes" if _repeated_cell_count else "partial"), "run with caveat"),
            ("Operational profile", "score vs cost/duration/messages/tool calls", ", ".join(_reliability), _availability("yes" if _reliability else "no"), "run" if _reliability else "roadmap"),
            ("Action semantics", "distinguish reward-gating ACTION from diagnostic actions", f"{_action_required_tasks} ACTION-gated tasks", _availability("yes" if _has_cols("reward_basis") else "no"), "run"),
            ("Simulator effects", "compare user_llm/user_strategy slices", ", ".join(_simulator), _availability("yes" if _simulator else "partial"), "run with caveat"),
        ],
        columns=["Claim/question", "Method", "τ-bench evidence", "Support", "Action"],
    )
    fig_avail, _ax_avail = plt.subplots(figsize=(9, 4.2))
    _heat = pd.DataFrame(
        [
            ("leaderboard", 1, 1, 0, 0, 0, 0),
            ("resolution", 1, 1, 0, 0, 0, 0),
            ("stability", 1, 1, 0, 0, 0, 0),
            ("heterogeneity", 1, 1, 1, 0, 0, 0),
            ("repeats/pass@k", 1, 1, 0, 1 if _repeated_cell_count else 0.5, 0, 0),
            ("operations", 1, 1, 0, 0, 1 if _reliability else 0, 0),
            ("trajectory", 1, 1, 0, 0, 0, 1 if _trajectory else 0),
        ],
        columns=["question", "outcomes", "tasks/systems", "dimensions", "repeats", "ops", "trajectory"],
    ).set_index("question")
    sns.heatmap(_heat, cmap=sns.color_palette(["#F2F2F2", "#F6C85F", "#4C78A8"]), vmin=0, vmax=1, cbar=False, linewidths=0.5, linecolor="white", ax=_ax_avail)
    _ax_avail.set_title("Evidence availability by question")
    fig_avail.tight_layout()
    mo.vstack(
        [
            mo.md("## Taxonomy And Capability Matrix"),
            mo.md("τ-bench measures an agent in a user-simulator plus tool/environment loop. Domain, simulator, tool state, and reward basis are part of the measurement instrument."),
            mo.ui.table(taxonomy, pagination=False),
            mo.ui.table(evidence_counts, pagination=False),
            mo.mpl.interactive(fig_avail),
            mo.ui.table(capability_matrix, pagination=False),
        ]
    )
    return capability_matrix, evidence_counts


@app.cell
def _(mo, rank_unit):
    mo.md(
        f"""
    ## Estimand

    This notebook separates three targets:

    - **Fixed leaderboard description:** what happened on the observed τ-bench result files.
    - **Task-population inference:** what might change if similar tasks from the same domains were resampled.
    - **System comparison:** the selected ranked unit is `{rank_unit.value}`.

    τ-bench rewards are outcome/state based by default. For airline, retail, and telecom, `reward_basis` is normally `DB + COMMUNICATE`: the agent must leave the environment DB in the correct state and communicate required information. `evaluation_criteria.actions` records one reference path for deriving or diagnosing behavior; it is not a hard trajectory requirement unless `ACTION` is in `reward_basis`.
    """
    )
    return


@app.cell
def _(N_BOOT, RNG_SEED, np, pd, rank_unit, top_k, trials):
    _selected_unit = rank_unit.value
    _selected_top_k = int(top_k.value)

    def analysis_frame(frame, unit=_selected_unit):
        _out = frame[frame.get("included_in_score", True).fillna(True).astype(bool)].copy() if not frame.empty else frame.copy()
        if unit != "system_id" and unit in _out.columns:
            _out["system_id"] = _out[unit].fillna("unknown").astype(str)
        _out["system_label"] = _out["system_id"].astype(str) if "system_id" in _out else []
        return _out

    def aggregate_task_system_scores(frame, score_col="score"):
        return (
            frame.groupby(["task_id", "system_id"], dropna=False)
            .agg(score=(score_col, "mean"), pass_rate=("passed", "mean"), n_trials=("run_id", "count"))
            .reset_index()
        )

    def score_matrix(frame, score_col="score"):
        _agg = aggregate_task_system_scores(frame, score_col=score_col)
        return _agg.pivot_table(index="task_id", columns="system_id", values="score", aggfunc="mean", observed=False).sort_index().sort_index(axis=1)

    def rank_scores(scores):
        return scores.rank(method="min", ascending=False, na_option="bottom").astype("Int64")

    def observed_leaderboard(matrix):
        _scores = matrix.mean(axis=0, skipna=True)
        _ranks = rank_scores(_scores)
        return (
            pd.DataFrame({"system_id": _scores.index, "observed_score": _scores.values, "observed_rank": _ranks.values})
            .sort_values(["observed_rank", "observed_score", "system_id"], ascending=[True, False, True])
            .reset_index(drop=True)
        )

    def bootstrap_rank_stability(frame, draws=N_BOOT, seed=RNG_SEED):
        _matrix = score_matrix(frame)
        if _matrix.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        _values = _matrix.to_numpy(dtype=float)
        _task_count, _system_count = _values.shape
        _rng = np.random.default_rng(seed)
        _sampled = _rng.integers(0, _task_count, size=(draws, _task_count))
        _boot_scores = np.empty((draws, _system_count), dtype=float)
        for _draw_index, _task_indices in enumerate(_sampled):
            _values_i = _values[_task_indices, :]
            _valid = np.sum(~np.isnan(_values_i), axis=0)
            _sums = np.nansum(_values_i, axis=0)
            _boot_scores[_draw_index] = np.divide(_sums, _valid, out=np.full(_system_count, np.nan), where=_valid > 0)
        _score_frame = pd.DataFrame(_boot_scores, columns=_matrix.columns)
        _rank_frame = _score_frame.rank(axis=1, method="min", ascending=False, na_option="bottom").astype("Int64")
        _leaderboard = observed_leaderboard(_matrix)
        _rank_summary = _rank_frame.astype(float).quantile([0.05, 0.25, 0.5, 0.75, 0.95]).T.reset_index().rename(columns={"index": "system_id", 0.05: "rank_p05", 0.25: "rank_p25", 0.5: "rank_p50", 0.75: "rank_p75", 0.95: "rank_p95"})
        _score_summary = _score_frame.quantile([0.05, 0.5, 0.95]).T.reset_index().rename(columns={"index": "system_id", 0.05: "score_p05", 0.5: "score_p50", 0.95: "score_p95"})
        _top_probs = pd.DataFrame(
            {
                "system_id": _rank_frame.columns,
                "top1_probability": (_rank_frame <= 1).mean(axis=0).values,
                "top3_probability": (_rank_frame <= 3).mean(axis=0).values,
                f"top{_selected_top_k}_probability": (_rank_frame <= _selected_top_k).mean(axis=0).values,
            }
        )
        _leaderboard = _leaderboard.merge(_rank_summary, on="system_id").merge(_score_summary, on="system_id").merge(_top_probs, on="system_id")
        _leaderboard["rank_interval_width"] = _leaderboard["rank_p95"] - _leaderboard["rank_p05"]
        _distribution = _rank_frame.melt(var_name="system_id", value_name="rank").groupby(["system_id", "rank"], dropna=False).size().reset_index(name="count")
        _distribution["probability"] = _distribution["count"] / draws
        _pairwise = pd.DataFrame(index=_score_frame.columns, columns=_score_frame.columns, dtype=float)
        for _left in _score_frame.columns:
            for _right in _score_frame.columns:
                _pairwise.loc[_left, _right] = float((_score_frame[_left] >= _score_frame[_right]).mean())
        return _leaderboard, _distribution, _pairwise

    analysis_trials = analysis_frame(trials)
    leaderboard, rank_distribution, pairwise_win_probability = bootstrap_rank_stability(analysis_trials)
    return (
        analysis_trials,
        bootstrap_rank_stability,
        leaderboard,
        observed_leaderboard,
        score_matrix,
    )


@app.cell
def _(leaderboard, mo, plt, sns, top_n):
    _selected_top_n = int(top_n.value)
    if leaderboard.empty:
        _leaderboard_view = mo.callout("No leaderboard rows are available after filtering.", kind="warn")
    else:
        _leaderboard_top = leaderboard.head(_selected_top_n).copy()
        _leaderboard_fig, _leaderboard_ax = plt.subplots(figsize=(9, max(4, 0.35 * len(_leaderboard_top))))
        sns.barplot(data=_leaderboard_top, y="system_id", x="observed_score", hue="system_id", dodge=False, legend=False, ax=_leaderboard_ax)
        _leaderboard_ax.errorbar(
            x=_leaderboard_top["score_p50"],
            y=range(len(_leaderboard_top)),
            xerr=[_leaderboard_top["score_p50"] - _leaderboard_top["score_p05"], _leaderboard_top["score_p95"] - _leaderboard_top["score_p50"]],
            fmt="none",
            color="black",
            capsize=3,
        )
        _leaderboard_ax.set_title("Observed reward with bootstrap score interval")
        _leaderboard_ax.set_xlabel("Mean reward")
        _leaderboard_ax.set_ylabel("")
        _leaderboard_fig.tight_layout()
        _leaderboard_view = mo.vstack([mo.mpl.interactive(_leaderboard_fig), mo.ui.table(_leaderboard_top, pagination=False)])
    _leaderboard_view
    return


@app.cell
def _(leaderboard, mo, plt, top_n):
    _rank_selected_top_n = int(top_n.value)
    if leaderboard.empty:
        _rank_view = mo.md("")
    else:
        _rank_top = leaderboard.head(_rank_selected_top_n).copy()
        _rank_fig, _rank_ax = plt.subplots(figsize=(9, max(4, 0.35 * len(_rank_top))))
        _rank_y = range(len(_rank_top))
        _rank_ax.errorbar(
            _rank_top["rank_p50"],
            _rank_y,
            xerr=[_rank_top["rank_p50"] - _rank_top["rank_p05"], _rank_top["rank_p95"] - _rank_top["rank_p50"]],
            fmt="o",
            capsize=3,
            color="#4C78A8",
        )
        _rank_ax.scatter(_rank_top["observed_rank"], _rank_y, marker="x", color="#D62728", label="observed rank")
        _rank_ax.set_yticks(list(_rank_y), labels=_rank_top["system_id"])
        _rank_ax.invert_yaxis()
        _rank_ax.set_xlabel("Rank, rank 1 is best")
        _rank_ax.set_title("Bootstrap rank intervals")
        _rank_ax.legend()
        _rank_fig.tight_layout()
        _rank_view = mo.mpl.interactive(_rank_fig)
    _rank_view
    return


@app.cell
def _(
    PRACTICAL_EQUIVALENCE_PP,
    analysis_trials,
    leaderboard,
    mo,
    np,
    pd,
    score_matrix,
    top_n,
):
    def paired_resolution(frame, systems):
        _matrix = score_matrix(frame)
        _rows = []
        _ordered = [system for system in systems if system in _matrix.columns]
        for _left, _right in zip(_ordered[:-1], _ordered[1:], strict=False):
            _paired = _matrix[[_left, _right]].dropna()
            if _paired.empty:
                _rows.append({"left": _left, "right": _right, "n_tasks": 0, "gap": np.nan, "ci05": np.nan, "ci95": np.nan, "separated": False})
                continue
            _deltas = _paired[_left] - _paired[_right]
            _rng = np.random.default_rng(123)
            _boot = [_deltas.iloc[_rng.integers(0, len(_deltas), size=len(_deltas))].mean() for _ in range(1000)]
            _rows.append(
                {
                    "left": _left,
                    "right": _right,
                    "n_tasks": len(_deltas),
                    "gap": _deltas.mean(),
                    "ci05": np.quantile(_boot, 0.05),
                    "ci95": np.quantile(_boot, 0.95),
                    "inside_practical_band": abs(_deltas.mean()) <= PRACTICAL_EQUIVALENCE_PP / 100,
                    "separated": np.quantile(_boot, 0.05) > 0 or np.quantile(_boot, 0.95) < 0,
                }
            )
        return pd.DataFrame(_rows)

    if leaderboard.empty:
        adjacent_resolution = pd.DataFrame()
        _resolution_view = mo.md("")
    else:
        _resolution_systems = leaderboard.head(int(top_n.value))["system_id"].tolist()
        adjacent_resolution = paired_resolution(analysis_trials, _resolution_systems)
        _resolution_view = mo.vstack(
            [
                mo.md("## Paired Resolution / Adjacent Ranks"),
                mo.md("Adjacent systems are compared on paired task means. Intervals crossing zero indicate ranks that this task set does not cleanly separate."),
                mo.ui.table(adjacent_resolution, pagination=False),
            ]
        )
    _resolution_view
    return (adjacent_resolution,)


@app.cell
def _(
    analysis_trials,
    bootstrap_rank_stability,
    mo,
    pd,
    slice_dimension,
    slice_min_tasks,
):
    _dimension = slice_dimension.value
    _summaries = []
    _skipped = []
    if _dimension in analysis_trials.columns and not analysis_trials.empty:
        for _slice_value, _slice_frame in analysis_trials.dropna(subset=[_dimension]).groupby(_dimension, sort=True, dropna=False):
            _slice_n_tasks = _slice_frame["task_id"].nunique()
            _slice_n_systems = _slice_frame["system_id"].nunique()
            if _slice_n_tasks < int(slice_min_tasks.value) or _slice_n_systems < 2:
                _skipped.append({"slice_value": str(_slice_value), "n_tasks": _slice_n_tasks, "n_systems": _slice_n_systems, "reason": "underpowered"})
                continue
            _slice_lb, _, _ = bootstrap_rank_stability(_slice_frame, draws=300, seed=17)
            if not _slice_lb.empty:
                _slice_lb.insert(0, "slice_value", str(_slice_value))
                _slice_lb.insert(0, "dimension", _dimension)
                _slice_lb.insert(2, "n_tasks", _slice_n_tasks)
                _slice_lb.insert(3, "n_systems", _slice_n_systems)
                _summaries.append(_slice_lb.head(5))
    slice_summary = pd.concat(_summaries, ignore_index=True) if _summaries else pd.DataFrame()
    skipped_slices = pd.DataFrame(_skipped)
    mo.vstack(
        [
            mo.md(f"## Slice Sensitivity: `{_dimension}`"),
            mo.ui.table(slice_summary, pagination=True, page_size=20),
            mo.accordion({"Skipped slices": mo.ui.table(skipped_slices, pagination=False)}),
        ]
    )
    return (slice_summary,)


@app.cell
def _(analysis_trials, mo, np, pd):
    def pass_at_k_for_cell(scores, k):
        _values = [float(value) for value in scores if pd.notna(value)]
        if not _values:
            return np.nan
        _successes = sum(value >= 1.0 for value in _values)
        _n = len(_values)
        if k <= 1:
            return _successes / _n
        if _successes == 0:
            return 0.0
        if _n < k:
            return 1.0
        _failures = _n - _successes
        if _failures < k:
            return 1.0
        from math import comb

        return 1 - comb(_failures, k) / comb(_n, k)

    if analysis_trials.empty:
        repeated_summary = pd.DataFrame()
        passk_summary = pd.DataFrame()
    else:
        repeated_summary = (
            analysis_trials.groupby(["task_id", "system_id"], dropna=False)
            .agg(n_trials=("run_id", "count"), mean_score=("score", "mean"), score_sd=("score", "std"))
            .reset_index()
        )
        _passk_rows = []
        for (_system_id, _task_id), _group in analysis_trials.groupby(["system_id", "task_id"], dropna=False):
            for _k in [1, 2, 3, 4]:
                _passk_rows.append({"system_id": _system_id, "task_id": _task_id, "k": _k, "pass_at_k": pass_at_k_for_cell(_group["score"], _k)})
        passk_summary = pd.DataFrame(_passk_rows).groupby(["system_id", "k"], dropna=False)["pass_at_k"].mean().reset_index() if _passk_rows else pd.DataFrame()
    mo.vstack(
        [
            mo.md("## Repeated-Trial Reliability / Pass@k Approximation"),
            mo.md("Repeated τ-bench trials let us separate task composition from rollout stochasticity. Pass@k here is estimated from observed repeated trials per task-system cell, so sparse cells should be read as exploratory."),
            mo.ui.table(repeated_summary.sort_values("n_trials", ascending=False).head(20) if not repeated_summary.empty else repeated_summary, pagination=False),
            mo.ui.table(passk_summary, pagination=True, page_size=20),
        ]
    )
    return (passk_summary,)


@app.cell
def _(
    analysis_trials,
    leaderboard,
    mo,
    observed_leaderboard,
    pd,
    score_matrix,
    top_n,
):
    def task_influence(frame, contenders):
        _matrix = score_matrix(frame)
        if _matrix.empty:
            return pd.DataFrame()
        _contenders = [system for system in contenders if system in _matrix.columns]
        _base = observed_leaderboard(_matrix[_contenders]).set_index("system_id")["observed_rank"]
        _rows = []
        for _task_id in _matrix.index:
            _reduced = _matrix.drop(index=_task_id)
            _ranks = observed_leaderboard(_reduced[_contenders]).set_index("system_id")["observed_rank"]
            _shifts = (_ranks - _base).abs()
            _rows.append(
                {
                    "task_id": _task_id,
                    "max_abs_rank_shift": _shifts.max(),
                    "mean_abs_rank_shift": _shifts.mean(),
                    "score_spread": _matrix.loc[_task_id, _contenders].max(skipna=True) - _matrix.loc[_task_id, _contenders].min(skipna=True),
                }
            )
        return pd.DataFrame(_rows).sort_values(["max_abs_rank_shift", "score_spread"], ascending=[False, False])

    if leaderboard.empty:
        influence = pd.DataFrame()
    else:
        _influence_contenders = leaderboard.head(int(top_n.value))["system_id"].tolist()
        influence = task_influence(analysis_trials, _influence_contenders).head(25)
    mo.vstack(
        [
            mo.md("## Task Influence"),
            mo.md("Leave-one-task-out influence highlights tasks that move top-system ranks or create high score spread."),
            mo.ui.table(influence, pagination=False),
        ]
    )
    return (influence,)


@app.cell
def _(analysis_trials, mo, pd, plt, sns):
    _ops_cols = [column for column in ["duration", "agent_cost", "user_cost", "num_messages", "tool_call_count"] if column in analysis_trials.columns and analysis_trials[column].notna().any()]
    if not _ops_cols:
        ops_summary = pd.DataFrame()
        _ops_view = mo.callout("No cost/duration/message/tool-call columns are populated.", kind="warn")
    else:
        ops_summary = (
            analysis_trials.groupby("system_id", dropna=False)
            .agg(observed_score=("score", "mean"), **{f"{column}_mean": (column, "mean") for column in _ops_cols})
            .reset_index()
            .sort_values("observed_score", ascending=False)
        )
        _first_op = f"{_ops_cols[0]}_mean"
        _ops_fig, _ops_ax = plt.subplots(figsize=(7, 4.5))
        sns.scatterplot(data=ops_summary, x=_first_op, y="observed_score", hue="system_id", ax=_ops_ax)
        _ops_ax.set_title(f"Reward versus {_ops_cols[0]}")
        _ops_fig.tight_layout()
        _ops_view = mo.vstack([mo.mpl.interactive(_ops_fig), mo.ui.table(ops_summary, pagination=True, page_size=20)])
    mo.vstack([mo.md("## Operational Profile"), _ops_view])
    return


@app.cell
def _(
    adjacent_resolution,
    capability_matrix,
    data_summary,
    evidence_counts,
    influence,
    leaderboard,
    mo,
    passk_summary,
    slice_summary,
):
    mo.vstack(
        [
            mo.md("## Final Audit Summary"),
            mo.md(
                """
    This notebook is an outcome-level rank-resolution audit for τ-bench result files.

    The headline outputs to report are observed reward leaderboard and bootstrap rank intervals, adjacent-rank paired resolution, domain/reward-basis slices where powered, repeated-trial/pass@k diagnostics where repeated runs exist, task influence, and operational profile.

    Main caveat: task resampling approximates uncertainty over the observed task universe. It does not by itself certify simulator reliability, temporal drift, or production tail-risk bounds.
    """
            ),
            mo.accordion(
                {
                    "Data summary": mo.ui.table(data_summary, pagination=False),
                    "Evidence counts": mo.ui.table(evidence_counts, pagination=False),
                    "Capability matrix": mo.ui.table(capability_matrix, pagination=False),
                    "Leaderboard": mo.ui.table(leaderboard, pagination=True, page_size=20),
                    "Adjacent resolution": mo.ui.table(adjacent_resolution, pagination=True, page_size=20),
                    "Slice summary": mo.ui.table(slice_summary, pagination=True, page_size=20),
                    "Pass@k summary": mo.ui.table(passk_summary, pagination=True, page_size=20),
                    "Task influence": mo.ui.table(influence, pagination=True, page_size=20),
                }
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
