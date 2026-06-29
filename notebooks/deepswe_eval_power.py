import marimo

__generated_with = "0.23.11"
app = marimo.App(width="full")


@app.cell
def _():
    from dataclasses import dataclass
    from pathlib import Path
    import json

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import requests
    import seaborn as sns

    ROOT = Path(__file__).resolve().parents[1]
    CACHE_ROOT = ROOT / ".cache" / "deepswe_eval_power" / "deepswe"
    ARTIFACT_BASE = "https://deepswe.datacurve.ai/artifacts"
    RNG_SEED = 42
    N_BOOT = 800
    PRACTICAL_EQUIVALENCE_PP = 2.0
    MIN_SLICE_TASKS = 2
    MIN_SLICE_SYSTEMS = 2

    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["figure.dpi"] = 130
    return (
        ARTIFACT_BASE,
        CACHE_ROOT,
        MIN_SLICE_SYSTEMS,
        MIN_SLICE_TASKS,
        N_BOOT,
        PRACTICAL_EQUIVALENCE_PP,
        RNG_SEED,
        dataclass,
        json,
        mo,
        np,
        pd,
        plt,
        requests,
        sns,
    )


@app.cell
def _(mo):
    mo.md(
        """
# DeepSWE Eval Power / Rank Resolution Audit

DeepSWE should be read as a **rank-resolution audit**, not only as a point leaderboard.

The default ranked unit is the evaluated **config/system**:
model + provider + reasoning effort under the constant `mini-swe-agent` harness.

## How to read this notebook

Read each block as a question-driven audit:

1. **Define the estimand:** fixed observed leaderboard, task-population inference, or config/system comparison.
2. **Check the evidence:** identify which taxonomy fields and metadata exist.
3. **Run only supported methods:** paired comparisons, task bootstrap, slices, influence, variance, and operational profile where the data support them.
4. **Report claims with caveats:** non-separation is not equivalence, and config-level claims are not pure model claims.
"""
    )
    return


@app.cell
def _(mo):
    version = mo.ui.dropdown(
        options=["v1.1", "v1"],
        value="v1.1",
        label="Data version",
    )
    rank_unit = mo.ui.dropdown(
        options=["config", "model", "provider", "reasoning_effort"],
        value="config",
        label="Ranked unit",
    )
    slice_dimension = mo.ui.dropdown(
        options=["repository", "language", "provider", "model", "reasoning_effort"],
        value="language",
        label="Slice dimension",
    )
    slice_min_tasks = mo.ui.slider(1, 10, value=2, step=1, show_value=True, label="Min tasks per slice")
    top_n = mo.ui.slider(5, 25, value=15, step=1, show_value=True, label="Top N")
    top_k = mo.ui.slider(3, 15, value=10, step=1, show_value=True, label="Top K")
    mo.vstack(
        [
            mo.md("## Setup And Data / Eval Universe"),
            mo.hstack([version, rank_unit, slice_dimension, slice_min_tasks, top_n, top_k], justify="start", gap=1),
        ]
    )
    return rank_unit, slice_dimension, slice_min_tasks, top_k, top_n, version


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
        artifact_cols: tuple[str, ...] = ()
        judge_cols: tuple[str, ...] = ()
        simulator_cols: tuple[str, ...] = ()
        temporal_cols: tuple[str, ...] = ()
        risk_cols: tuple[str, ...] = ()

        @property
        def required_cols(self):
            return (self.task_col, self.system_col, self.run_col, self.score_col)


    DEEPSWE_SPEC = EvalSpec(
        name="DeepSWE",
        task_col="task_id",
        system_col="system_id",
        run_col="run_id",
        score_col="score",
        system_factor_cols=("config", "model", "provider", "reasoning_effort", "harness"),
        dimension_cols=("repository", "language", "provider", "model", "reasoning_effort"),
        reliability_cols=(
            "cost_usd",
            "n_agent_steps",
            "agent_duration_seconds",
            "trial_duration_seconds",
            "n_input_tokens",
            "n_output_tokens",
            "peak_context_tokens",
        ),
        artifact_cols=("has_trajectory", "has_agent_log", "has_model_patch", "has_verifier_output"),
        temporal_cols=("finished_at",),
    )
    return DEEPSWE_SPEC, EvalSpec


@app.cell
def _(
    ARTIFACT_BASE,
    CACHE_ROOT,
    json,
    pd,
    requests,
):
    def fetch_artifact(version_id, name, *, timeout=60):
        path = CACHE_ROOT / version_id / f"{name}.json"
        url = f"{ARTIFACT_BASE}/{version_id}/{name}.json"
        status = "cache"
        if not path.exists():
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2, sort_keys=True))
            status = "downloaded"
        else:
            payload = json.loads(path.read_text())
        return payload, {"artifact": name, "url": url, "cache_path": str(path), "status": status}


    def normalize_payloads(version_id):
        trials_payload, trials_status = fetch_artifact(version_id, "trials")
        tasks_payload, tasks_status = fetch_artifact(version_id, "tasks")
        leaderboard_payload, leaderboard_status = fetch_artifact(version_id, "leaderboard-live")

        trials = pd.DataFrame(trials_payload.get("rows", []))
        tasks = pd.DataFrame(tasks_payload.get("rows", []))
        leaderboard_live = pd.DataFrame(leaderboard_payload.get("rows", []))

        if "id" in tasks.columns:
            tasks = tasks.rename(columns={"id": "task_id"})
        if "task_name" not in tasks.columns and "task_id" in tasks.columns:
            tasks["task_name"] = tasks["task_id"]

        if "task_name" in trials.columns:
            trials["task_id"] = trials["task_name"]
        if "trial_name" in trials.columns:
            trials["trial_id"] = trials["trial_name"]
            trials["run_id"] = trials["trial_id"]
        if "config" in trials.columns:
            trials["system_id"] = trials["config"].astype(str)
        if "passed" not in trials.columns and "score_value" in trials.columns:
            trials["passed"] = pd.to_numeric(trials["score_value"], errors="coerce").fillna(0) >= 1
        if "score_value" in trials.columns:
            trials["score"] = pd.to_numeric(trials["score_value"], errors="coerce")
        else:
            trials["score"] = trials["passed"].fillna(False).astype(float)

        trials["source"] = trials.get("source", "deep-swe")
        trials["eval_scope"] = trials.get("eval_scope", "full")
        trials["included_in_score"] = trials.get("included_in_score", True)
        trials = trials[
            (trials["source"] == "deep-swe")
            & (trials["eval_scope"] == "full")
            & (trials["included_in_score"].fillna(False).astype(bool))
        ].copy()

        if not tasks.empty and "task_id" in tasks.columns:
            metadata = [
                c
                for c in ["task_id", "repository", "language", "problem_title", "display_description", "prompt_characters"]
                if c in tasks.columns
            ]
            trials = trials.merge(tasks[metadata], on="task_id", how="left", validate="many_to_one")

        for column in [
            "cost_usd",
            "n_agent_steps",
            "agent_duration_seconds",
            "trial_duration_seconds",
            "n_input_tokens",
            "n_output_tokens",
            "peak_context_tokens",
            "score",
            "score_value",
            "pass_rate",
            "partial",
            "p2p",
            "f2p",
        ]:
            if column in trials.columns:
                trials[column] = pd.to_numeric(trials[column], errors="coerce")

        statuses = pd.DataFrame([trials_status, tasks_status, leaderboard_status])
        return trials, tasks, leaderboard_live, statuses


    return fetch_artifact, normalize_payloads


@app.cell
def _(mo, normalize_payloads, version):
    try:
        trials, tasks, leaderboard_live, fetch_status = normalize_payloads(version.value)
        data_error = None
    except Exception as exc:
        trials, tasks, leaderboard_live, fetch_status = None, None, None, None
        data_error = f"{type(exc).__name__}: {exc}"

    data_status_view = (
        mo.callout(f"Could not load DeepSWE {version.value}: `{data_error}`", kind="danger")
        if data_error
        else mo.md(f"Loaded DeepSWE `{version.value}` artifacts.")
    )
    data_status_view
    return data_error, fetch_status, leaderboard_live, tasks, trials


@app.cell
def _(data_error, fetch_status, mo, pd, trials):
    if data_error:
        data_summary = pd.DataFrame()
    else:
        data_summary = pd.DataFrame(
            [
                {
                    "rows": len(trials),
                    "tasks": trials["task_id"].nunique(),
                    "configs": trials["config"].nunique(),
                    "models": trials["model"].nunique(),
                    "providers": trials["provider"].nunique() if "provider" in trials.columns else None,
                    "harnesses": trials["harness"].nunique() if "harness" in trials.columns else None,
                    "first_finished_at": trials["finished_at"].min() if "finished_at" in trials.columns else None,
                    "last_finished_at": trials["finished_at"].max() if "finished_at" in trials.columns else None,
                }
            ]
        )
    mo.vstack(
        [
            mo.ui.table(data_summary, pagination=False),
            mo.accordion({"Artifact fetch/cache status": mo.ui.table(fetch_status, pagination=False)}) if not data_error else mo.md(""),
        ]
    )
    return data_summary


@app.cell
def _(DEEPSWE_SPEC, data_error, mo, np, pd, plt, sns, trials, version):
    def has_cols(*columns):
        return (not data_error) and all(column in trials.columns and trials[column].notna().any() for column in columns)


    def availability(status):
        return {"yes": "available", "partial": "partial", "no": "unavailable"}[status]


    def action(status):
        return {"run": "run", "caveat": "run with caveat", "availability": "availability only", "roadmap": "roadmap"}[status]


    n_tasks = int(trials["task_id"].nunique()) if has_cols("task_id") else 0
    n_systems = int(trials["system_id"].nunique()) if has_cols("system_id") else 0
    n_runs = int(trials[DEEPSWE_SPEC.run_col].nunique()) if has_cols(DEEPSWE_SPEC.run_col) else 0
    repeated_cells = False
    repeated_cell_count = 0
    _missing_cells = 0
    coverage_pct = np.nan
    if not data_error and {"task_id", "system_id", DEEPSWE_SPEC.run_col}.issubset(trials.columns):
        cell_counts = trials.groupby(["task_id", "system_id"], dropna=False)[DEEPSWE_SPEC.run_col].size()
        repeated_cell_count = int((cell_counts > 1).sum())
        repeated_cells = bool(repeated_cell_count > 0)
        possible_cells = n_tasks * n_systems
        _observed_cells = int(cell_counts.size)
        _missing_cells = possible_cells - _observed_cells
        coverage_pct = 100 * _observed_cells / possible_cells if possible_cells else np.nan

    has_dimensions = [column for column in DEEPSWE_SPEC.dimension_cols if has_cols(column)]
    has_reliability = [column for column in DEEPSWE_SPEC.reliability_cols if has_cols(column)]
    has_artifacts = [column for column in DEEPSWE_SPEC.artifact_cols if has_cols(column)]
    _harness_count = trials["harness"].nunique(dropna=True) if has_cols("harness") else 0
    _artifact_min_coverage = (
        min(100 * trials[column].fillna(False).astype(bool).mean() for column in has_artifacts)
        if has_artifacts
        else np.nan
    )

    evidence_counts = pd.DataFrame(
        [
            ("tasks", n_tasks),
            ("systems/configs", n_systems),
            ("runs", n_runs),
            ("repeated config-task cells", repeated_cell_count),
            ("missing config-task cells", _missing_cells),
            ("config-task coverage %", coverage_pct),
            ("available dimensions", len(has_dimensions)),
            ("available reliability fields", len(has_reliability)),
            ("available artifact fields", len(has_artifacts)),
            ("minimum artifact coverage %", _artifact_min_coverage),
            ("harness count", _harness_count),
            ("judge fields", len([column for column in DEEPSWE_SPEC.judge_cols if has_cols(column)])),
            ("risk fields", len([column for column in DEEPSWE_SPEC.risk_cols if has_cols(column)])),
            ("temporal fields", len([column for column in DEEPSWE_SPEC.temporal_cols if has_cols(column)])),
        ],
        columns=["evidence", "value"],
    )

    taxonomy = pd.DataFrame(
        [
            ("S", "evaluated system", "config/system: model + provider + reasoning effort + harness"),
            ("M", "model", "model, provider, reasoning_effort"),
            ("H", "harness/scaffold", "harness; constant `mini-swe-agent` in DeepSWE v1/v1.1"),
            ("E", "environment", "not separately encoded; bundled into run/config execution"),
            ("B", "benchmark", f"DeepSWE {version.value}"),
            ("C", "domain/cluster", "repository, language, provider, model, reasoning_effort slices"),
            ("T", "task", "task_id"),
            ("R", "rollout/run", "trial_id"),
            ("trajectory", "trajectory", "artifact availability flags only"),
            ("e", "event", "trace contents not parsed here"),
        ],
        columns=["Symbol", "Meaning", "DeepSWE mapping"],
    )
    capability_matrix = pd.DataFrame(
        [
            (
                "What is the observed leaderboard?",
                "observed score with official intervals where available",
                "system_id, task_id, score; optional official leaderboard intervals",
                f"{n_tasks} tasks; {n_systems} configs",
                availability("yes" if has_cols("system_id", "task_id", "score") else "no"),
                "all required outcome columns present" if has_cols("system_id", "task_id", "score") else "missing outcome columns",
                action("run"),
            ),
            (
                "Can nearby ranks be distinguished?",
                "paired deltas, adjacent CIs, MDE, all-pairs max-T",
                "paired task outcomes for multiple systems",
                f"{coverage_pct:.1f}% config-task coverage; {_missing_cells} missing cells",
                availability("yes" if has_cols("task_id", "system_id", "score") else "no"),
                "paired task outcomes available" if has_cols("task_id", "system_id", "score") else "missing paired outcome columns",
                action("run"),
            ),
            (
                "Are ranks stable under task resampling?",
                "task bootstrap rank intervals and top-K probabilities",
                "task-level outcomes by system",
                f"{n_tasks} task resampling units",
                availability("yes" if has_cols("task_id", "system_id", "score") else "no"),
                "task IDs define the resampling unit" if has_cols("task_id") else "missing task IDs",
                action("run"),
            ),
            (
                "Where does performance differ?",
                "slice leaderboards by powered dimensions",
                "domain/slice metadata",
                ", ".join(has_dimensions) if has_dimensions else "no slice dimensions",
                availability("yes" if has_dimensions else "no"),
                "slice dimensions available" if has_dimensions else "no usable slice metadata",
                action("run" if has_dimensions else "roadmap"),
            ),
            (
                "Which tasks/domains move aggregate ranks?",
                "leave-one-task/domain-out influence",
                "task outcomes and domain metadata",
                f"{n_tasks} tasks; dimensions: {', '.join(has_dimensions) if has_dimensions else 'none'}",
                availability("yes" if has_cols("task_id", "system_id", "score") and has_dimensions else "partial"),
                "domain influence uses available dimensions; task influence still runs with task IDs" if has_cols("task_id", "system_id", "score") else "missing outcome columns",
                action("run" if has_cols("task_id", "system_id", "score") and has_dimensions else "caveat"),
            ),
            (
                "Would one more run help?",
                "within-run vs task variance proxy, K-simulation",
                "repeated runs per task/system",
                f"{repeated_cell_count} repeated config-task cells",
                availability("yes" if repeated_cells else "partial"),
                "repeats exist, but the design is not fully crossed" if repeated_cells else "no repeated config-task cells",
                "run with caveat",
            ),
            (
                "Is the eval universe imbalanced or underpowered?",
                "effective counts, coverage, skipped slices",
                "dimension metadata and config-task coverage",
                f"{len(has_dimensions)} dimensions; {coverage_pct:.1f}% config-task coverage",
                availability("yes" if has_dimensions else "partial"),
                "coverage is global; some slices may remain underpowered" if has_dimensions else "coverage only; no dimension imbalance",
                action("run" if has_dimensions else "caveat"),
            ),
            (
                "Do similar scores hide different operational costs?",
                "score vs cost/steps/duration/tokens",
                "reliability/cost columns",
                ", ".join(has_reliability) if has_reliability else "no reliability fields",
                availability("yes" if has_reliability else "no"),
                "operational metadata available" if has_reliability else "cost/latency/token columns missing",
                action("run" if has_reliability else "roadmap"),
            ),
            (
                "What is confounded?",
                "factor variation matrix",
                "model/config/provider/harness/environment fields",
                f"{_harness_count} harness value(s)",
                availability("partial" if _harness_count <= 1 else "yes"),
                "harness is constant, so model x harness interaction is not identifiable" if _harness_count <= 1 else "crossed harness variation exists",
                "run with attribution caveat",
            ),
            (
                "Are traces available for future failure analysis?",
                "artifact availability by config/outcome",
                "trajectory/log/patch/verifier artifact flags",
                f"{len(has_artifacts)} artifact flags; min coverage {_artifact_min_coverage:.1f}%" if has_artifacts else "no artifact flags",
                availability("yes" if has_artifacts else "no"),
                "checks observability only; trace contents are not parsed" if has_artifacts else "no trace/artifact flags",
                "availability only",
            ),
            (
                "Are judges stable?",
                "judge repeat agreement/calibration",
                "judge IDs, repeated judge labels, expert labels",
                "no judge columns declared",
                availability("yes" if any(has_cols(c) for c in DEEPSWE_SPEC.judge_cols) else "no"),
                "missing judge IDs/repeated labels/expert labels",
                "roadmap",
            ),
            (
                "Does performance drift over time?",
                "snapshot comparison",
                "multiple eval snapshots or reruns over time",
                "finished_at exists, but one eval snapshot",
                availability("partial" if has_cols("finished_at") else "no"),
                "timestamp metadata exists, but there is only one eval snapshot" if has_cols("finished_at") else "missing time/snapshot metadata",
                "roadmap",
            ),
            (
                "Are rare/tail failures bounded?",
                "severity labels, stress probes, rare-event bounds",
                "risk/severity labels or targeted probes",
                "no risk/severity columns declared",
                availability("yes" if any(has_cols(c) for c in DEEPSWE_SPEC.risk_cols) else "no"),
                "missing severity labels or targeted stress probes",
                "roadmap",
            ),
        ],
        columns=["Claim/question", "Method", "Data needed", "DeepSWE evidence", "Support", "Support detail", "Action"],
    )
    evidence_requirements = pd.DataFrame(
        [
            ("leaderboard", 1, 1, 0, 0, 0, 0, 0, 0),
            ("resolution", 1, 1, 0, 0, 0, 0, 0, 0),
            ("stability", 1, 1, 0, 0, 0, 0, 0, 0),
            ("heterogeneity", 1, 1, 1, 0, 0, 0, 0, 0),
            ("influence", 1, 1, 1, 0, 0, 0, 0, 0),
            ("variance budget", 1, 1, 0, 1 if repeated_cells else 0.5, 0, 0, 0, 0),
            ("coverage/missingness", 1, 1, 1, 0, 0, 0, 0, 0),
            ("reliability", 1, 1, 0, 0, 1 if has_reliability else 0, 0, 0, 0),
            ("confounding", 1, 1, 0, 0, 0, 0.5 if _harness_count <= 1 else 1, 0, 0),
            ("trace readiness", 1, 1, 0, 0, 0, 0, 1 if has_artifacts else 0, 0),
            ("judge calibration", 0, 0, 0, 0, 0, 0, 0, 0),
            ("tail risk", 0, 0, 0, 0, 0, 0, 0, 0),
        ],
        columns=["question", "outcomes", "tasks/systems", "dimensions", "repeats", "reliability", "harness variation", "artifacts", "judge/risk"],
    )
    fig_avail, _avail_ax = plt.subplots(figsize=(9, 4.8))
    heat_values = evidence_requirements.set_index("question")
    sns.heatmap(
        heat_values,
        cmap=sns.color_palette(["#F2F2F2", "#F6C85F", "#4C78A8"]),
        vmin=0,
        vmax=1,
        cbar=False,
        linewidths=0.5,
        linecolor="white",
        ax=_avail_ax,
    )
    _avail_ax.set_title("Evidence availability by question")
    _avail_ax.set_xlabel("required evidence")
    _avail_ax.set_ylabel("")
    fig_avail.tight_layout()
    audit_pipeline = pd.DataFrame(
        [
            ("1. Taxonomy", "`S = M x H x E`; `B -> C -> T -> R -> trajectory -> event`", "names the object being measured"),
            ("2. Data mapping", "DeepSWE columns mapped to taxonomy symbols", "makes config/task/run/artifact limits explicit"),
            ("3. Evidence availability", "available / partial / unavailable by question", "prevents running unsupported analyses"),
            ("4. Method routing", "question chooses paired, bootstrap, slice, influence, variance, or availability check", "keeps methods attached to claims"),
            ("5. Claim report", "claim + method + evidence + finding + caveat", "produces the final audit artifact"),
        ],
        columns=["stage", "notebook object", "purpose"],
    )
    mo.vstack(
        [
            mo.md("## Taxonomy"),
            mo.md(
                "`S = M x H x E`, and `B -> C -> T -> R -> trajectory -> event`. DeepSWE can identify configs, models, providers, reasoning effort, tasks, domains, rollouts, and artifact availability. It cannot identify model x harness interaction here because the harness is constant."
            ),
            mo.md("### Audit pipeline"),
            mo.md("The notebook is not a grab bag of charts. It routes from taxonomy to data availability to the methods that can support each claim."),
            mo.ui.table(audit_pipeline, pagination=False),
            mo.ui.table(taxonomy, pagination=False),
            mo.md("### DeepSWE evidence counts"),
            mo.ui.table(evidence_counts, pagination=False),
            mo.md("### Chart glossary"),
            mo.ui.table(
                pd.DataFrame(
                    [
                        ("CI", "confidence interval; plausible range under the stated resampling model"),
                        ("MDE", "minimum detectable effect; rough gap size this eval can see with chosen power"),
                        ("max-T", "family-wise correction for many pairwise comparisons"),
                        ("task bootstrap", "resample tasks, not raw rows, so tasks remain the effective sample unit"),
                        ("non-separated", "not statistically distinguishable here; not proof of equality"),
                        ("influence", "how much ranks move when one task or domain is removed"),
                    ],
                    columns=["term", "meaning in this notebook"],
                ),
                pagination=False,
            ),
            mo.md("## Question / Capability Matrix"),
            mo.md(
                "The audit does not run every method on every eval. It first asks what evidence exists, then runs only the checks that the mapped data can support."
            ),
            mo.mpl.interactive(fig_avail),
            mo.ui.table(capability_matrix, pagination=False),
        ]
    )
    return capability_matrix, evidence_counts, fig_avail, taxonomy


@app.cell
def _(mo, rank_unit, version):
    mo.md(
        f"""
## Estimand

This notebook keeps three targets separate:

- **Fixed benchmark description:** what happened on the observed DeepSWE `{version.value}` task set.
- **Task-population inference:** what might change if similar tasks were resampled; this is what task bootstrap approximates.
- **System/config inference:** the selected unit is `{rank_unit.value}`. In DeepSWE, the default config view is not a pure model-only claim because the measured object is a configuration inside a constant `mini-swe-agent` harness.

How to read later sections:

- **Pairwise comparisons are paired by task.** Two configs are compared on the same tasks, so the task-level difference is the measurement unit.
- **Task/domain clustering is a design warning.** A benchmark with many rows can still have fewer effective task families if domains are concentrated.
- **Top-K windows are reporting choices.** The notebook checks whether conclusions change around the chosen top-N/top-K display boundary.
- **Repeated-run power is budget guidance.** It estimates whether extra rollouts reduce uncertainty enough to justify their cost.

Statistical non-separation is also not equivalence. If two configs are not distinguishable here, the notebook only says this benchmark lacks enough evidence to separate them at the chosen resolution.
"""
    )
    return


@app.cell
def _(N_BOOT, PRACTICAL_EQUIVALENCE_PP, RNG_SEED, leaderboard_live, np, pd, rank_unit, top_k, top_n, trials):
    selected_unit = rank_unit.value
    selected_top_n = int(top_n.value)
    selected_top_k = int(top_k.value)

    def clean_effort(value):
        if pd.isna(value) or value in ("", "default", None):
            return "default"
        return str(value)


    def display_label(row_or_value, unit=selected_unit):
        if not isinstance(row_or_value, pd.Series):
            text = str(row_or_value)
            return text.removeprefix("mini_swe_agent_").replace("_", "-")
        if unit == "config":
            model = str(row_or_value.get("model", "")).replace("_", "-")
            effort = clean_effort(row_or_value.get("reasoning_effort"))
            return model if effort == "default" else f"{model} / {effort}"
        value = row_or_value.get(unit)
        return clean_effort(value) if unit == "reasoning_effort" else str(value)


    def system_table(frame, unit=selected_unit):
        out = frame.copy()
        out["system_id"] = out[unit].fillna("unknown").astype(str)
        if unit == "config":
            out["system_label"] = out.apply(display_label, axis=1)
        else:
            out["system_label"] = out["system_id"].map(lambda value: display_label(value, unit))
        return out


    scored_trials = system_table(trials, selected_unit)
    cell_scores = (
        scored_trials.groupby(["task_id", "system_id"], dropna=False)
        .agg(
            score=("score", "mean"),
            n_runs=("trial_id", "count"),
            label=("system_label", "first"),
        )
        .reset_index()
    )
    matrix = cell_scores.pivot_table(index="task_id", columns="system_id", values="score", aggfunc="mean")
    labels = cell_scores.groupby("system_id")["label"].first().to_dict()

    trial_leaderboard = (
        cell_scores.groupby("system_id")
        .agg(score=("score", "mean"), n_tasks=("task_id", "nunique"), label=("label", "first"))
        .reset_index()
    )
    if selected_unit == "config" and not leaderboard_live.empty and {"config", "pass_rate"}.issubset(leaderboard_live.columns):
        observed = leaderboard_live.copy()
        observed["system_id"] = observed["config"].astype(str)
        observed["score"] = pd.to_numeric(observed["pass_rate"], errors="coerce")
        observed["score_ci_lo"] = pd.to_numeric(observed.get("ci_lo"), errors="coerce")
        observed["score_ci_hi"] = pd.to_numeric(observed.get("ci_hi"), errors="coerce")
        observed["label"] = observed.apply(display_label, axis=1)
        observed["n_tasks"] = observed.get("n_tasks_attempted", np.nan)
        observed_source = "official leaderboard-live pass_rate"
        observed = observed[observed["system_id"].isin(matrix.columns)]
    else:
        observed = trial_leaderboard.copy()
        observed["score_ci_lo"] = np.nan
        observed["score_ci_hi"] = np.nan
        observed_source = "trial aggregate binary pass rate"

    observed = (
        observed.sort_values(["score", "system_id"], ascending=[False, True])
        .assign(observed_rank=lambda df: np.arange(1, len(df) + 1))
        .reset_index(drop=True)
    )
    order = observed["system_id"].tolist()
    matrix = matrix.reindex(columns=order)

    def bootstrap_scores(score_matrix, draws=N_BOOT, seed=RNG_SEED):
        rng = np.random.default_rng(seed)
        values = score_matrix.to_numpy(dtype=float)
        n_tasks, n_systems = values.shape
        if n_tasks == 0 or n_systems == 0:
            return pd.DataFrame(), pd.DataFrame()
        sampled = rng.integers(0, n_tasks, size=(draws, n_tasks))
        scores = np.empty((draws, n_systems), dtype=float)
        for draw, task_idx in enumerate(sampled):
            sample = values[task_idx]
            valid = np.sum(~np.isnan(sample), axis=0)
            sums = np.nansum(sample, axis=0)
            scores[draw] = np.divide(sums, valid, out=np.full(n_systems, np.nan), where=valid > 0)
        score_frame = pd.DataFrame(scores, columns=score_matrix.columns)
        rank_frame = score_frame.rank(axis=1, method="min", ascending=False, na_option="bottom")
        return score_frame, rank_frame


    boot_scores, boot_ranks = bootstrap_scores(matrix)
    rank_summary = observed[["system_id", "label", "score", "observed_rank"]].copy()
    if not boot_scores.empty:
        rank_summary["boot_score_p05"] = boot_scores.quantile(0.05).reindex(rank_summary["system_id"]).values
        rank_summary["boot_score_p95"] = boot_scores.quantile(0.95).reindex(rank_summary["system_id"]).values
        rank_summary["rank_p05"] = boot_ranks.quantile(0.05).reindex(rank_summary["system_id"]).values
        rank_summary["rank_p50"] = boot_ranks.quantile(0.50).reindex(rank_summary["system_id"]).values
        rank_summary["rank_p95"] = boot_ranks.quantile(0.95).reindex(rank_summary["system_id"]).values
        rank_summary[f"p_top{selected_top_k}"] = (boot_ranks <= selected_top_k).mean().reindex(rank_summary["system_id"]).values

    top_pair_order = order[:selected_top_n]
    observed_task_scores = matrix.mean(axis=0, skipna=True)
    all_pair_rows = []
    pair_t_statistics = []
    pair_keys = []
    for left_index, left in enumerate(top_pair_order[:-1]):
        for right in top_pair_order[left_index + 1 :]:
            task_delta = (matrix[left] - matrix[right]).dropna()
            if len(task_delta) < 2:
                continue
            gap = float(observed_task_scores.get(left, np.nan) - observed_task_scores.get(right, np.nan))
            if np.isnan(gap):
                gap = float(
                    observed.loc[observed["system_id"] == left, "score"].iloc[0]
                    - observed.loc[observed["system_id"] == right, "score"].iloc[0]
                )
            boot_delta = boot_scores[left] - boot_scores[right] if not boot_scores.empty else pd.Series(dtype=float)
            se = float(boot_delta.std(ddof=1)) if len(boot_delta.dropna()) > 1 else float(task_delta.std(ddof=1) / np.sqrt(len(task_delta)))
            ci_lo = float(boot_delta.quantile(0.025)) if len(boot_delta.dropna()) > 1 else gap - 1.96 * se
            ci_hi = float(boot_delta.quantile(0.975)) if len(boot_delta.dropna()) > 1 else gap + 1.96 * se
            if len(boot_delta.dropna()) > 1:
                p_boot = float(2 * min((boot_delta <= 0).mean(), (boot_delta >= 0).mean()))
            else:
                p_boot = np.nan
            rank_left = int(observed.loc[observed["system_id"] == left, "observed_rank"].iloc[0])
            rank_right = int(observed.loc[observed["system_id"] == right, "observed_rank"].iloc[0])
            pair_key = f"{left}__vs__{right}"
            if se > 0 and len(boot_delta.dropna()) > 1:
                pair_t_statistics.append(((boot_delta - gap).abs() / se).rename(pair_key))
                pair_keys.append(pair_key)
            all_pair_rows.append(
                {
                    "pair_key": pair_key,
                    "rank_A": rank_left,
                    "rank_B": rank_right,
                    "A": labels.get(left, left),
                    "B": labels.get(right, right),
                    "gap_pp": 100 * gap,
                    "ci_lo_pp": 100 * ci_lo,
                    "ci_hi_pp": 100 * ci_hi,
                    "mde80_pp": 100 * (1.96 + 0.84) * se,
                    "n_tasks": int(len(task_delta)),
                    "p_boot": p_boot,
                    "win_probability": float(((boot_delta > 0).mean() + 0.5 * (boot_delta == 0).mean())) if len(boot_delta.dropna()) else np.nan,
                    "within_practical_band": abs(100 * gap) < PRACTICAL_EQUIVALENCE_PP,
                    "pointwise_ci_excludes_zero": ci_lo > 0 or ci_hi < 0,
                    "holm_significant": False,
                    "bh_significant": False,
                    "family_ci_lo_pp": np.nan,
                    "family_ci_hi_pp": np.nan,
                    "familywise_significant": False,
                }
            )

    all_pairs = pd.DataFrame(all_pair_rows)
    if not all_pairs.empty and all_pairs["p_boot"].notna().any():
        _p = all_pairs["p_boot"].fillna(1.0).clip(0, 1)
        _m = len(_p)
        _order_p = _p.sort_values().index.tolist()
        _holm_significant = pd.Series(False, index=all_pairs.index)
        for _rank, _idx in enumerate(_order_p, start=1):
            if _p.loc[_idx] <= 0.05 / (_m - _rank + 1):
                _holm_significant.loc[_idx] = True
            else:
                break
        _bh_threshold = pd.Series(
            [0.05 * _rank / _m for _rank in range(1, _m + 1)],
            index=_order_p,
        )
        _bh_candidates = _p.loc[_order_p] <= _bh_threshold
        _bh_significant = pd.Series(False, index=all_pairs.index)
        if _bh_candidates.any():
            _max_bh_rank = int(np.where(_bh_candidates.to_numpy())[0].max())
            _bh_significant.loc[_order_p[: _max_bh_rank + 1]] = True
        all_pairs["holm_significant"] = _holm_significant.reindex(all_pairs.index).fillna(False).astype(bool)
        all_pairs["bh_significant"] = _bh_significant.reindex(all_pairs.index).fillna(False).astype(bool)
    max_t_critical = float("nan")
    if pair_t_statistics and not all_pairs.empty:
        max_t_frame = pd.concat(pair_t_statistics, axis=1)
        max_t_critical = float(max_t_frame.max(axis=1).quantile(0.95))
        for _idx, _pair_row in all_pairs.iterrows():
            _se_pp = _pair_row["mde80_pp"] / (1.96 + 0.84)
            _lo = _pair_row["gap_pp"] - max_t_critical * _se_pp
            _hi = _pair_row["gap_pp"] + max_t_critical * _se_pp
            all_pairs.loc[_idx, "family_ci_lo_pp"] = _lo
            all_pairs.loc[_idx, "family_ci_hi_pp"] = _hi
            all_pairs.loc[_idx, "familywise_significant"] = _lo > 0 or _hi < 0

    adjacent = all_pairs[all_pairs["rank_B"] == all_pairs["rank_A"] + 1].reset_index(drop=True)
    tier_rows = []
    tier = 1
    for _idx, _tier_system_id in enumerate(order):
        if _idx > 0 and not adjacent.empty:
            _prev = adjacent[
                (adjacent["rank_A"] == _idx)
                & (adjacent["rank_B"] == _idx + 1)
            ]
            if not _prev.empty and bool(_prev["familywise_significant"].iloc[0]):
                tier += 1
        tier_rows.append(
            {
                "observed_rank": _idx + 1,
                "system_id": _tier_system_id,
                "label": labels.get(_tier_system_id, _tier_system_id),
                "non_separation_tier": tier,
            }
        )
    nonseparation_tiers = pd.DataFrame(tier_rows)
    return (
        adjacent,
        all_pairs,
        boot_ranks,
        boot_scores,
        cell_scores,
        display_label,
        labels,
        matrix,
        max_t_critical,
        nonseparation_tiers,
        observed,
        observed_source,
        rank_summary,
        scored_trials,
        selected_top_k,
        selected_top_n,
        selected_unit,
    )


@app.cell
def _(mo, pd):
    def pct(value):
        return f"{100 * value:.1f}%"


    def pp(value):
        return f"{value:.1f} pp"


    def tidy_table(frame, digits=1):
        out = frame.copy()
        for _col in out.columns:
            if pd.api.types.is_numeric_dtype(out[_col]):
                out[_col] = out[_col].round(digits)
        return out


    def section(title, question, why, method, chart, findings, caveat=None):
        bullets = "\n".join(f"- {item}" for item in findings if item)
        caveat_text = f"<p><strong>Caveat:</strong> {caveat}</p>" if caveat else ""
        return mo.md(
            f"## {title}\n"
            f"<div style='line-height:1.35'>"
            f"<p><strong>Question:</strong> {question}<br>"
            f"<strong>Why it matters:</strong> {why}<br>"
            f"<strong>Method:</strong> {method}<br>"
            f"<strong>How to read the chart:</strong> {chart}</p>"
            f"</div>\n\n"
            f"**Finding:**\n{bullets}\n\n"
            f"{caveat_text}"
        )


    return pct, pp, section, tidy_table


@app.cell
def _(mo):
    mo.md(
        """
## Supported Analyses

The sections below run the subset of the audit that DeepSWE can support from outcome, task, config, domain, cost, and artifact metadata. Roadmap-only checks stay visible in the capability matrix and final report.
"""
    )
    return


@app.cell
def _(mo, observed, observed_source, plt, rank_summary, section, selected_top_n, selected_unit, tidy_table):
    plot_data = observed.head(selected_top_n).iloc[::-1].copy()
    fig, _ax = plt.subplots(figsize=(9, max(4, 0.34 * len(plot_data))))
    _xerr = None
    if plot_data["score_ci_lo"].notna().any() and plot_data["score_ci_hi"].notna().any():
        _xerr = [
            (plot_data["score"] - plot_data["score_ci_lo"]).clip(lower=0).to_numpy(),
            (plot_data["score_ci_hi"] - plot_data["score"]).clip(lower=0).to_numpy(),
        ]
    _ax.barh(plot_data["label"], 100 * plot_data["score"], xerr=None if _xerr is None else [[100 * v for v in _xerr[0]], [100 * v for v in _xerr[1]]], color="#4C78A8", alpha=0.9)
    _ax.set_xlabel("Score (%)")
    _ax.set_title(f"Observed leaderboard ({selected_unit}); source: {observed_source}")
    _ax.set_xlim(0, max(100, 100 * plot_data["score"].max() + 8))
    for _y, _score_value in enumerate(plot_data["score"]):
        _ax.text(100 * _score_value + 0.8, _y, f"{100 * _score_value:.1f}%", va="center", fontsize=8)
    fig.tight_layout()

    _leader = observed.iloc[0]
    _runner_up = observed.iloc[1] if len(observed) > 1 else _leader
    mo.vstack(
        [
            section(
                "What is the observed leaderboard?",
                "Which evaluated systems are highest scoring on the observed DeepSWE universe?",
                "A point leaderboard is the descriptive starting point, but it is not yet evidence that nearby ranks are meaningfully separated.",
                "Sort the selected ranked unit by observed score; use official leaderboard intervals for config-level views when available.",
                "Bars are observed scores. Error bars, when present, are official leaderboard intervals, not the task-bootstrap audit below.",
                [
                    f"Top observed unit: `{_leader['label']}` at {100 * _leader['score']:.1f}%.",
                    f"Observed rank-1 vs rank-2 gap: {100 * (_leader['score'] - _runner_up['score']):.1f} pp.",
                    f"The displayed score source is `{observed_source}`.",
                    "What to take away: the leaderboard is the object being audited, not the final inferential claim.",
                ],
                "Config/system is the default because it is what DeepSWE actually ran. Model-only aggregation is a sensitivity view.",
            ),
            mo.mpl.interactive(fig),
            mo.ui.table(
                tidy_table(
                    rank_summary[
                        [
                            "observed_rank",
                            "label",
                            "score",
                            "boot_score_p05",
                            "boot_score_p95",
                            "rank_p05",
                            "rank_p50",
                            "rank_p95",
                        ]
                    ].head(selected_top_n)
                ),
                pagination=False,
            ),
        ]
    )
    return fig


@app.cell
def _(
    PRACTICAL_EQUIVALENCE_PP,
    adjacent,
    all_pairs,
    max_t_critical,
    mo,
    nonseparation_tiers,
    np,
    plt,
    section,
    selected_top_n,
    sns,
    tidy_table,
):
    if adjacent.empty:
        resolution_view = adjacent
        fig_res, _ax = plt.subplots(figsize=(8, 3))
        _ax.text(0.5, 0.5, "No adjacent pairs available", ha="center", va="center")
        _ax.axis("off")
    else:
        resolution_view = adjacent.head(selected_top_n).copy()
        resolution_view["pair"] = resolution_view["rank_A"].astype(str) + " vs " + resolution_view["rank_B"].astype(str)
        fig_res, _ax = plt.subplots(figsize=(9, max(4, 0.32 * len(resolution_view))))
        _y = np.arange(len(resolution_view))
        _ax.axvspan(-PRACTICAL_EQUIVALENCE_PP, PRACTICAL_EQUIVALENCE_PP, color="gray", alpha=0.15, label="practical band")
        _ax.errorbar(
            resolution_view["gap_pp"],
            _y,
            xerr=[
                resolution_view["gap_pp"] - resolution_view["family_ci_lo_pp"].fillna(resolution_view["ci_lo_pp"]),
                resolution_view["family_ci_hi_pp"].fillna(resolution_view["ci_hi_pp"]) - resolution_view["gap_pp"],
            ],
            fmt="o",
            color="#F58518",
            ecolor="#9A6324",
            capsize=3,
        )
        _ax.axvline(0, color="black", linewidth=1)
        _ax.set_yticks(_y)
        _ax.set_yticklabels(resolution_view["pair"])
        _ax.invert_yaxis()
        _ax.set_xlabel("Adjacent gap (percentage points)")
        _ax.set_title("Adjacent-rank gaps with family-wise intervals")
        _ax.legend(loc="lower right")
        fig_res.tight_layout()

    if all_pairs.empty:
        fig_pair_heat, _pair_ax = plt.subplots(figsize=(8, 3))
        _pair_ax.text(0.5, 0.5, "No all-pairs comparisons available", ha="center", va="center")
        _pair_ax.axis("off")
    else:
        _pair_heat = all_pairs.pivot_table(
            index="A",
            columns="B",
            values="familywise_significant",
            aggfunc="max",
            fill_value=False,
        )
        _pair_heat = _pair_heat.astype(float)
        fig_pair_heat, _pair_ax = plt.subplots(figsize=(9, max(4, 0.36 * len(_pair_heat))))
        sns.heatmap(
            _pair_heat,
            cmap=sns.color_palette(["#F2F2F2", "#4C78A8"]),
            cbar=False,
            linewidths=0.5,
            linecolor="white",
            ax=_pair_ax,
        )
        _pair_ax.set_title(f"Top-{selected_top_n} all-pairs separated after max-T")
        _pair_ax.set_xlabel("lower observed rank")
        _pair_ax.set_ylabel("higher observed rank")
        fig_pair_heat.tight_layout()

    first_vs_rest = all_pairs[all_pairs["rank_A"] == 1].head(max(selected_top_n - 1, 1)).copy()
    if first_vs_rest.empty:
        fig_first, _first_ax = plt.subplots(figsize=(8, 3))
        _first_ax.text(0.5, 0.5, "No first-vs-rest comparisons available", ha="center", va="center")
        _first_ax.axis("off")
    else:
        _first_plot = first_vs_rest.iloc[::-1].copy()
        fig_first, _first_ax = plt.subplots(figsize=(9, max(4, 0.32 * len(_first_plot))))
        _first_ax.axvline(0, color="black", linewidth=1)
        _first_ax.errorbar(
            _first_plot["gap_pp"],
            np.arange(len(_first_plot)),
            xerr=[
                _first_plot["gap_pp"] - _first_plot["family_ci_lo_pp"].fillna(_first_plot["ci_lo_pp"]),
                _first_plot["family_ci_hi_pp"].fillna(_first_plot["ci_hi_pp"]) - _first_plot["gap_pp"],
            ],
            fmt="o",
            color="#4C78A8",
            ecolor="#2F4B7C",
            capsize=3,
        )
        _first_ax.set_yticks(np.arange(len(_first_plot)))
        _first_ax.set_yticklabels(_first_plot["B"])
        _first_ax.set_xlabel("Leader minus comparison config (percentage points)")
        _first_ax.set_title("First-vs-rest gaps with family-wise intervals")
        fig_first.tight_layout()

    tier_view = nonseparation_tiers.head(selected_top_n).copy()
    fig_tiers, _tier_ax = plt.subplots(figsize=(9, 2.2))
    if tier_view["non_separation_tier"].nunique() <= 1:
        _tier_ax.text(
            0.5,
            0.58,
            f"Top-{selected_top_n} adjacent ranks are not separated into multiple defensible tiers.",
            ha="center",
            va="center",
            fontsize=12,
            weight="bold",
        )
        _tier_ax.text(
            0.5,
            0.36,
            "Read this as insufficient resolution for exact local ranks, not as equality.",
            ha="center",
            va="center",
            fontsize=9,
        )
        _tier_ax.axis("off")
    else:
        _tier_counts = tier_view.groupby("non_separation_tier", as_index=False).agg(
            min_rank=("observed_rank", "min"),
            max_rank=("observed_rank", "max"),
            n_configs=("label", "count"),
        )
        _tier_ax.barh(_tier_counts["non_separation_tier"].astype(str), _tier_counts["n_configs"], color="#54A24B")
        for _, _tier_row in _tier_counts.iterrows():
            _tier_ax.text(
                _tier_row["n_configs"] + 0.05,
                str(_tier_row["non_separation_tier"]),
                f"ranks {_tier_row['min_rank']:.0f}-{_tier_row['max_rank']:.0f}",
                va="center",
                fontsize=8,
            )
        _tier_ax.set_xlabel("Configs in band")
        _tier_ax.set_ylabel("Non-separated band")
        _tier_ax.set_title("Non-separated rank bands")
    fig_tiers.tight_layout()

    _resolution_ambiguous = int(adjacent["within_practical_band"].sum()) if not adjacent.empty else 0
    _resolution_pointwise = int(adjacent["pointwise_ci_excludes_zero"].sum()) if not adjacent.empty else 0
    _resolution_family = int(adjacent["familywise_significant"].sum()) if not adjacent.empty else 0
    _all_pair_family = int(all_pairs["familywise_significant"].sum()) if not all_pairs.empty else 0
    _holm_sig = int(all_pairs["holm_significant"].sum()) if not all_pairs.empty else 0
    _bh_sig = int(all_pairs["bh_significant"].sum()) if not all_pairs.empty else 0
    _resolution_median_mde = float(adjacent["mde80_pp"].median()) if not adjacent.empty else float("nan")
    _n_tiers = int(nonseparation_tiers["non_separation_tier"].nunique()) if not nonseparation_tiers.empty else 0
    mo.vstack(
        [
            section(
                "Can nearby ranks be distinguished?",
                "Are nearby ranks, the leader, and top-N pairwise gaps statistically separable?",
                "Leaderboards imply more ordering than point estimates can usually justify. Resolution checks tell us which differences are actually visible at this benchmark size.",
                "Compute paired task deltas, bootstrap score gaps, pointwise intervals, MDE, max-T family-wise intervals, and secondary Holm/BH decisions.",
                "The gray band is a practical +/- gap band; intervals crossing zero are not separated. The heatmap marks pairs that survive max-T. The rank-band panel/table groups adjacent ranks that are not separated.",
                [
                    f"{_resolution_ambiguous} adjacent pairs are inside the +/- {PRACTICAL_EQUIVALENCE_PP:g} pp practical band.",
                    f"{_resolution_pointwise} adjacent-pair pointwise intervals exclude zero; {_resolution_family} survive max-T family-wise correction.",
                    f"Across top-{selected_top_n} all-pairs comparisons, {_all_pair_family} pairs survive family-wise correction.",
                    f"Secondary corrections flag {_holm_sig} Holm-significant and {_bh_sig} BH-significant top-{selected_top_n} pairs.",
                    f"Median pointwise 80% MDE is {_resolution_median_mde:.1f} pp.",
                    f"Adjacent non-separation produces {_n_tiers} coarse tiers; those tiers are safer to discuss than exact local ranks.",
                    "What to take away: read exact ranks only where the gap survives the interval and multiple-comparison checks.",
                ],
                "Non-separation is not equivalence. It means this eval does not provide enough evidence to separate those configs at this resolution.",
            ),
            mo.mpl.interactive(fig_res),
            mo.mpl.interactive(fig_pair_heat),
            mo.mpl.interactive(fig_first),
            mo.mpl.interactive(fig_tiers),
            mo.ui.table(
                tidy_table(tier_view[["observed_rank", "label", "non_separation_tier"]]),
                pagination=False,
            ),
            mo.ui.table(
                tidy_table(
                    resolution_view[
                        [
                            "rank_A",
                            "rank_B",
                            "A",
                            "B",
                            "gap_pp",
                            "family_ci_lo_pp",
                            "family_ci_hi_pp",
                            "mde80_pp",
                            "p_boot",
                            "win_probability",
                            "holm_significant",
                            "bh_significant",
                            "familywise_significant",
                            "within_practical_band",
                        ]
                    ]
                ),
                pagination=False,
            ),
            mo.accordion(
                {
                    "Non-separation tiers": mo.ui.table(
                        tidy_table(nonseparation_tiers.head(selected_top_n)), pagination=False
                    ),
                    "Full adjacent-pair audit table": mo.ui.table(tidy_table(adjacent), pagination=True, page_size=20),
                    "Full all-pairs top-N max-T audit": mo.ui.table(
                        tidy_table(all_pairs.drop(columns=["pair_key"], errors="ignore")),
                        pagination=True,
                        page_size=20,
                    ),
                    "First-vs-rest table": mo.ui.table(
                        tidy_table(first_vs_rest.drop(columns=["pair_key"], errors="ignore")),
                        pagination=False,
                    ),
                    "Max-T critical value": mo.md(f"`{max_t_critical:.2f}`"),
                }
            ),
        ]
    )
    return fig_first, fig_pair_heat, fig_res, fig_tiers, first_vs_rest, resolution_view


@app.cell
def _(boot_ranks, mo, np, observed, plt, rank_summary, section, selected_top_k, selected_top_n, sns, tidy_table):
    stability_view = rank_summary.head(selected_top_n).copy()
    fig_rank, _rank_ax = plt.subplots(figsize=(9, max(4, 0.34 * len(stability_view))))
    _rank_y = np.arange(len(stability_view))
    _rank_ax.errorbar(
        stability_view["rank_p50"],
        _rank_y,
        xerr=[
            stability_view["rank_p50"] - stability_view["rank_p05"],
            stability_view["rank_p95"] - stability_view["rank_p50"],
        ],
        fmt="o",
        color="#54A24B",
        ecolor="#2E6B2E",
        capsize=3,
    )
    _rank_ax.set_yticks(_rank_y)
    _rank_ax.set_yticklabels(stability_view["label"])
    _rank_ax.invert_yaxis()
    _rank_ax.invert_xaxis()
    _rank_ax.set_xlabel("Bootstrap rank (lower is better)")
    _rank_ax.set_title("Task-bootstrap rank intervals")
    fig_rank.tight_layout()

    boundary = observed.iloc[max(selected_top_k - 4, 0) : selected_top_k + 4][["system_id", "label", "observed_rank", "score"]].copy()
    prob_col = f"p_top{selected_top_k}"
    boundary = boundary.merge(rank_summary[["system_id", prob_col]], on="system_id", how="left")
    fig_topk, _topk_ax = plt.subplots(figsize=(8, 3.8))
    _topk_ax.bar(boundary["label"], boundary[prob_col], color="#72B7B2")
    _topk_ax.set_ylim(0, 1)
    _topk_ax.set_ylabel(f"P(in top {selected_top_k})")
    _topk_ax.set_title(f"Boundary stability around top {selected_top_k}")
    _topk_ax.tick_params(axis="x", rotation=35, labelsize=8)
    fig_topk.tight_layout()

    _rank_heat = (
        boot_ranks[stability_view["system_id"].tolist()]
        .apply(lambda col: col.value_counts(normalize=True), axis=0)
        .fillna(0)
        .sort_index()
        .T
    )
    _rank_heat.index = stability_view.set_index("system_id").loc[_rank_heat.index, "label"]
    fig_rank_heat, _heat_ax = plt.subplots(figsize=(9, max(4, 0.35 * len(_rank_heat))))
    sns.heatmap(
        100 * _rank_heat,
        cmap="Blues",
        cbar_kws={"label": "Bootstrap probability (%)"},
        ax=_heat_ax,
    )
    _heat_ax.set_title("Bootstrap rank distribution")
    _heat_ax.set_xlabel("Rank")
    _heat_ax.set_ylabel("")
    fig_rank_heat.tight_layout()

    observed_top = set(observed.head(selected_top_k)["system_id"])
    overlaps = []
    for _, _boot_row in boot_ranks.iterrows():
        overlaps.append(len(observed_top & set(_boot_row.sort_values().head(selected_top_k).index)))
    mean_overlap = float(np.mean(overlaps)) if overlaps else float("nan")
    full_overlap = float(np.mean(np.array(overlaps) == selected_top_k)) if overlaps else float("nan")

    mo.vstack(
        [
            section(
                "Are ranks stable under task resampling?",
                "Would the same systems keep similar ranks if the benchmark sampled a different but related task set?",
                "Exact ranks are useful only if they survive task-composition uncertainty. Top-K boundary probabilities reveal which inclusion decisions are robust or fragile.",
                "Bootstrap tasks with replacement, recompute scores/ranks, then summarize rank intervals and top-K membership probabilities.",
                "Narrow intervals and dark heatmap mass near one rank indicate stable placement. Partial top-K probabilities near the cutoff indicate fragile boundary membership.",
                [
                    f"Mean overlap with observed top {selected_top_k}: {mean_overlap:.1f} of {selected_top_k}.",
                    f"Full top-{selected_top_k} agreement occurs in {100 * full_overlap:.1f}% of bootstrap draws.",
                    "Wide rank intervals mean local ordering is unstable even when coarse quality groups are visible.",
                    "What to take away: robust systems stay in the same rank band; fragile ones hover around the top-K boundary.",
                ],
                "The bootstrap is a task-population sensitivity analysis over the observed task set, not proof that this exact task list was random.",
            ),
            mo.mpl.interactive(fig_rank),
            mo.mpl.interactive(fig_topk),
            mo.mpl.interactive(fig_rank_heat),
            mo.ui.table(tidy_table(boundary), pagination=False),
        ]
    )
    return boundary, fig_rank, fig_rank_heat, fig_topk


@app.cell
def _(
    MIN_SLICE_SYSTEMS,
    cell_scores,
    mo,
    np,
    observed,
    pd,
    plt,
    scored_trials,
    section,
    selected_top_n,
    slice_dimension,
    slice_min_tasks,
    sns,
    tidy_table,
):
    dim = slice_dimension.value
    min_slice_tasks = int(slice_min_tasks.value)
    slice_rows = []
    skipped_rows = []
    slice_delta_rows = []
    top_systems = cell_scores.groupby("system_id")["score"].mean().sort_values(ascending=False).head(8).index.tolist()
    top_pair = observed.head(2)["system_id"].tolist()
    if dim in scored_trials.columns:
        for _slice_value, _slice_group in scored_trials.dropna(subset=[dim]).groupby(dim, observed=False):
            _slice_n_tasks = _slice_group["task_id"].nunique()
            _slice_n_systems = _slice_group["system_id"].nunique()
            if _slice_n_tasks < min_slice_tasks or _slice_n_systems < MIN_SLICE_SYSTEMS:
                skipped_rows.append({"dimension": dim, "slice": str(_slice_value), "n_tasks": _slice_n_tasks, "n_systems": _slice_n_systems})
                continue
            by_system = (
                _slice_group.groupby(["task_id", "system_id", "system_label"], dropna=False)["score"]
                .mean()
                .reset_index()
                .groupby(["system_id", "system_label"], dropna=False)
                .agg(score=("score", "mean"), se=("score", lambda s: s.std(ddof=1) / np.sqrt(len(s)) if len(s) > 1 else np.nan))
                .reset_index()
            )
            for _, _slice_row in by_system.sort_values("score", ascending=False).head(3).iterrows():
                slice_rows.append(
                    {
                        "dimension": dim,
                        "slice": str(_slice_value),
                        "n_tasks": _slice_n_tasks,
                        "n_systems": _slice_n_systems,
                        "system_id": _slice_row["system_id"],
                        "label": _slice_row["system_label"],
                        "score_pct": 100 * _slice_row["score"],
                        "se_pp": 100 * _slice_row["se"],
                    }
                )
            if len(top_pair) == 2:
                _slice_task_scores = (
                    _slice_group[_slice_group["system_id"].isin(top_pair)]
                    .groupby(["task_id", "system_id"], dropna=False)["score"]
                    .mean()
                    .reset_index()
                    .pivot_table(index="task_id", columns="system_id", values="score", aggfunc="mean")
                )
                if all(_slice_system_id in _slice_task_scores.columns for _slice_system_id in top_pair):
                    _delta = (_slice_task_scores[top_pair[0]] - _slice_task_scores[top_pair[1]]).dropna()
                    if len(_delta) >= 2:
                        _se = float(_delta.std(ddof=1) / np.sqrt(len(_delta)))
                        slice_delta_rows.append(
                            {
                                "slice": str(_slice_value),
                                "n_tasks": int(len(_delta)),
                                "A": by_system.loc[by_system["system_id"].eq(top_pair[0]), "system_label"].iloc[0]
                                if by_system["system_id"].eq(top_pair[0]).any()
                                else top_pair[0],
                                "B": by_system.loc[by_system["system_id"].eq(top_pair[1]), "system_label"].iloc[0]
                                if by_system["system_id"].eq(top_pair[1]).any()
                                else top_pair[1],
                                "delta_pp": 100 * float(_delta.mean()),
                                "ci_lo_pp": 100 * float(_delta.mean() - 1.96 * _se),
                                "ci_hi_pp": 100 * float(_delta.mean() + 1.96 * _se),
                            }
                        )
    slice_leaders = pd.DataFrame(slice_rows)
    skipped_slices = pd.DataFrame(skipped_rows)
    slice_pair_deltas = pd.DataFrame(slice_delta_rows)
    repo_task_counts = (
        scored_trials[["repository", "task_id"]]
        .dropna()
        .drop_duplicates()
        .groupby("repository", observed=False)
        .size()
        .rename("n_tasks")
        .reset_index()
        .sort_values("n_tasks", ascending=False)
        if "repository" in scored_trials.columns
        else pd.DataFrame(columns=["repository", "n_tasks"])
    )
    repo_sparse_note = (
        mo.md(
            f"**Repository sparsity note.** DeepSWE has many one-off repositories: "
            f"{int((repo_task_counts['n_tasks'] < 5).sum())} repositories have fewer than 5 tasks, "
            f"so repository slices are mostly exploratory unless you lower the task threshold."
        )
        if dim == "repository" and not repo_task_counts.empty
        else mo.md("")
    )

    if slice_leaders.empty:
        fig_slice, _slice_ax = plt.subplots(figsize=(8, 3))
        _slice_ax.text(0.5, 0.5, f"No powered slices for {dim}", ha="center", va="center")
        _slice_ax.axis("off")
    else:
        major_slices = (
            slice_leaders[["slice", "n_tasks"]]
            .drop_duplicates()
            .sort_values("n_tasks", ascending=False)
            .head(8)["slice"]
            .tolist()
        )
        heat = (
            slice_leaders[slice_leaders["slice"].isin(major_slices) & slice_leaders["system_id"].isin(top_systems)]
            .pivot_table(index="label", columns="slice", values="score_pct", aggfunc="max")
        )
        fig_slice, _slice_ax = plt.subplots(figsize=(8.5, max(4, 0.42 * len(heat))))
        sns.heatmap(heat, annot=True, fmt=".1f", cmap="Blues", cbar_kws={"label": "Score (%)"}, ax=_slice_ax)
        _slice_ax.set_title(f"Slice leaders by {dim}; powered slices only")
        _slice_ax.set_xlabel(dim)
        _slice_ax.set_ylabel("system")
        _slice_ax.tick_params(axis="x", rotation=0)
        _slice_ax.tick_params(axis="y", rotation=0)
        fig_slice.tight_layout()

    if slice_pair_deltas.empty:
        fig_slice_delta, _delta_ax = plt.subplots(figsize=(8, 3))
        _delta_ax.text(0.5, 0.5, "No powered top-pair slice deltas", ha="center", va="center")
        _delta_ax.axis("off")
    else:
        _delta_plot = slice_pair_deltas.sort_values("n_tasks", ascending=False).head(10).iloc[::-1].copy()
        fig_slice_delta, _delta_ax = plt.subplots(figsize=(9, max(4, 0.33 * len(_delta_plot))))
        _delta_ax.axvline(0, color="black", linewidth=1)
        _delta_ax.errorbar(
            _delta_plot["delta_pp"],
            np.arange(len(_delta_plot)),
            xerr=[
                _delta_plot["delta_pp"] - _delta_plot["ci_lo_pp"],
                _delta_plot["ci_hi_pp"] - _delta_plot["delta_pp"],
            ],
            fmt="o",
            color="#E45756",
            ecolor="#8C2D2D",
            capsize=3,
        )
        _delta_ax.set_yticks(np.arange(len(_delta_plot)))
        _delta_ax.set_yticklabels(_delta_plot["slice"])
        _delta_a = _delta_plot["A"].iloc[0]
        _delta_b = _delta_plot["B"].iloc[0]
        _delta_ax.set_xlabel(f"Delta in pp; positive = {_delta_a} beats {_delta_b}")
        _delta_ax.set_title(f"Top-pair delta by {dim}: {_delta_a} vs {_delta_b}")
        fig_slice_delta.tight_layout()

    powered = int(slice_leaders[["dimension", "slice"]].drop_duplicates().shape[0]) if not slice_leaders.empty else 0
    skipped = int(len(skipped_slices))
    mo.vstack(
        [
            section(
                "Where does performance differ by slice?",
                f"Do aggregate ranks hide different behavior across `{dim}` slices?",
                "A single average can hide domain-specific strengths, weaknesses, and unstable local ordering.",
                "Compute slice-level task means only for slices with enough tasks and systems; separately compare the observed top two configs within powered slices.",
                "The heatmap shows which systems lead powered slices. The delta chart shows whether the top-pair gap changes sign or size across slices.",
                [
                    f"{powered} slices pass the minimum coverage rule: at least {min_slice_tasks} tasks and {MIN_SLICE_SYSTEMS} systems.",
                    f"{skipped} slices are marked underpowered and kept out of the main comparison.",
                    "Lower thresholds are useful for exploration; higher thresholds are safer for inferential claims.",
                    f"What to take away: compare slice rows only when their task counts are large enough to make the local average meaningful.",
                ],
                "Slice results are descriptive unless enough tasks support that slice. With very small thresholds, treat slice differences as hypothesis-generating.",
            ),
            repo_sparse_note,
            mo.ui.table(tidy_table(repo_task_counts.head(12)), pagination=False) if dim == "repository" else mo.md(""),
            mo.mpl.interactive(fig_slice),
            mo.mpl.interactive(fig_slice_delta),
            mo.ui.table(tidy_table(slice_leaders.head(selected_top_n)), pagination=False),
            mo.accordion(
                {
                    "Underpowered slice examples": mo.ui.table(tidy_table(skipped_slices.head(20)), pagination=False),
                    "Top-pair slice deltas": mo.ui.table(tidy_table(slice_pair_deltas), pagination=True, page_size=20),
                    "Full powered slice table": mo.ui.table(tidy_table(slice_leaders), pagination=True, page_size=20),
                }
            ),
        ]
    )
    return fig_slice, fig_slice_delta, skipped_slices, slice_leaders, slice_pair_deltas


@app.cell
def _(cell_scores, matrix, mo, observed, plt, scored_trials, section, selected_top_n, sns, tidy_table):
    top_systems_for_influence = observed.head(10)["system_id"].tolist()
    baseline_ranks = observed.set_index("system_id")["observed_rank"]
    influence_rows = []

    def rank_without(filtered_matrix):
        scores = filtered_matrix.mean(axis=0, skipna=True)
        return scores.rank(method="min", ascending=False)


    for _task_id in matrix.index:
        _ranks = rank_without(matrix.drop(index=_task_id))
        _shifts = (_ranks.reindex(top_systems_for_influence) - baseline_ranks.reindex(top_systems_for_influence)).abs()
        influence_rows.append(
            {
                "unit": "task",
                "value": _task_id,
                "n_trials_removed": int((scored_trials["task_id"] == _task_id).sum()),
                "max_abs_rank_shift_top10": float(_shifts.max()),
                "mean_abs_rank_shift_top10": float(_shifts.mean()),
                "affected_top10_count": int((_shifts.fillna(0) > 0).sum()),
            }
        )

    if "repository" in scored_trials.columns:
        task_repo = scored_trials[["task_id", "repository"]].dropna().drop_duplicates()
        for repo, repo_tasks in task_repo.groupby("repository", observed=False):
            remaining_tasks = [task for task in matrix.index if task not in set(repo_tasks["task_id"])]
            if len(remaining_tasks) < 5:
                continue
            _ranks = rank_without(matrix.loc[remaining_tasks])
            _shifts = (_ranks.reindex(top_systems_for_influence) - baseline_ranks.reindex(top_systems_for_influence)).abs()
            influence_rows.append(
                {
                    "unit": "repository",
                    "value": repo,
                    "n_trials_removed": int(scored_trials["repository"].eq(repo).sum()),
                    "max_abs_rank_shift_top10": float(_shifts.max()),
                    "mean_abs_rank_shift_top10": float(_shifts.mean()),
                    "affected_top10_count": int((_shifts.fillna(0) > 0).sum()),
                }
            )
    influence = (
        pd.DataFrame(influence_rows)
        .sort_values(["max_abs_rank_shift_top10", "mean_abs_rank_shift_top10"], ascending=False)
        .reset_index(drop=True)
    )
    _influence_chart = influence[influence["max_abs_rank_shift_top10"] > 0].head(selected_top_n).iloc[::-1].copy()
    _influence_chart["label"] = _influence_chart["unit"] + ": " + _influence_chart["value"].astype(str).str.slice(0, 48)
    fig_inf, _inf_ax = plt.subplots(figsize=(9, max(4, 0.34 * len(_influence_chart))))
    if _influence_chart.empty:
        _inf_ax.text(0.5, 0.5, "No leave-one-out unit shifts the observed top-10 ranks.", ha="center", va="center")
        _inf_ax.axis("off")
    else:
        _colors = _influence_chart["unit"].map({"task": "#E45756", "repository": "#F58518"}).fillna("#888")
        _inf_ax.barh(_influence_chart["label"], _influence_chart["max_abs_rank_shift_top10"], color=_colors)
        _inf_ax.set_xlabel("Max absolute rank shift among observed top 10")
        _inf_ax.set_title("Highest-leverage tasks and repositories")
        _inf_ax.legend(
            handles=[
                plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#E45756", markersize=9, label="task"),
                plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#F58518", markersize=9, label="repository"),
            ],
            loc="lower right",
            frameon=False,
        )
    fig_inf.tight_layout()

    task_matrix = matrix.copy()
    task_diagnostics = pd.DataFrame(
        {
            "task_id": task_matrix.index,
            "difficulty_score": task_matrix.mean(axis=1, skipna=True).values,
            "discrimination_spread": (task_matrix.max(axis=1, skipna=True) - task_matrix.min(axis=1, skipna=True)).values,
            "systems_with_result": task_matrix.notna().sum(axis=1).values,
        }
    )
    if "task_id" in scored_trials.columns:
        _task_meta_cols = [col for col in ["task_id", "language", "repository"] if col in scored_trials.columns]
        _task_meta = scored_trials[_task_meta_cols].drop_duplicates("task_id")
        task_diagnostics = task_diagnostics.merge(_task_meta, on="task_id", how="left")
        _within_task_var = (
            scored_trials.groupby(["task_id", "system_id"], observed=False)["score"]
            .var(ddof=1)
            .groupby("task_id")
            .mean()
            .rename("mean_within_run_var")
            .reset_index()
        )
        task_diagnostics = task_diagnostics.merge(_within_task_var, on="task_id", how="left")
    task_influence = influence[influence["unit"].eq("task")][
        ["value", "max_abs_rank_shift_top10", "mean_abs_rank_shift_top10", "affected_top10_count"]
    ].rename(columns={"value": "task_id"})
    task_diagnostics = task_diagnostics.merge(task_influence, on="task_id", how="left")
    task_diagnostics["max_abs_rank_shift_top10"] = task_diagnostics["max_abs_rank_shift_top10"].fillna(0)
    task_diagnostics["affected_top10_count"] = task_diagnostics["affected_top10_count"].fillna(0)
    _diagnostic_plot = task_diagnostics.sort_values(
        ["max_abs_rank_shift_top10", "discrimination_spread"], ascending=False
    ).head(selected_top_n).iloc[::-1].copy()
    _diagnostic_plot["short_task"] = _diagnostic_plot["task_id"].astype(str).str.slice(0, 42)
    fig_task_diag, _task_ax = plt.subplots(figsize=(9, max(4, 0.34 * len(_diagnostic_plot))))
    if _diagnostic_plot.empty:
        _task_ax.text(0.5, 0.5, "No task diagnostics available", ha="center", va="center")
        _task_ax.axis("off")
    else:
        _colors = (
            _diagnostic_plot["language"].astype(str).map(
                {"go": "#4C78A8", "python": "#F58518", "typescript": "#54A24B", "rust": "#E45756", "javascript": "#B279A2"}
            )
            if "language" in _diagnostic_plot.columns
            else None
        )
        _task_ax.barh(
            _diagnostic_plot["short_task"],
            _diagnostic_plot["max_abs_rank_shift_top10"],
            color=_colors.fillna("#888") if _colors is not None else "#4C78A8",
        )
        _task_ax.set_xlabel("Max absolute rank shift among observed top 10")
        _task_ax.set_title("Top-leverage tasks")
        if "language" in _diagnostic_plot.columns:
            _handles = [
                plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=color, markersize=9, label=language)
                for language, color in {
                    "go": "#4C78A8",
                    "python": "#F58518",
                    "typescript": "#54A24B",
                    "rust": "#E45756",
                    "javascript": "#B279A2",
                }.items()
                if language in set(_diagnostic_plot["language"].astype(str))
            ]
            _task_ax.legend(handles=_handles, loc="lower right", frameon=False, fontsize=8)
    fig_task_diag.tight_layout()

    _highest = influence.iloc[0]
    mo.vstack(
        [
            section(
                "Which tasks/domains are highest leverage?",
                "Which individual tasks or repositories materially move the aggregate leaderboard?",
                "If a conclusion depends on a few units, the benchmark may still have signal, but the exact ordering is fragile.",
                "Remove one task or repository at a time, recompute ranks, and measure rank shifts among the observed top 10. Also inspect task difficulty and discrimination.",
                "The first chart shows units that move ranks. The second chart shows which tasks have the most leverage; difficulty/discrimination details are in the table.",
                [
                    f"Highest leverage unit: `{_highest['unit']}: {_highest['value']}`; max top-10 shift is {_highest['max_abs_rank_shift_top10']:.1f} ranks.",
                    f"It affects {_highest['affected_top10_count']:.0f} of the observed top-10 configs.",
                    "What to take away: many one-rank shifts mean local swaps near rank boundaries, not a wholesale leaderboard collapse.",
                    "This is an influence check, not a claim that the task/domain is bad.",
                    "High leverage means the aggregate result is sensitive to that unit's inclusion.",
                ],
                "Leave-one-out influence is descriptive sensitivity analysis; it does not say the removed unit is invalid.",
            ),
            mo.mpl.interactive(fig_inf),
            mo.mpl.interactive(fig_task_diag),
            mo.ui.table(tidy_table(influence.head(selected_top_n)), pagination=False),
            mo.accordion(
                {
                    "Task diagnostics": mo.ui.table(
                        tidy_table(
                            task_diagnostics.sort_values("max_abs_rank_shift_top10", ascending=False)
                            .assign(
                                difficulty_pct=lambda df: 100 * df["difficulty_score"],
                                discrimination_pp=lambda df: 100 * df["discrimination_spread"],
                            )
                            .head(30)
                        ),
                        pagination=False,
                    ),
                    "Full influence table": mo.ui.table(tidy_table(influence), pagination=True, page_size=20),
                }
            ),
        ]
    )
    return fig_inf, fig_task_diag, influence, task_diagnostics


@app.cell
def _(RNG_SEED, boot_scores, cell_scores, mo, np, observed, pd, plt, scored_trials, section, selected_top_n, tidy_table):
    between = cell_scores.groupby("system_id")["score"].var(ddof=1).rename("between_task_var").reset_index()
    within_rows = []
    for (_cell_task_id, _cell_system_id), _run_group in scored_trials.groupby(["task_id", "system_id"], observed=False):
        if len(_run_group) >= 2:
            within_rows.append(
                {
                    "task_id": _cell_task_id,
                    "system_id": _cell_system_id,
                    "within_run_var": float(_run_group["score"].var(ddof=1)),
                    "n_runs": int(len(_run_group)),
                }
            )
    within = pd.DataFrame(within_rows)
    within_by_system = (
        within.groupby("system_id")
        .agg(mean_within_run_var=("within_run_var", "mean"), median_runs_per_cell=("n_runs", "median"))
        .reset_index()
    )
    variance_budget = between.merge(within_by_system, on="system_id", how="left")
    variance_budget["task_share_proxy"] = variance_budget["between_task_var"] / (
        variance_budget["between_task_var"] + variance_budget["mean_within_run_var"].fillna(0)
    )

    _variance_n_tasks = scored_trials["task_id"].nunique()
    median_between = float(variance_budget["between_task_var"].median())
    median_within = float(variance_budget["mean_within_run_var"].median())
    k_rows = []
    for k in [1, 2, 4, 8, 16]:
        _k_se = np.sqrt((median_between + median_within / k) / _variance_n_tasks)
        k_rows.append(
            {
                "runs_per_task": k,
                "median_se_pp": 100 * _k_se,
                "mde80_pp": 100 * (1.96 + 0.84) * _k_se,
                "within_share_proxy": (median_within / k) / (median_between + median_within / k),
            }
        )
    k_simulation = pd.DataFrame(k_rows)
    _mean_trial_cost = float(scored_trials["cost_usd"].mean()) if "cost_usd" in scored_trials.columns else np.nan
    _cost_power_rows = []
    if not np.isnan(_mean_trial_cost):
        _cost_n_tasks = scored_trials["task_id"].nunique()
        _cost_n_systems = scored_trials["system_id"].nunique()
        _baseline_mde = float(k_simulation.loc[k_simulation["runs_per_task"].eq(1), "mde80_pp"].iloc[0])
        for _, _k_row in k_simulation.iterrows():
            _extra_runs_per_cell = max(int(_k_row["runs_per_task"]) - 1, 0)
            _incremental_cost = _extra_runs_per_cell * _cost_n_tasks * _cost_n_systems * _mean_trial_cost
            _mde_reduction = _baseline_mde - float(_k_row["mde80_pp"])
            _cost_power_rows.append(
                {
                    "runs_per_task": int(_k_row["runs_per_task"]),
                    "approx_incremental_cost_usd": _incremental_cost,
                    "mde80_pp": float(_k_row["mde80_pp"]),
                    "mde_reduction_pp": _mde_reduction,
                    "mde_pp_reduced_per_100_usd": 100 * _mde_reduction / _incremental_cost if _incremental_cost > 0 else np.nan,
                    "usd_per_1pp_mde_reduction": _incremental_cost / _mde_reduction if _mde_reduction > 0 else np.nan,
                }
            )
    cost_power = pd.DataFrame(_cost_power_rows)

    fig_k, _k_ax = plt.subplots(figsize=(7.5, 4))
    _k_ax.plot(k_simulation["runs_per_task"], k_simulation["median_se_pp"], marker="o", label="Median SE")
    _k_ax.plot(k_simulation["runs_per_task"], k_simulation["mde80_pp"], marker="o", label="80% MDE")
    _k_ax.set_xscale("log", base=2)
    _k_ax.set_xticks(k_simulation["runs_per_task"])
    _k_ax.set_xticklabels(k_simulation["runs_per_task"])
    _k_ax.set_xlabel("Runs per task")
    _k_ax.set_ylabel("Percentage points")
    _k_ax.set_title("Estimated returns to repeated runs")
    _k_ax.legend()
    _k_ax.annotate(
        "flatter curve = rollouts help less",
        xy=(8, k_simulation.loc[k_simulation["runs_per_task"].eq(8), "mde80_pp"].iloc[0]),
        xytext=(3, k_simulation["mde80_pp"].max() * 0.92),
        arrowprops={"arrowstyle": "->", "color": "#555"},
        fontsize=8,
        color="#333",
    )
    fig_k.tight_layout()

    top_system_ids = observed.head(selected_top_n)["system_id"].tolist()
    interval_rows = []
    nested_draws = 400
    rng_nested = np.random.default_rng(RNG_SEED + 1000)
    all_tasks = sorted(scored_trials["task_id"].dropna().unique())
    for _system_id in top_system_ids:
        _system_rows = scored_trials[scored_trials["system_id"].eq(_system_id)]
        if _system_rows.empty:
            continue
        _score = float(_system_rows["score"].mean())
        _naive_se = float(_system_rows["score"].std(ddof=1) / np.sqrt(len(_system_rows))) if len(_system_rows) > 1 else np.nan
        _task_boot = boot_scores[_system_id].dropna() if _system_id in boot_scores else pd.Series(dtype=float)
        _by_task = {
            task_id: group["score"].dropna().to_numpy(dtype=float)
            for task_id, group in _system_rows.groupby("task_id", observed=False)
        }
        nested_scores = []
        for _ in range(nested_draws):
            sampled_tasks = rng_nested.choice(all_tasks, size=len(all_tasks), replace=True)
            sampled_scores = []
            for task_id in sampled_tasks:
                values = _by_task.get(task_id)
                if values is not None and len(values):
                    sampled_scores.append(float(rng_nested.choice(values)))
            nested_scores.append(np.mean(sampled_scores) if sampled_scores else np.nan)
        nested_scores = pd.Series(nested_scores).dropna()
        interval_rows.append(
            {
                "system_id": _system_id,
                "label": observed.loc[observed["system_id"].eq(_system_id), "label"].iloc[0],
                "score_pct": 100 * _score,
                "naive_lo_pp": 100 * (_score - 1.96 * _naive_se) if not np.isnan(_naive_se) else np.nan,
                "naive_hi_pp": 100 * (_score + 1.96 * _naive_se) if not np.isnan(_naive_se) else np.nan,
                "task_boot_lo_pp": 100 * float(_task_boot.quantile(0.025)) if len(_task_boot) else np.nan,
                "task_boot_hi_pp": 100 * float(_task_boot.quantile(0.975)) if len(_task_boot) else np.nan,
                "nested_lo_pp": 100 * float(nested_scores.quantile(0.025)) if len(nested_scores) else np.nan,
                "nested_hi_pp": 100 * float(nested_scores.quantile(0.975)) if len(nested_scores) else np.nan,
                "n_trials": int(len(_system_rows)),
                "n_tasks": int(_system_rows["task_id"].nunique()),
            }
        )
    interval_comparison = pd.DataFrame(interval_rows)
    if interval_comparison.empty:
        fig_interval, _int_ax = plt.subplots(figsize=(8, 3))
        _int_ax.text(0.5, 0.5, "No interval comparison available", ha="center", va="center")
        _int_ax.axis("off")
    else:
        _interval_plot = interval_comparison.head(min(selected_top_n, 10)).iloc[::-1].copy()
        _y = np.arange(len(_interval_plot))
        fig_interval, _int_ax = plt.subplots(figsize=(9, max(4, 0.36 * len(_interval_plot))))
        for _offset, _lo_col, _hi_col, _label, _color in [
            (-0.18, "naive_lo_pp", "naive_hi_pp", "naive trial rows", "#E45756"),
            (0.0, "task_boot_lo_pp", "task_boot_hi_pp", "task bootstrap", "#4C78A8"),
            (0.18, "nested_lo_pp", "nested_hi_pp", "nested bootstrap", "#54A24B"),
        ]:
            _mid = 100 * _interval_plot["score_pct"] / 100
            _int_ax.errorbar(
                _interval_plot["score_pct"],
                _y + _offset,
                xerr=[
                    _interval_plot["score_pct"] - _interval_plot[_lo_col],
                    _interval_plot[_hi_col] - _interval_plot["score_pct"],
                ],
                fmt="o",
                color=_color,
                ecolor=_color,
                capsize=2,
                label=_label,
                alpha=0.85,
            )
        _int_ax.set_yticks(_y)
        _int_ax.set_yticklabels(_interval_plot["label"])
        _int_ax.set_xlabel("Score interval (%)")
        _int_ax.set_title("Naive vs task-clustered vs nested intervals")
        _int_ax.legend(loc="lower right", fontsize=8)
        fig_interval.tight_layout()

    median_task_share = float(variance_budget["task_share_proxy"].median())
    _best_cost_power = (
        cost_power.dropna(subset=["mde_pp_reduced_per_100_usd"]).sort_values("mde_pp_reduced_per_100_usd", ascending=False).head(1)
        if not cost_power.empty
        else pd.DataFrame()
    )
    _best_cost_power_text = (
        f"Best approximate return is K={int(_best_cost_power['runs_per_task'].iloc[0])}: "
        f"{_best_cost_power['mde_pp_reduced_per_100_usd'].iloc[0]:.2f} MDE pp reduced per $100."
        if not _best_cost_power.empty
        else "Cost-power cannot be estimated because `cost_usd` is unavailable."
    )
    mo.vstack(
        [
            section(
                "Would one more run help?",
                "Is uncertainty mostly from task composition or repeated rollout noise, and would more runs reduce it?",
                "Repeating the same tasks does not buy the same information as adding new tasks if task composition dominates.",
                "Compare between-task variance with within-task repeated-run variance; contrast naive row-level intervals with task/nested bootstrap intervals; simulate MDE as runs per task increase.",
                "If task/nested intervals are wider than naive intervals, raw trial rows overstate precision. If the K-curve flattens, extra rollouts have diminishing returns.",
                [
                    f"Median task-share proxy is {100 * median_task_share:.1f}%.",
                    f"Estimated MDE falls from {k_simulation['mde80_pp'].iloc[0]:.1f} pp at 1 run/task to {k_simulation['mde80_pp'].iloc[-1]:.1f} pp at 16 runs/task.",
                    "If the curve flattens, new or better-balanced tasks are likely more valuable than more rollouts.",
                    _best_cost_power_text,
                    "What to take away: compare the K-curve with the interval chart before buying more repeated runs.",
                ],
                "The nested and cost-power views are approximate for DeepSWE because repeated runs are not necessarily a complete crossed design, and harness is constant.",
            ),
            mo.mpl.interactive(fig_interval),
            mo.mpl.interactive(fig_k),
            mo.ui.table(tidy_table(interval_comparison.head(selected_top_n)), pagination=False),
            mo.ui.table(tidy_table(k_simulation), pagination=False),
            mo.ui.table(tidy_table(cost_power), pagination=False) if not cost_power.empty else mo.md("`cost_usd` unavailable; cost-power table skipped."),
            mo.accordion({"Variance budget by system": mo.ui.table(tidy_table(variance_budget), pagination=True, page_size=20)}),
        ]
    )
    return cost_power, fig_interval, fig_k, interval_comparison, k_simulation, variance_budget


@app.cell
def _(matrix, mo, pd, plt, scored_trials, section, skipped_slices, tidy_table):
    def effective_count(series):
        shares = series.value_counts(normalize=True, dropna=True)
        return float(1 / shares.pow(2).sum()) if len(shares) else float("nan")


    _task_meta = scored_trials.drop_duplicates("task_id")
    _system_meta = scored_trials.drop_duplicates("system_id")
    task_dims = [c for c in ["repository", "language"] if c in _task_meta.columns]
    system_dims = [c for c in ["provider", "model", "reasoning_effort", "harness"] if c in _system_meta.columns]

    def imbalance_rows(frame, dims, universe):
        return [
            {
                "universe": universe,
                "dimension": dim,
                "n_values": frame[dim].nunique(dropna=True),
                "largest_share_pct": 100 * frame[dim].value_counts(normalize=True, dropna=True).iloc[0],
                "effective_count": effective_count(frame[dim]),
            }
            for dim in dims
            if frame[dim].notna().any()
        ]


    task_imbalance = pd.DataFrame(imbalance_rows(_task_meta, task_dims, "task"))
    system_imbalance = pd.DataFrame(imbalance_rows(_system_meta, system_dims, "system/config"))
    imbalance = pd.concat([task_imbalance, system_imbalance], ignore_index=True)

    fig_task_imb, _task_imb_ax = plt.subplots(figsize=(6.5, 3.4))
    _task_imb_ax.bar(task_imbalance["dimension"], task_imbalance["largest_share_pct"], color="#B279A2")
    _task_imb_ax.set_ylabel("Largest value share (%)")
    _task_imb_ax.set_title("Task universe concentration")
    fig_task_imb.tight_layout()

    fig_system_imb, _system_imb_ax = plt.subplots(figsize=(7.5, 3.4))
    _system_imb_ax.bar(system_imbalance["dimension"], system_imbalance["largest_share_pct"], color="#59A14F")
    _system_imb_ax.set_ylabel("Largest value share (%)")
    _system_imb_ax.set_title("System/config variation")
    _system_imb_ax.tick_params(axis="x", rotation=20)
    fig_system_imb.tight_layout()

    total_cells = int(matrix.shape[0] * matrix.shape[1])
    _observed_cells = int(matrix.notna().sum().sum())
    _missing_cells = total_cells - _observed_cells
    coverage_summary = pd.DataFrame(
        [
            {
                "tasks": int(matrix.shape[0]),
                "systems": int(matrix.shape[1]),
                "possible_system_task_cells": total_cells,
                "observed_system_task_cells": _observed_cells,
                "missing_system_task_cells": _missing_cells,
                "coverage_pct": 100 * _observed_cells / total_cells if total_cells else float("nan"),
            }
        ]
    )

    _imbalance_most_concentrated = imbalance.sort_values("largest_share_pct", ascending=False).iloc[0]
    mo.vstack(
        [
            section(
                "Is the eval universe imbalanced or underpowered?",
                "Does the benchmark universe concentrate evidence in a few domains, and are some comparisons underpowered?",
                "A leaderboard can look precise while leaning heavily on one repo/domain or while hiding missing config-task cells.",
                "Measure effective counts, largest-dimension shares, config-task coverage, missing cells, and underpowered slice counts.",
                "The first chart describes task/domain concentration; the second describes which config factors vary. The coverage table shows how complete the config-task comparison grid is.",
                [
                    f"Most concentrated dimension: `{_imbalance_most_concentrated['dimension']}`; largest value share is {_imbalance_most_concentrated['largest_share_pct']:.1f}%.",
                    f"Config-task coverage is {coverage_summary['coverage_pct'].iloc[0]:.1f}% with {_missing_cells} missing system-task cells.",
                    f"There are {len(skipped_slices)} underpowered slice rows in the selected slice view.",
                    "Underpowered means too few tasks or systems to support a slice-level comparison without fake precision.",
                    "What to take away: task dimensions and config dimensions answer different questions and should not be read as one pooled universe.",
                ],
                "Global coverage can be high while particular slices are still too small for responsible slice claims.",
            ),
            mo.mpl.interactive(fig_task_imb),
            mo.mpl.interactive(fig_system_imb),
            mo.ui.table(tidy_table(imbalance), pagination=False),
            mo.ui.table(tidy_table(coverage_summary), pagination=False),
        ]
    )
    return coverage_summary, effective_count, fig_system_imb, fig_task_imb, imbalance


@app.cell
def _(mo, observed, pd, plt, scored_trials, section, selected_top_n, tidy_table):
    reliability_cols = [
        c
        for c in [
            "cost_usd",
            "n_agent_steps",
            "agent_duration_seconds",
            "trial_duration_seconds",
            "n_input_tokens",
            "n_output_tokens",
            "peak_context_tokens",
        ]
        if c in scored_trials.columns
    ]
    reliability = (
        scored_trials.groupby(["system_id", "system_label"], dropna=False)
        .agg(score=("score", "mean"), n_runs=("trial_id", "count"))
        .reset_index()
    )
    for _rel_col in reliability_cols:
        reliability[f"{_rel_col}_mean"] = (
            scored_trials.groupby("system_id")[_rel_col].mean().reindex(reliability["system_id"]).values
        )
    reliability = reliability.merge(observed[["system_id", "observed_rank"]], on="system_id", how="left").sort_values("observed_rank")
    rel_top = reliability.head(selected_top_n)
    rel_top = rel_top.copy()
    fig_rel, _rel_ax = plt.subplots(figsize=(7.5, 4.8))
    cost_col = "cost_usd_mean"
    _leader_system = rel_top.sort_values("observed_rank").iloc[0]["system_id"] if len(rel_top) else None
    _near_score_cutoff = rel_top["score"].max() - 0.02 if len(rel_top) else float("nan")
    if cost_col in rel_top.columns:
        _sizes = rel_top["n_agent_steps_mean"] if "n_agent_steps_mean" in rel_top.columns else rel_top["n_runs"]
        _sizes = 40 + 160 * (_sizes - _sizes.min()) / max(float(_sizes.max() - _sizes.min()), 1)
        _rel_ax.scatter(rel_top[cost_col], 100 * rel_top["score"], s=_sizes, alpha=0.75, color="#4C78A8")
        _cost_hi = rel_top[cost_col].quantile(0.9) if len(rel_top) else float("nan")
        _label_mask = (
            rel_top["system_id"].eq(_leader_system)
            | rel_top["score"].ge(_near_score_cutoff)
            | rel_top[cost_col].ge(_cost_hi)
        )
        for _, _rel_row in rel_top.iterrows():
            _rel_ax.annotate(
                f"#{int(_rel_row['observed_rank'])}",
                (_rel_row[cost_col], 100 * _rel_row["score"]),
                xytext=(4, -9),
                textcoords="offset points",
                fontsize=7,
                color="#333",
            )
        for _, _rel_row in rel_top[_label_mask].iterrows():
            _rel_ax.annotate(
                _rel_row["system_label"],
                (_rel_row[cost_col], 100 * _rel_row["score"]),
                xytext=(5, 7),
                textcoords="offset points",
                fontsize=7,
                alpha=0.9,
            )
        _rel_ax.set_xlabel("Mean cost per trial (USD)")
    else:
        _rel_ax.scatter(rel_top["n_runs"], 100 * rel_top["score"], alpha=0.75)
        for _, _rel_row in rel_top.iterrows():
            _rel_ax.annotate(
                f"#{int(_rel_row['observed_rank'])}",
                (_rel_row["n_runs"], 100 * _rel_row["score"]),
                xytext=(4, -9),
                textcoords="offset points",
                fontsize=7,
                color="#333",
            )
        _rel_ax.set_xlabel("Runs")
    _rel_ax.set_ylabel("Trial aggregate score (%)")
    _rel_ax.set_title("Score vs operational cost/profile")
    fig_rel.tight_layout()

    cost_range = (
        float(rel_top[cost_col].max() - rel_top[cost_col].min())
        if cost_col in rel_top.columns and len(rel_top)
        else float("nan")
    )
    near_score = rel_top[abs(rel_top["score"] - rel_top["score"].max()) <= 0.02].copy()
    missing_profile_cols = [
        label
        for label, column in [
            ("cost", "cost_usd_mean"),
            ("steps", "n_agent_steps_mean"),
            ("agent duration", "agent_duration_seconds_mean"),
            ("trial duration", "trial_duration_seconds_mean"),
            ("input tokens", "n_input_tokens_mean"),
            ("output tokens", "n_output_tokens_mean"),
            ("peak context tokens", "peak_context_tokens_mean"),
        ]
        if column not in rel_top.columns
    ]
    rel_display = rel_top.copy()
    rel_display["score_pct"] = 100 * rel_display["score"]
    rel_display["plot_note"] = rel_display.apply(
        lambda row: (
            f"rank #{int(row['observed_rank'])}; "
            f"{row['system_label']}; "
            f"score {100 * row['score']:.1f}%; "
            f"cost ${row[cost_col]:.2f}" if cost_col in rel_display.columns and not pd.isna(row[cost_col]) else
            f"rank #{int(row['observed_rank'])}; {row['system_label']}; score {100 * row['score']:.1f}%; cost unavailable"
        ),
        axis=1,
    )
    for _optional_label, _optional_col, _fmt in [
        ("steps", "n_agent_steps_mean", "{:.1f}"),
        ("agent sec", "agent_duration_seconds_mean", "{:.1f}"),
        ("trial sec", "trial_duration_seconds_mean", "{:.1f}"),
        ("input tok", "n_input_tokens_mean", "{:.0f}"),
        ("output tok", "n_output_tokens_mean", "{:.0f}"),
        ("peak ctx", "peak_context_tokens_mean", "{:.0f}"),
    ]:
        if _optional_col in rel_display.columns:
            rel_display["plot_note"] = rel_display["plot_note"] + rel_display[_optional_col].map(
                lambda value, label=_optional_label, fmt=_fmt: "" if pd.isna(value) else f"; {label} {fmt.format(value)}"
            )
    rel_display_cols = [
        column
        for column in [
            "observed_rank",
            "system_label",
            "score_pct",
            "cost_usd_mean",
            "n_agent_steps_mean",
            "agent_duration_seconds_mean",
            "trial_duration_seconds_mean",
            "n_input_tokens_mean",
            "n_output_tokens_mean",
            "peak_context_tokens_mean",
            "plot_note",
        ]
        if column in rel_display.columns
    ]
    _profile_availability = (
        "All configured cost/profile columns are available."
        if not missing_profile_cols
        else "Unavailable profile fields in this data: " + ", ".join(missing_profile_cols) + "."
    )
    mo.vstack(
        [
            section(
                "Do similar scores hide different operational costs?",
                "Can configs with similar scores have meaningfully different operating profiles?",
                "Agentic systems are not only pass/fail objects: cost, latency, steps, and token use can change deployment value even at similar scores.",
                "Aggregate score, cost, steps, duration, and token metrics by config, then compare high-scoring and near-score configs.",
                "Every point is marked with its observed rank. Full labels mark the leader, near-score contenders, and high-cost outliers. Points farther right cost more; point size reflects steps when available.",
                [
                    f"Top-{selected_top_n} mean cost range is ${cost_range:.2f} per trial." if cost_col in rel_top.columns else "Cost is unavailable.",
                    f"{len(near_score)} top configs are within 2 pp of the selected leader in the displayed top-{selected_top_n}.",
                    _profile_availability,
                    "What to take away: rank numbers keep every point identifiable, and the table carries the full operating profile for each displayed config.",
                    "This section treats reliability operationally: cost, latency, steps, and token use.",
                    "Same score can still mean a very different operating profile.",
                ],
                "Cost and latency are descriptive and may depend on harness/runtime conditions; they are not pure model properties.",
            ),
            mo.mpl.interactive(fig_rel),
            mo.ui.table(tidy_table(rel_display[rel_display_cols]), pagination=False),
            mo.accordion({"Near-score configs": mo.ui.table(tidy_table(near_score), pagination=False)}),
        ]
    )
    return fig_rel, reliability


@app.cell
def _(mo, pd, plt, scored_trials, section, sns, tidy_table):
    factor_rows = []
    for factor in ["config", "model", "provider", "reasoning_effort", "harness"]:
        if factor in scored_trials.columns:
            factor_rows.append(
                {
                    "factor": factor,
                    "n_values": scored_trials[factor].nunique(dropna=True),
                    "status": "constant" if scored_trials[factor].nunique(dropna=True) <= 1 else "varies",
                    "role": "config component" if factor in {"model", "provider", "reasoning_effort", "harness"} else "evaluated system",
                }
            )
    factor_rows.append({"factor": "environment", "n_values": None, "status": "not separately encoded", "role": "not identifiable"})
    confounding = pd.DataFrame(factor_rows)

    fig_conf, _conf_ax = plt.subplots(figsize=(7.5, 3.8))
    _confounding_chart = confounding.dropna(subset=["n_values"])
    _conf_ax.bar(_confounding_chart["factor"], _confounding_chart["n_values"], color="#59A14F")
    _conf_ax.set_ylabel("Distinct values")
    _conf_ax.set_title("Config factors that actually vary")
    _conf_ax.tick_params(axis="x", rotation=20)
    fig_conf.tight_layout()

    harness_count = int(scored_trials["harness"].nunique()) if "harness" in scored_trials.columns else 0
    mo.vstack(
        [
            section(
                "What is confounded?",
                "Which differences can DeepSWE attribute, and which are confounded by design?",
                "Agentic evals measure configurations. If a factor does not vary, the data cannot estimate its interaction with other factors.",
                "Count distinct values for config, model, provider, reasoning effort, harness, and environment encoding.",
                "Bars with multiple values can support descriptive slice views. Constant or absent factors cannot support interaction claims.",
                [
                    "DeepSWE config is read as harness + provider + model + reasoning effort.",
                    f"harness count is {harness_count}; model x harness interaction is not identifiable here.",
                    "Provider, model, and reasoning effort are config dimensions, not task dimensions.",
                    "What to take away: config comparisons are valid, but model-only stories need extra caveats.",
                ],
                "Because the harness is constant, DeepSWE cannot answer model x harness questions without a crossed model x harness design.",
            ),
            mo.mpl.interactive(fig_conf),
            mo.ui.table(tidy_table(confounding), pagination=False),
        ]
    )
    return confounding, fig_conf


@app.cell
def _(mo, pd, plt, scored_trials, section, sns, tidy_table):
    artifact_cols = [
        _artifact_col
        for _artifact_col in ["has_trajectory", "has_agent_log", "has_model_patch", "has_verifier_output"]
        if _artifact_col in scored_trials.columns
    ]
    artifact_summary = pd.DataFrame(
        [
            {
                "artifact": _artifact_col,
                "available_pct": 100 * scored_trials[_artifact_col].fillna(False).astype(bool).mean(),
            }
            for _artifact_col in artifact_cols
        ]
    )
    by_outcome_rows = []
    if "outcome" in scored_trials.columns:
        for outcome, _outcome_group in scored_trials.groupby("outcome", observed=False):
            _outcome_row = {"outcome": outcome, "n_trials": len(_outcome_group)}
            for _artifact_col in artifact_cols:
                _outcome_row[_artifact_col] = 100 * _outcome_group[_artifact_col].fillna(False).astype(bool).mean()
            by_outcome_rows.append(_outcome_row)
    trace_by_outcome = pd.DataFrame(by_outcome_rows)

    if not trace_by_outcome.empty:
        artifact_matrix = trace_by_outcome.set_index("outcome")[artifact_cols].T
    else:
        artifact_matrix = artifact_summary.set_index("artifact")[["available_pct"]].rename(columns={"available_pct": "overall"})

    fig_trace, _trace_ax = plt.subplots(figsize=(8.5, max(3.2, 0.45 * len(artifact_matrix))))
    sns.heatmap(
        artifact_matrix,
        annot=True,
        fmt=".0f",
        cmap="Blues",
        vmin=0,
        vmax=100,
        cbar_kws={"label": "Available (%)"},
        ax=_trace_ax,
    )
    _trace_ax.set_xlabel("Outcome")
    _trace_ax.set_ylabel("Artifact")
    _trace_ax.set_title("Artifact availability by outcome")
    fig_trace.tight_layout()

    _weakest = artifact_summary.sort_values("available_pct").iloc[0] if len(artifact_summary) else None
    mo.vstack(
        [
            section(
                "Are traces available for future failure analysis?",
                "Does DeepSWE expose enough artifacts to support future trace-level failure analysis?",
                "Outcome-level audits say whether systems differ; traces are needed to explain how and why failures happen.",
                "Report availability of trajectory, agent log, model patch, and verifier-output flags overall and by outcome.",
                "Cells show artifact coverage by outcome. Uniformly high coverage means later trace parsing is less likely to be biased toward successes or failures.",
                [
                    f"Least available artifact: `{_weakest['artifact']}` at {_weakest['available_pct']:.1f}%." if _weakest is not None else "No artifact flags found.",
                    "What to take away: this matrix checks observability, not whether the traces already explain failures.",
                    "This notebook reports observability only; it does not parse trace contents.",
                    "Trace parsing is a future layer for failure modes and process mining.",
                ],
                "Availability is not explanation. It only says whether later trace analysis is feasible.",
            ),
            mo.mpl.interactive(fig_trace),
            mo.ui.table(tidy_table(artifact_summary), pagination=False),
            mo.accordion({"Artifact availability by outcome": mo.ui.table(tidy_table(trace_by_outcome), pagination=False)}),
        ]
    )
    return artifact_matrix, artifact_summary, fig_trace, trace_by_outcome


@app.cell
def _(
    adjacent,
    all_pairs,
    artifact_summary,
    boundary,
    capability_matrix,
    confounding,
    cost_power,
    coverage_summary,
    first_vs_rest,
    imbalance,
    influence,
    interval_comparison,
    k_simulation,
    mo,
    nonseparation_tiers,
    np,
    observed,
    pd,
    rank_summary,
    reliability,
    selected_unit,
    selected_top_k,
    selected_top_n,
    slice_pair_deltas,
    slice_leaders,
    task_diagnostics,
    trace_by_outcome,
    variance_budget,
    version,
):
    _summary_top_label = observed.iloc[0]["label"]
    _summary_top_score = 100 * observed.iloc[0]["score"]
    _summary_ambiguous = int(adjacent["within_practical_band"].sum()) if not adjacent.empty else 0
    _summary_family_adjacent = int(adjacent["familywise_significant"].sum()) if not adjacent.empty else 0
    _summary_family_all_pairs = int(all_pairs["familywise_significant"].sum()) if not all_pairs.empty else 0
    _summary_tiers = int(nonseparation_tiers["non_separation_tier"].nunique()) if not nonseparation_tiers.empty else 0
    _summary_max_influence = influence.iloc[0] if not influence.empty else None
    _summary_most_concentrated = imbalance.sort_values("largest_share_pct", ascending=False).iloc[0]
    _coverage = coverage_summary.iloc[0]
    _median_rank_width = float((rank_summary["rank_p95"] - rank_summary["rank_p05"]).median())
    _boundary_fragile = (
        boundary[(boundary[f"p_top{selected_top_k}"] > 0.1) & (boundary[f"p_top{selected_top_k}"] < 0.9)]
        if f"p_top{selected_top_k}" in boundary
        else boundary.iloc[0:0]
    )
    _powered_slices = int(slice_leaders[["dimension", "slice"]].drop_duplicates().shape[0]) if not slice_leaders.empty else 0
    _task_share = float(variance_budget["task_share_proxy"].median()) if "task_share_proxy" in variance_budget else float("nan")
    _mde_drop = (
        float(k_simulation["mde80_pp"].iloc[0] - k_simulation["mde80_pp"].iloc[-1])
        if not k_simulation.empty
        else float("nan")
    )
    _best_cost_power = (
        cost_power.dropna(subset=["mde_pp_reduced_per_100_usd"]).sort_values("mde_pp_reduced_per_100_usd", ascending=False).head(1)
        if not cost_power.empty
        else pd.DataFrame()
    )
    _best_cost_power_evidence = (
        f"K={int(_best_cost_power['runs_per_task'].iloc[0])} gives "
        f"{_best_cost_power['mde_pp_reduced_per_100_usd'].iloc[0]:.2f} MDE pp reduced per $100"
        if not _best_cost_power.empty
        else "cost_usd unavailable or no MDE reduction"
    )
    _cost_col = "cost_usd_mean"
    _top_reliability = reliability.sort_values("observed_rank").head(selected_top_n)
    _cost_range = (
        float(_top_reliability[_cost_col].max() - _top_reliability[_cost_col].min())
        if _cost_col in _top_reliability
        else float("nan")
    )
    _harness_status = confounding.loc[confounding["factor"].eq("harness"), "status"]
    _harness_status = _harness_status.iloc[0] if len(_harness_status) else "unknown"
    _artifact_min = artifact_summary.sort_values("available_pct").iloc[0] if len(artifact_summary) else None
    _first_separated = int(first_vs_rest["familywise_significant"].sum()) if not first_vs_rest.empty else 0
    _interval_width_ratio = np.nan
    if not interval_comparison.empty:
        _naive_width = (interval_comparison["naive_hi_pp"] - interval_comparison["naive_lo_pp"]).median()
        _task_width = (interval_comparison["task_boot_hi_pp"] - interval_comparison["task_boot_lo_pp"]).median()
        _interval_width_ratio = float(_task_width / _naive_width) if _naive_width and not np.isnan(_naive_width) else np.nan
    _high_leverage_tasks = int((task_diagnostics["max_abs_rank_shift_top10"] > 0).sum()) if not task_diagnostics.empty else 0
    _slice_delta_sign_flips = (
        bool((slice_pair_deltas["ci_lo_pp"] < 0).any() and (slice_pair_deltas["ci_hi_pp"] > 0).any())
        if not slice_pair_deltas.empty
        else False
    )

    claim_report = pd.DataFrame(
        [
            {
                "claim": "DeepSWE reports an observed config-level leaderboard, not pure model ability.",
                "question answered": "What is this eval measuring?",
                "method": "taxonomy mapping + observed leaderboard",
                "evidence": f"top {selected_unit}: {_summary_top_label} at {_summary_top_score:.1f}%",
                "finding": "observed score is descriptive for this eval universe",
                "caveat": "model-only aggregation is a sensitivity view, not the default estimand",
            },
            {
                "claim": "Exact nearby ranks are over-resolved.",
                "question answered": "Can nearby ranks be distinguished?",
                "method": "adjacent paired deltas, MDE, max-T correction",
                "evidence": f"{_summary_family_adjacent}/{len(adjacent)} adjacent pairs survive max-T; {_summary_ambiguous} are within the practical band",
                "finding": f"top-{selected_top_n} all-pairs has {_summary_family_all_pairs}/{len(all_pairs)} family-wise significant pairs",
                "caveat": "not distinguishable does not prove equivalence",
            },
            {
                "claim": "The leader is not necessarily separated from every close contender.",
                "question answered": "Is rank 1 clearly ahead of the rest?",
                "method": "first-vs-rest paired bootstrap with max-T intervals",
                "evidence": f"{_first_separated}/{len(first_vs_rest)} first-vs-rest comparisons survive family-wise correction",
                "finding": "leader claims should name which comparisons are actually separated",
                "caveat": "first-vs-rest is still a config-level comparison on the observed task universe",
            },
            {
                "claim": "Coarse tiers are safer than exact local ranks.",
                "question answered": "What ranking granularity is defensible?",
                "method": "adjacent non-separation tier construction",
                "evidence": f"{_summary_tiers} tiers from adjacent family-wise separation",
                "finding": "tier claims are more legible than rank-by-rank stories",
                "caveat": "tiering is an audit summary, not an official leaderboard replacement",
            },
            {
                "claim": "Rank placement has task-population uncertainty.",
                "question answered": "Are ranks stable under task resampling?",
                "method": "task bootstrap rank intervals and top-K boundary probabilities",
                "evidence": f"median p05-p95 rank width is {_median_rank_width:.1f}; {len(_boundary_fragile)} boundary configs have partial top-{selected_top_k} membership",
                "finding": "some placements are robust, but boundary membership is fragile",
                "caveat": "bootstrap treats observed tasks as a proxy for a task population",
            },
            {
                "claim": "Aggregate scores hide slice behavior.",
                "question answered": "Where does performance differ by domain/slice?",
                "method": "powered slice leaderboards",
                "evidence": f"{_powered_slices} powered slices available; top-pair slice deltas include CI crossing zero: {_slice_delta_sign_flips}",
                "finding": "slice views show where aggregate ranking may not generalize uniformly",
                "caveat": "underpowered slices are excluded from main comparisons",
            },
            {
                "claim": "Some tasks/domains have high leverage.",
                "question answered": "Which units materially move ranks?",
                "method": "leave-one-task/domain-out rank shifts",
                "evidence": (
                    f"{_summary_max_influence['unit']}: {_summary_max_influence['value']} shifts top-10 ranks by up to "
                    f"{_summary_max_influence['max_abs_rank_shift_top10']:.1f}"
                    if _summary_max_influence is not None
                    else "no influence rows"
                ),
                "finding": f"aggregate order depends more on some units than others; {_high_leverage_tasks} tasks shift at least one top-10 rank",
                "caveat": "high leverage is sensitivity, not a defect label",
            },
            {
                "claim": "More repeated runs may have limited returns if task variance dominates.",
                "question answered": "Would one more run help?",
                "method": "variance budget proxy + K-simulation",
                "evidence": f"median task-share proxy is {100 * _task_share:.1f}%; task/naive interval width ratio is {_interval_width_ratio:.1f}x; K=1 to K=16 reduces MDE by {_mde_drop:.1f} pp",
                "finding": "use the curve to decide whether to buy rollouts or tasks",
                "caveat": "this is approximate; repeated-run design is not fully crossed for every config-task cell",
            },
            {
                "claim": "Repeated-run precision has an approximate dollar cost.",
                "question answered": "How much uncertainty reduction does another rollout buy?",
                "method": "K-simulation joined to mean logged trial cost",
                "evidence": _best_cost_power_evidence,
                "finding": "cost-power helps compare extra rollouts against adding tasks or domains",
                "caveat": "approximate only; not a model x harness estimate because harness is constant",
            },
            {
                "claim": "Coverage and imbalance constrain claims.",
                "question answered": "Is the eval universe balanced and sufficiently covered?",
                "method": "effective counts, config-task coverage, underpowered slices",
                "evidence": f"{_coverage['coverage_pct']:.1f}% config-task coverage; most concentrated dimension is {_summary_most_concentrated['dimension']}",
                "finding": "missing cells and concentrated domains should temper broad claims",
                "caveat": "coverage can be good globally while specific slices remain weak",
            },
            {
                "claim": "Similar scores can hide different operating profiles.",
                "question answered": "Do point scores hide operational quality differences?",
                "method": "score vs cost/steps/duration/token reliability profile",
                "evidence": f"top-{selected_top_n} cost range is ${_cost_range:.2f} per trial" if _cost_col in _top_reliability else "cost unavailable",
                "finding": "operational quality is multi-dimensional, not just pass rate",
                "caveat": "cost/latency are descriptive and may depend on harness/runtime conditions",
            },
            {
                "claim": "Model x harness interaction is not identifiable in DeepSWE.",
                "question answered": "Is this model ability or configuration/harness behavior?",
                "method": "confounding matrix",
                "evidence": f"harness is {_harness_status}",
                "finding": "DeepSWE can inspect config/model/provider/reasoning slices, but not crossed harness effects",
                "caveat": "requires a crossed model x harness design",
            },
            {
                "claim": "Trace-level failure analysis is feasible only if artifacts exist.",
                "question answered": "Can future work explain failures from traces?",
                "method": "artifact availability audit",
                "evidence": (
                    f"least available artifact is {_artifact_min['artifact']} at {_artifact_min['available_pct']:.1f}%"
                    if _artifact_min is not None
                    else "no artifact flags"
                ),
                "finding": "this notebook checks observability, not trace semantics",
                "caveat": "trajectory parsing is future work",
            },
        ]
    )

    roadmap = pd.DataFrame(
        [
            ("trace semantics", "availability only", "trajectory files and event schema"),
            ("judge calibration", "unavailable unless repeated judge labels exist", "judge IDs, repeated labels, expert labels"),
            ("simulator reliability", "unavailable unless simulator IDs/repeats exist", "simulator identity and repeated simulation outcomes"),
            ("tail risk", "unavailable unless severity/risk labels exist", "severity labels or stress probes"),
            ("temporal drift", "roadmap; one live snapshot is not enough", "repeated snapshots or reruns over time"),
            ("model x harness interaction", "unavailable in DeepSWE", "crossed model/harness data"),
            ("production reliability", "roadmap only", "production traces, risk labels, monitoring data"),
        ],
        columns=["question area", "MVP stance", "needed data"],
    )
    yuri_coverage = pd.DataFrame(
        [
            ("paired task bootstrap", "included", "resolution, first-vs-rest, all-pairs"),
            ("adjacent-rank distinguishability", "included", "adjacent paired deltas, MDE, max-T"),
            ("max-T / multiple-comparison correction", "included", "max-T headline; Holm/BH secondary table"),
            ("non-separation tiers", "included", "rank-band summary, not equality claim"),
            ("rank bootstrap stability", "included", "rank intervals, rank heatmap, top-K probabilities"),
            ("naive vs task-clustered uncertainty", "included", "interval comparison section"),
            ("nested repeated-run bootstrap", "partial", "approximate because DeepSWE is not fully crossed"),
            ("variance budget / K-simulation", "included", "task-vs-run proxy and rollout-return curve"),
            ("cost-power / uncertainty per dollar", "partial", "new approximate view using logged cost_usd"),
            ("slice heterogeneity", "included", "powered slices by selected dimension"),
            ("task/domain influence", "included", "leave-one-task/repository-out rank shifts"),
            ("task difficulty/discrimination diagnostics", "included", "top-leverage tasks plus diagnostic table"),
            ("model x harness attribution", "unavailable", "harness is constant; requires crossed model x harness variation"),
            ("trace parsing", "omitted", "artifact availability only in this notebook"),
        ],
        columns=["analysis", "status", "where / caveat"],
    )
    tldr = mo.md(
        f"""
## TL;DR

**What DeepSWE can claim.** It supports config-level leaderboard auditing: observed score, paired resolution, rank stability, slice sensitivity, influence, approximate variance budget, coverage, operational profile, confounding, and artifact readiness.

**What it cannot claim.** It does not identify pure model ability, model x harness interaction, trace-level failure causes, judge calibration, temporal drift, or tail-risk bounds from this data alone.

**Strongest empirical insight.** Exact local ranks are too fine-grained: {_summary_family_adjacent}/{len(adjacent)} adjacent pairs survive max-T, yielding {_summary_tiers} non-separated rank bands. The useful output is a resolution report, not just a point leaderboard.

"""
    )

    mo.vstack(
        [
            mo.md("## Claim Report"),
            mo.md(
                f"DeepSWE `{version.value}` top `{selected_unit}` unit is `{_summary_top_label}` at {_summary_top_score:.1f}%. The table below is the audit artifact: claim, method, evidence, finding, and caveat. The capability matrix above explains why each claim is run, run with caveat, availability-only, or roadmap."
            ),
            mo.ui.table(claim_report, pagination=False),
            tldr,
            mo.md("## Yuri / Resampling Coverage Checklist"),
            mo.ui.table(yuri_coverage, pagination=False),
            mo.md("## Full Taxonomy Roadmap"),
            mo.ui.table(roadmap, pagination=False),
            mo.accordion(
                {
                    "Capability matrix": mo.ui.table(capability_matrix, pagination=False),
                    "Artifact availability by outcome": mo.ui.table(trace_by_outcome, pagination=False),
                }
            ),
        ]
    )
    return claim_report, roadmap, yuri_coverage


if __name__ == "__main__":
    app.run()
