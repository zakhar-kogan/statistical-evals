import marimo

__generated_with = "0.23.11"
app = marimo.App(width="full")


@app.cell
def _():
    from dataclasses import dataclass
    from pathlib import Path
    import json
    import os
    import re

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import requests
    import seaborn as sns

    ROOT = Path(__file__).resolve().parents[1]
    CACHE_ROOT = ROOT / ".cache" / "mle_bench_eval_power"
    REPORT_CACHE = CACHE_ROOT / "grading_reports"
    GITHUB_TREE_API = "https://api.github.com/repos/openai/mle-bench/git/trees/main?recursive=1"
    GITHUB_MEDIA_BASE = "https://media.githubusercontent.com/media/openai/mle-bench/main"
    GITHUB_RAW_BASE = "https://raw.githubusercontent.com/openai/mle-bench/main"
    RNG_SEED = 42
    N_BOOT = 800
    PRACTICAL_EQUIVALENCE_PP = 2.0
    MIN_SLICE_TASKS = 2

    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["figure.dpi"] = 130

    def _number(value):
        if value in (None, "", "None"):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _bool(value):
        if isinstance(value, bool):
            return value
        if value in (None, "", "None"):
            return False
        if isinstance(value, (int, float)):
            return bool(value)
        return str(value).strip().lower() in {"1", "true", "yes", "y"}

    def _path_list(value):
        if not value:
            return []
        paths = []
        for part in str(value).split(os.pathsep):
            if part.strip():
                paths.append(Path(part).expanduser())
        return paths

    def _read_json(path):
        return json.loads(Path(path).read_text())

    def _safe_name(value):
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")

    def _run_group_from_path(path):
        parts = Path(path).parts
        if "runs" in parts:
            index = parts.index("runs")
            if index + 1 < len(parts):
                return parts[index + 1]
        return Path(path).parent.name

    def _system_from_group(run_group):
        text = str(run_group)
        text = re.sub(r"^\d{4}-\d{2}-\d{2}T[^_]+(?:_[A-Z]+)?_run-group_", "", text)
        text = text.replace("run-group_", "")
        return text or str(run_group)

    def _medal_level(row):
        if _bool(row.get("gold_medal")):
            return "gold"
        if _bool(row.get("silver_medal")):
            return "silver"
        if _bool(row.get("bronze_medal")):
            return "bronze"
        return "none"

    def _score_margin(row, threshold_name):
        score = _number(row.get("raw_score"))
        threshold = _number(row.get(threshold_name))
        if score is None or threshold is None:
            return None
        return threshold - score if _bool(row.get("is_lower_better")) else score - threshold

    def _normalize_report(payload, source_path):
        run_group = _run_group_from_path(source_path)
        system_id = _system_from_group(run_group)
        report_id = Path(source_path).stem.replace("_grading_report", "")
        top = payload if isinstance(payload, dict) else {}
        rows = []
        for index, item in enumerate(top.get("competition_reports") or []):
            row = dict(item)
            row["task_id"] = str(row.get("competition_id") or f"unknown-{index}")
            row["task_name"] = row["task_id"]
            row["system_id"] = system_id
            row["run_group"] = run_group
            row["run_id"] = report_id
            row["trial_id"] = f"{run_group}:{report_id}:{row['task_id']}:{index}"
            row["source"] = "mle-bench"
            row["eval_scope"] = "public-grading-reports"
            row["included_in_score"] = True
            row["score"] = float(_bool(row.get("any_medal")))
            row["score_value"] = row["score"]
            row["any_medal"] = row["score"]
            row["above_median"] = float(_bool(row.get("above_median")))
            row["gold_medal"] = float(_bool(row.get("gold_medal")))
            row["silver_medal"] = float(_bool(row.get("silver_medal")))
            row["bronze_medal"] = float(_bool(row.get("bronze_medal")))
            row["valid_submission"] = float(_bool(row.get("valid_submission")))
            row["submission_exists"] = float(_bool(row.get("submission_exists")))
            row["raw_score"] = _number(row.get("score"))
            row["raw_metric_score"] = row["raw_score"]
            row["score"] = row["any_medal"]
            row["score_value"] = row["score"]
            row["medal_level"] = _medal_level(row)
            row["margin_to_bronze"] = _score_margin(row, "bronze_threshold")
            row["margin_to_median"] = _score_margin(row, "median_threshold")
            row["report_file"] = str(source_path)
            rows.append(row)
        return rows

    def _normalize_flat_frame(frame, source_path):
        rows = []
        for index, item in frame.iterrows():
            row = item.to_dict()
            task_id = row.get("task_id") or row.get("competition_id") or f"unknown-{index}"
            system_id = row.get("system_id") or row.get("agent") or row.get("run_group") or "unknown"
            run_id = row.get("run_id") or row.get("trial_id") or Path(source_path).stem
            score = row.get("score")
            if score is None and "any_medal" in row:
                score = row.get("any_medal")
            row.update(
                {
                    "task_id": str(task_id),
                    "task_name": str(task_id),
                    "system_id": str(system_id),
                    "run_id": str(run_id),
                    "trial_id": str(row.get("trial_id") or f"{system_id}:{run_id}:{task_id}:{index}"),
                    "source": row.get("source") or "mle-bench-local",
                    "eval_scope": row.get("eval_scope") or "local",
                    "included_in_score": _bool(row.get("included_in_score", True)),
                    "score": _number(score),
                    "score_value": _number(score),
                    "report_file": str(source_path),
                }
            )
            rows.append(row)
        return rows

    def _load_local_path(path):
        path = Path(path)
        if path.is_dir():
            rows = []
            for child in sorted(path.rglob("*")):
                if child.suffix.lower() in {".json", ".jsonl", ".csv"}:
                    rows.extend(_load_local_path(child))
            return rows
        if path.suffix.lower() == ".csv":
            return _normalize_flat_frame(pd.read_csv(path), path)
        if path.suffix.lower() == ".jsonl":
            return _normalize_flat_frame(pd.read_json(path, lines=True), path)
        if path.suffix.lower() == ".json":
            payload = _read_json(path)
            if isinstance(payload, dict) and "competition_reports" in payload:
                return _normalize_report(payload, path)
            if isinstance(payload, list):
                return _normalize_flat_frame(pd.DataFrame(payload), path)
            if isinstance(payload, dict) and "rows" in payload:
                return _normalize_flat_frame(pd.DataFrame(payload.get("rows") or []), path)
        return []

    def _download_text(url, timeout=90):
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return response.text

    def _public_report_paths(timeout=90):
        response = requests.get(GITHUB_TREE_API, timeout=timeout)
        response.raise_for_status()
        tree = response.json().get("tree") or []
        return sorted(
            item["path"]
            for item in tree
            if item.get("type") == "blob"
            and item.get("path", "").startswith("runs/")
            and item.get("path", "").endswith("_grading_report.json")
        )

    def _load_public_reports():
        cached = sorted(REPORT_CACHE.rglob("*_grading_report.json"))
        if not cached:
            REPORT_CACHE.mkdir(parents=True, exist_ok=True)
            paths = _public_report_paths()
            max_reports = os.environ.get("MLE_BENCH_MAX_REPORTS")
            if max_reports:
                paths = paths[: int(max_reports)]
            for path in paths:
                target = REPORT_CACHE / Path(path).parent.name / Path(path).name
                if target.exists():
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                text = _download_text(f"{GITHUB_MEDIA_BASE}/{path}")
                target.write_text(text)
            cached = sorted(REPORT_CACHE.rglob("*_grading_report.json"))
        rows = []
        for path in cached:
            rows.extend(_normalize_report(_read_json(path), path))
        return rows, {
            "source_kind": "public-github-cache",
            "paths": [str(path) for path in cached],
            "cache_dir": str(REPORT_CACHE),
        }

    def _split_metadata():
        rows = []
        split_dir = CACHE_ROOT / "splits"
        for split_name in ["low", "medium", "high", "split75"]:
            local = split_dir / f"{split_name}.txt"
            try:
                if not local.exists():
                    split_dir.mkdir(parents=True, exist_ok=True)
                    local.write_text(_download_text(f"{GITHUB_RAW_BASE}/experiments/splits/{split_name}.txt"))
                for line in local.read_text().splitlines():
                    task_id = line.strip()
                    if task_id:
                        rows.append({"task_id": task_id, "split": "all" if split_name == "split75" else split_name})
            except Exception:
                continue
        return pd.DataFrame(rows)

    def _category_metadata():
        local = CACHE_ROOT / "competition_categories.csv"
        try:
            if not local.exists():
                local.parent.mkdir(parents=True, exist_ok=True)
                local.write_text(_download_text(f"{GITHUB_MEDIA_BASE}/experiments/competition_categories.csv"))
            categories = pd.read_csv(local)
        except Exception:
            return pd.DataFrame()
        if "competition_id" in categories.columns:
            categories = categories.rename(columns={"competition_id": "task_id"})
        return categories

    def load_mle_bench_results():
        override = os.environ.get("MLE_BENCH_RESULTS")
        status = {}
        try:
            if override:
                rows = []
                for path in _path_list(override):
                    rows.extend(_load_local_path(path))
                status = {"source_kind": "local-override", "paths": [str(p) for p in _path_list(override)]}
            else:
                rows, status = _load_public_reports()
        except Exception as exc:
            rows = []
            status = {"source_kind": "unavailable", "error": f"{type(exc).__name__}: {exc}"}

        data = pd.DataFrame(rows)
        if data.empty:
            return data, pd.DataFrame(), status

        for column in [
            "score",
            "score_value",
            "any_medal",
            "above_median",
            "gold_medal",
            "silver_medal",
            "bronze_medal",
            "valid_submission",
            "submission_exists",
            "raw_score",
            "raw_metric_score",
            "gold_threshold",
            "silver_threshold",
            "bronze_threshold",
            "median_threshold",
            "margin_to_bronze",
            "margin_to_median",
        ]:
            if column in data.columns:
                data[column] = pd.to_numeric(data[column], errors="coerce")

        splits = _split_metadata()
        if not splits.empty:
            split_pivot = splits.pivot_table(index="task_id", values="split", aggfunc=lambda values: ",".join(sorted(set(values))))
            split_pivot = split_pivot.reset_index().rename(columns={"split": "task_splits"})
            data = data.merge(split_pivot, on="task_id", how="left")
            data["difficulty"] = data["task_splits"].fillna("").map(
                lambda value: "low"
                if "low" in value.split(",")
                else ("medium" if "medium" in value.split(",") else ("high" if "high" in value.split(",") else "unknown"))
            )
        else:
            data["task_splits"] = "unknown"
            data["difficulty"] = "unknown"

        categories = _category_metadata()
        if not categories.empty and "task_id" in categories.columns:
            metadata_cols = [c for c in categories.columns if c == "task_id" or c not in data.columns]
            data = data.merge(categories[metadata_cols], on="task_id", how="left")

        data["source"] = data.get("source", "mle-bench")
        data["eval_scope"] = data.get("eval_scope", "public-grading-reports")
        data["included_in_score"] = data.get("included_in_score", True).fillna(True).astype(bool)

        metadata = (
            data[["task_id", "task_name", "difficulty"] + [c for c in ["category", "task_splits"] if c in data.columns]]
            .drop_duplicates("task_id")
            .reset_index(drop=True)
        )
        return data, metadata, status

    @dataclass(frozen=True)
    class EvalSpec:
        name: str
        task_col: str
        system_col: str
        run_col: str
        score_col: str
        dimension_cols: tuple[str, ...] = ()
        reliability_cols: tuple[str, ...] = ()

        @property
        def required_cols(self):
            return (self.task_col, self.system_col, self.run_col, self.score_col)

    MLE_SPEC = EvalSpec(
        name="MLE-bench",
        task_col="task_id",
        system_col="system_id",
        run_col="run_id",
        score_col="score",
        dimension_cols=("difficulty", "category", "is_lower_better", "medal_level"),
        reliability_cols=("valid_submission", "submission_exists", "above_median", "margin_to_bronze", "margin_to_median"),
    )

    def _require_columns(data, spec):
        missing = [column for column in spec.required_cols if column not in data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def filtered_rows(data, spec):
        if data.empty:
            return data.copy()
        _require_columns(data, spec)
        out = data[data["included_in_score"].fillna(False).astype(bool)].copy()
        out = out[out[spec.score_col].notna()].copy()
        return out

    def aggregate_task_cells(data, spec):
        if data.empty:
            return pd.DataFrame(columns=[spec.task_col, spec.system_col, spec.score_col])
        grouped = (
            data.groupby([spec.task_col, spec.system_col, "source", "eval_scope"], dropna=False)
            .agg(
                score=(spec.score_col, "mean"),
                n_trials=(spec.score_col, "size"),
                above_median=("above_median", "mean"),
                valid_submission=("valid_submission", "mean"),
            )
            .reset_index()
        )
        return grouped

    def leaderboard_from_cells(cells, spec, metric="score"):
        if cells.empty:
            return pd.DataFrame(columns=[spec.system_col, "mean_score", "tasks", "rank"])
        board = (
            cells.groupby(spec.system_col)
            .agg(mean_score=(metric, "mean"), tasks=(spec.task_col, "nunique"), n_cells=(metric, "size"))
            .reset_index()
            .sort_values(["mean_score", "tasks", spec.system_col], ascending=[False, False, True])
        )
        board["rank"] = board["mean_score"].rank(method="min", ascending=False).astype(int)
        return board

    def task_bootstrap(cells, spec, *, draws=N_BOOT, seed=RNG_SEED, metric="score"):
        if cells.empty:
            return pd.DataFrame()
        rng = np.random.default_rng(seed)
        tasks = np.array(sorted(cells[spec.task_col].dropna().unique()))
        systems = sorted(cells[spec.system_col].dropna().unique())
        draws_out = []
        for draw in range(draws):
            sampled = rng.choice(tasks, size=len(tasks), replace=True)
            sample = pd.DataFrame({spec.task_col: sampled})
            boot = sample.merge(cells, on=spec.task_col, how="left")
            means = boot.groupby(spec.system_col)[metric].mean()
            ranks = means.rank(method="min", ascending=False)
            for system in systems:
                draws_out.append(
                    {
                        "draw": draw,
                        spec.system_col: system,
                        "mean_score": means.get(system, np.nan),
                        "rank": ranks.get(system, np.nan),
                    }
                )
        return pd.DataFrame(draws_out)

    def bootstrap_summary(boot, spec):
        if boot.empty:
            return pd.DataFrame()
        return (
            boot.groupby(spec.system_col)
            .agg(
                boot_mean=("mean_score", "mean"),
                score_p05=("mean_score", lambda values: np.nanquantile(values, 0.05)),
                score_p95=("mean_score", lambda values: np.nanquantile(values, 0.95)),
                rank_p05=("rank", lambda values: np.nanquantile(values, 0.05)),
                rank_p50=("rank", lambda values: np.nanquantile(values, 0.50)),
                rank_p95=("rank", lambda values: np.nanquantile(values, 0.95)),
            )
            .reset_index()
            .sort_values(["boot_mean", spec.system_col], ascending=[False, True])
        )

    def adjacent_resolution(board, cells, spec):
        if board.empty or len(board) < 2:
            return pd.DataFrame()
        rows = []
        ordered = board.sort_values("rank")
        for left, right in zip(ordered.iloc[:-1].to_dict("records"), ordered.iloc[1:].to_dict("records")):
            a = left[spec.system_col]
            b = right[spec.system_col]
            pivot = cells[cells[spec.system_col].isin([a, b])].pivot_table(
                index=spec.task_col, columns=spec.system_col, values="score", aggfunc="mean"
            )
            paired = pivot.dropna()
            diff = paired[a] - paired[b] if not paired.empty else pd.Series(dtype=float)
            rows.append(
                {
                    "higher_ranked": a,
                    "lower_ranked": b,
                    "rank_gap": int(right["rank"] - left["rank"]),
                    "score_gap_pp": 100 * (left["mean_score"] - right["mean_score"]),
                    "paired_tasks": len(paired),
                    "paired_mean_gap_pp": 100 * diff.mean() if len(diff) else np.nan,
                    "higher_wins": int((diff > 0).sum()) if len(diff) else 0,
                    "ties": int((diff == 0).sum()) if len(diff) else 0,
                    "lower_wins": int((diff < 0).sum()) if len(diff) else 0,
                }
            )
        return pd.DataFrame(rows)

    def slice_table(data, spec, dimension):
        if data.empty or dimension not in data.columns:
            return pd.DataFrame()
        rows = []
        for value, group in data.dropna(subset=[dimension]).groupby(dimension):
            cells = aggregate_task_cells(group, spec)
            board = leaderboard_from_cells(cells, spec)
            if cells[spec.task_col].nunique() >= MIN_SLICE_TASKS:
                for item in board.to_dict("records"):
                    rows.append(
                        {
                            "slice_dimension": dimension,
                            "slice": value,
                            spec.system_col: item[spec.system_col],
                            "mean_score": item["mean_score"],
                            "tasks": item["tasks"],
                            "rank": item["rank"],
                        }
                    )
        return pd.DataFrame(rows)

    def pass_at_k(data, spec, max_k=8):
        if data.empty:
            return pd.DataFrame()
        rows = []
        for system, group in data.groupby(spec.system_col):
            task_trials = group.groupby(spec.task_col)["score"].apply(list)
            for k in range(1, max_k + 1):
                values = []
                for scores in task_trials:
                    arr = np.array(scores, dtype=float)
                    if len(arr) == 0:
                        continue
                    values.append(float(arr[:k].max() >= 1.0))
                rows.append({"system_id": system, "k": k, "pass_at_k": np.mean(values) if values else np.nan})
        return pd.DataFrame(rows)

    def task_influence(cells, spec):
        if cells.empty:
            return pd.DataFrame()
        full = leaderboard_from_cells(cells, spec).set_index(spec.system_col)["mean_score"]
        rows = []
        for task in sorted(cells[spec.task_col].unique()):
            reduced = cells[cells[spec.task_col] != task]
            board = leaderboard_from_cells(reduced, spec).set_index(spec.system_col)["mean_score"]
            common = full.index.intersection(board.index)
            rows.append(
                {
                    "task_id": task,
                    "max_abs_shift_pp": 100 * (full.loc[common] - board.loc[common]).abs().max() if len(common) else np.nan,
                    "mean_abs_shift_pp": 100 * (full.loc[common] - board.loc[common]).abs().mean() if len(common) else np.nan,
                }
            )
        return pd.DataFrame(rows).sort_values("max_abs_shift_pp", ascending=False)

    return (
        MLE_SPEC,
        adjacent_resolution,
        aggregate_task_cells,
        bootstrap_summary,
        filtered_rows,
        leaderboard_from_cells,
        load_mle_bench_results,
        mo,
        pass_at_k,
        pd,
        plt,
        slice_table,
        task_bootstrap,
        task_influence,
    )


@app.cell
def _(mo):
    mo.md("""
    # MLE-bench Eval Power / Rank Resolution Audit

    MLE-bench evaluates machine-learning engineering agents on Kaggle-style competitions. This notebook treats each competition as the resampling task unit and ranks systems by **Any Medal** by default.

    The important statistical wrinkle is that grading reports can contain repeated submissions/runs for the same competition and run group. The notebook mean-aggregates repeated `(competition, system, source, eval_scope)` cells before task bootstrap, matching the rank-stability convention used elsewhere in this repo.

    Public data source: the `openai/mle-bench` GitHub repo's public `runs/*/*_grading_report.json` files. Set `MLE_BENCH_RESULTS` to a file, directory, or path-list to override with local exports.
    """)
    return


@app.cell
def _(mo):
    ranked_metric = mo.ui.dropdown(
        options=["score", "above_median", "valid_submission", "submission_exists"],
        value="score",
        label="Rank metric",
    )
    slice_dimension = mo.ui.dropdown(
        options=["difficulty", "category", "is_lower_better", "medal_level"],
        value="difficulty",
        label="Slice dimension",
    )
    top_n = mo.ui.slider(5, 30, value=15, step=1, show_value=True, label="Top N")
    top_k = mo.ui.slider(2, 10, value=6, step=1, show_value=True, label="Pass@k max")
    mo.vstack(
        [
            mo.md("## Setup And Data / Eval Universe"),
            mo.hstack([ranked_metric, slice_dimension, top_n, top_k], justify="start", gap=1),
        ]
    )
    return ranked_metric, slice_dimension, top_k, top_n


@app.cell
def _(load_mle_bench_results):
    raw_rows, task_metadata, load_status = load_mle_bench_results()
    return load_status, raw_rows, task_metadata


@app.cell
def _(load_status, mo, pd, raw_rows, task_metadata):
    if raw_rows.empty:
        status = load_status.get("error", "No MLE-bench rows were found.")
        _display = mo.callout(
            mo.md(
                f"""
    No MLE-bench data loaded.

    Loader status: `{load_status.get("source_kind", "unknown")}`  
    Detail: `{status}`

    To run with local data, set `MLE_BENCH_RESULTS` to a grading report JSON, a directory of reports, a flat CSV/JSON/JSONL, or a path-list.
    """
            ),
            kind="warn",
        )
    else:
        summary = pd.DataFrame(
            [
                {"field": "rows", "value": len(raw_rows)},
                {"field": "tasks", "value": raw_rows["task_id"].nunique()},
                {"field": "systems", "value": raw_rows["system_id"].nunique()},
                {"field": "run_groups", "value": raw_rows["run_group"].nunique() if "run_group" in raw_rows else "n/a"},
                {"field": "metadata_rows", "value": len(task_metadata)},
                {"field": "source_kind", "value": load_status.get("source_kind")},
                {"field": "paths", "value": len(load_status.get("paths", []))},
            ]
        )
        _display = mo.vstack(
            [
                mo.callout(
                    mo.md(
                        f"Loaded MLE-bench results from `{load_status.get('source_kind')}`: `{len(load_status.get('paths', []))}` file/path(s)."
                    ),
                    kind="success",
                ),
                mo.ui.table(summary, selection=None),
            ]
        )
    _display
    return


@app.cell
def _(MLE_SPEC, filtered_rows, raw_rows):
    rows = filtered_rows(raw_rows, MLE_SPEC)
    return (rows,)


@app.cell
def _(mo, pd, rows):
    if rows.empty:
        _display = mo.md("## Taxonomy And Capability Matrix\n\nNo rows loaded.")
    else:
        fields = []
        for column in [
            "task_id",
            "system_id",
            "run_id",
            "score",
            "any_medal",
            "above_median",
            "valid_submission",
            "difficulty",
            "category",
            "is_lower_better",
            "raw_score",
            "gold_threshold",
            "silver_threshold",
            "bronze_threshold",
            "median_threshold",
            "margin_to_bronze",
            "margin_to_median",
            "submission_path",
            "created_at",
        ]:
            fields.append(
                {
                    "column": column,
                    "present": column in rows.columns,
                    "non_null": int(rows[column].notna().sum()) if column in rows.columns else 0,
                    "unique": int(rows[column].nunique(dropna=True)) if column in rows.columns else 0,
                }
            )
        _display = mo.vstack(
            [
                mo.md(
                    """
    ## Taxonomy And Capability Matrix

    MLE-bench adds dimensions DeepSWE does not have: ML competition difficulty, Kaggle metric direction, medal thresholds, valid-submission failures, raw metric score, and margin-to-medal diagnostics.
    """
                ),
                mo.ui.table(pd.DataFrame(fields), selection=None),
            ]
        )
    _display
    return


@app.cell
def _(
    MLE_SPEC,
    aggregate_task_cells,
    leaderboard_from_cells,
    ranked_metric,
    rows,
):
    cells = aggregate_task_cells(rows, MLE_SPEC)
    metric_col = ranked_metric.value
    if metric_col != "score" and metric_col in rows.columns:
        tmp = rows.copy()
        tmp["score"] = tmp[metric_col]
        cells_for_rank = aggregate_task_cells(tmp, MLE_SPEC)
    else:
        cells_for_rank = cells
    leaderboard = leaderboard_from_cells(cells_for_rank, MLE_SPEC)
    return cells_for_rank, leaderboard, metric_col


@app.cell
def _(leaderboard, metric_col, mo, plt, top_n):
    if leaderboard.empty:
        _display = mo.md("## Observed Leaderboard\n\nNo leaderboard rows.")
    else:
        shown = leaderboard.head(top_n.value).copy()
        fig, ax = plt.subplots(figsize=(9, max(3, 0.35 * len(shown))))
        ax.barh(shown["system_id"][::-1], 100 * shown["mean_score"][::-1])
        ax.set_xlabel(f"Mean {metric_col} (%)")
        ax.set_title("Observed MLE-bench leaderboard")
        _display = mo.vstack(
            [
                mo.md(
                    f"""
    ## Observed Leaderboard

    Rows are competition-level grading outcomes. Repeated runs for the same `(competition, system)` are mean-aggregated before ranking.
    """
                ),
                fig,
                mo.ui.table(
                    shown.assign(mean_score=lambda frame: (100 * frame["mean_score"]).round(2)),
                    selection=None,
                ),
            ]
        )
    _display
    return


@app.cell
def _(
    MLE_SPEC,
    bootstrap_summary,
    cells_for_rank,
    leaderboard,
    mo,
    task_bootstrap,
):
    boot = task_bootstrap(cells_for_rank, MLE_SPEC)
    boot_summary = bootstrap_summary(boot, MLE_SPEC)
    if boot_summary.empty:
        _display = mo.md("## Task Bootstrap Rank Stability\n\nNo bootstrap output.")
    else:
        table = leaderboard.merge(boot_summary, on="system_id", how="left")
        _display = mo.vstack(
            [
                mo.md(
                    """
    ## Task Bootstrap Rank Stability

    Bootstrap draws resample competitions with replacement. Intervals describe how much the rank would move if this competition set were treated as a sample from a broader MLE task population.
    """
                ),
                mo.ui.table(
                    table.assign(
                        mean_score=lambda frame: (100 * frame["mean_score"]).round(2),
                        score_p05=lambda frame: (100 * frame["score_p05"]).round(2),
                        score_p95=lambda frame: (100 * frame["score_p95"]).round(2),
                        rank_p05=lambda frame: frame["rank_p05"].round(1),
                        rank_p50=lambda frame: frame["rank_p50"].round(1),
                        rank_p95=lambda frame: frame["rank_p95"].round(1),
                    ),
                    selection=None,
                ),
            ]
        )
    _display
    return (boot_summary,)


@app.cell
def _(MLE_SPEC, adjacent_resolution, cells_for_rank, leaderboard, mo):
    adjacent = adjacent_resolution(leaderboard, cells_for_rank, MLE_SPEC)
    if adjacent.empty:
        _display = mo.md("## Paired Adjacent-Rank Checks\n\nNo adjacent pairs.")
    else:
        _display = mo.vstack(
            [
                mo.md(
                    """
    ## Paired Adjacent-Rank Checks

    Adjacent systems are compared only on competitions where both have a task cell. This surfaces thin pairwise evidence even when overall averages look separated.
    """
                ),
                mo.ui.table(adjacent.round(2), selection=None),
            ]
        )
    _display
    return


@app.cell
def _(MLE_SPEC, mo, rows, slice_dimension, slice_table):
    slices = slice_table(rows, MLE_SPEC, slice_dimension.value)
    if slices.empty:
        _display = mo.md("## Domain And Reward-Basis Slices\n\nNo slice output for the selected dimension.")
    else:
        _display = mo.vstack(
            [
                mo.md(
                    f"""
    ## Slice Sensitivity

    Selected dimension: `{slice_dimension.value}`. MLE-bench slices are diagnostics, not separate official leaderboards unless the split is one of the benchmark's reported Low/Medium/High/All splits.
    """
                ),
                mo.ui.table(
                    slices.assign(mean_score=lambda frame: (100 * frame["mean_score"]).round(2)),
                    selection=None,
                ),
            ]
        )
    _display
    return


@app.cell
def _(MLE_SPEC, mo, pass_at_k, rows, top_k):
    passk = pass_at_k(rows, MLE_SPEC, max_k=top_k.value)
    if passk.empty:
        _display = mo.md("## Repeated-Trial Variance And Pass@k Diagnostics\n\nNo repeated-trial output.")
    else:
        _display = mo.vstack(
            [
                mo.md(
                    """
    ## Repeated-Trial Variance And Pass@k Diagnostics

    Pass@k here is empirical over observed repeated reports: for each competition, did any of the first `k` observed trials earn any medal? It is a diagnostic, not necessarily the official leaderboard metric.
    """
                ),
                mo.ui.table(passk.assign(pass_at_k=lambda frame: (100 * frame["pass_at_k"]).round(2)), selection=None),
            ]
        )
    _display
    return


@app.cell
def _(MLE_SPEC, cells_for_rank, mo, task_influence):
    influence = task_influence(cells_for_rank, MLE_SPEC)
    if influence.empty:
        _display = mo.md("## Task Influence\n\nNo task influence output.")
    else:
        _display = mo.vstack(
            [
                mo.md(
                    """
    ## Task Influence

    Leave-one-competition-out influence measures how much each competition can move system means. Large shifts indicate that the leaderboard is sensitive to particular competitions or missing model-task coverage.
    """
                ),
                mo.ui.table(influence.head(25).round(2), selection=None),
            ]
        )
    _display
    return


@app.cell
def _(mo, pd, rows):
    if rows.empty:
        _display = mo.md("## Operational Profile\n\nNo rows loaded.")
    else:
        cols = [c for c in ["system_id", "valid_submission", "submission_exists", "above_median", "margin_to_bronze", "margin_to_median"] if c in rows.columns]
        profile = rows[cols].groupby("system_id").mean(numeric_only=True).reset_index() if cols else pd.DataFrame()
        _display = mo.vstack(
            [
                mo.md(
                    """
    ## Operational Profile

    MLE-bench exposes operational failure modes directly through missing, invalid, and below-threshold submissions. These are often as important as medal outcomes for evaluating an ML-engineering agent.
    """
                ),
                mo.ui.table(profile.round(3), selection=None),
            ]
        )
    _display
    return


@app.cell
def _(boot_summary, leaderboard, load_status, mo, rows):
    if rows.empty:
        _display = mo.md("## Audit Summary\n\nNo data loaded, so no statistical claims are supported.")
    else:
        leader = leaderboard.iloc[0]["system_id"] if not leaderboard.empty else "n/a"
        source_kind = load_status.get("source_kind", "unknown")
        _display = mo.md(
            f"""
    ## Audit Summary

    - Data source: `{source_kind}`.
    - Rows: `{len(rows)}`; competitions: `{rows["task_id"].nunique()}`; systems: `{rows["system_id"].nunique()}`.
    - Default estimand: mean Any Medal rate after repeated `(competition, system, source, eval_scope)` aggregation.
    - Observed leader: `{leader}`.
    - Bootstrap systems summarized: `{0 if boot_summary.empty else len(boot_summary)}`.

    Interpretation caveat: MLE-bench reports public grading outcomes for heterogeneous Kaggle competitions. Rank intervals quantify competition-sampling sensitivity, not uncertainty in Kaggle scoring itself.
    """
        )
    _display
    return


if __name__ == "__main__":
    app.run()
