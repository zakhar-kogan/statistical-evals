import marimo

__generated_with = "0.23.11"
app = marimo.App(width="full")


@app.cell
def _():
    from dataclasses import dataclass
    from pathlib import Path
    import json
    import math
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
        text = re.sub(r"_group_?\d+$", "", text)
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

        for _column in [
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
            if _column in data.columns:
                data[_column] = pd.to_numeric(data[_column], errors="coerce")

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

    def rank_stability_summary(boot, board, spec, top_k):
        if boot.empty or board.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        summary = bootstrap_summary(boot, spec)
        prob_col = f"p_top{top_k}"
        top_prob = (
            boot.assign(in_top_k=lambda frame: frame["rank"] <= top_k)
            .groupby(spec.system_col)["in_top_k"]
            .mean()
            .reset_index(name=prob_col)
        )
        summary = summary.merge(top_prob, on=spec.system_col, how="left")
        rank_counts = (
            boot.dropna(subset=["rank"])
            .assign(rank=lambda frame: frame["rank"].astype(int), present=1)
            .pivot_table(index=spec.system_col, columns="rank", values="present", aggfunc="sum", fill_value=0)
        )
        rank_probs = rank_counts.div(rank_counts.sum(axis=1), axis=0).fillna(0)
        observed_top = set(board.sort_values("rank").head(top_k)[spec.system_col])
        overlap_rows = []
        for draw, group in boot.dropna(subset=["rank"]).groupby("draw"):
            boot_top = set(group.sort_values("rank").head(top_k)[spec.system_col])
            overlap_rows.append({"draw": draw, "top_k_overlap": len(observed_top & boot_top)})
        overlap = pd.DataFrame(overlap_rows)
        return summary, rank_probs, overlap

    def _normal_cdf(value):
        if pd.isna(value):
            return np.nan
        return 0.5 * (1.0 + math.erf(float(value) / math.sqrt(2.0)))

    def _multiplicity_flags(frame):
        if frame.empty or "p_value" not in frame:
            return frame
        out = frame.copy()
        valid = out["p_value"].notna()
        ordered = out.loc[valid].sort_values("p_value")
        m = len(ordered)
        holm = pd.Series(False, index=ordered.index)
        for i, idx in enumerate(ordered.index, start=1):
            threshold = 0.05 / (m - i + 1)
            if ordered.loc[idx, "p_value"] <= threshold:
                holm.loc[idx] = True
            else:
                break
        bh = pd.Series(False, index=ordered.index)
        if m:
            below = ordered["p_value"].to_numpy() <= (np.arange(1, m + 1) / m) * 0.05
            if below.any():
                cutoff = np.where(below)[0].max()
                bh.loc[ordered.index[: cutoff + 1]] = True
        out["holm_significant"] = False
        out["bh_significant"] = False
        out.loc[holm.index, "holm_significant"] = holm
        out.loc[bh.index, "bh_significant"] = bh
        return out

    def all_pairs_rank_resolution(board, cells, spec, top_n=15, draws=N_BOOT, seed=RNG_SEED):
        if board.empty or cells.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), float("nan")
        ordered = board.sort_values(["rank", spec.system_col]).head(top_n).reset_index(drop=True)
        systems = ordered[spec.system_col].tolist()
        ranks = dict(zip(ordered[spec.system_col], ordered["rank"]))
        pivot = cells[cells[spec.system_col].isin(systems)].pivot_table(
            index=spec.task_col, columns=spec.system_col, values="score", aggfunc="mean"
        )
        rng = np.random.default_rng(seed + 17)
        rows = []
        boot_columns = {}
        t_columns = {}
        for i, a in enumerate(systems):
            for b in systems[i + 1 :]:
                paired = pivot[[a, b]].dropna()
                diff = (paired[a] - paired[b]).to_numpy(dtype=float)
                n = len(diff)
                pair_key = f"{a}__vs__{b}"
                if n:
                    obs = float(np.mean(diff))
                    if n > 1:
                        se = float(np.std(diff, ddof=1) / np.sqrt(n))
                    else:
                        se = float("nan")
                    boot_gap = np.full(draws, obs)
                    if n > 1:
                        for draw in range(draws):
                            boot_gap[draw] = float(np.mean(rng.choice(diff, size=n, replace=True)))
                    boot_columns[pair_key] = 100 * boot_gap
                    if se and not np.isnan(se) and se > 0:
                        t_columns[pair_key] = np.abs((boot_gap - obs) / se)
                    z = abs(obs / se) if se and not np.isnan(se) and se > 0 else np.nan
                    p_value = 2 * (1 - _normal_cdf(z)) if not np.isnan(z) else np.nan
                    point_lo = float(np.quantile(100 * boot_gap, 0.025))
                    point_hi = float(np.quantile(100 * boot_gap, 0.975))
                    mde80_pp = 100 * (1.96 + 0.84) * se if se and not np.isnan(se) else np.nan
                else:
                    obs = se = p_value = point_lo = point_hi = mde80_pp = np.nan
                rows.append(
                    {
                        "pair_key": pair_key,
                        "system_a": a,
                        "system_b": b,
                        "rank_a": int(ranks[a]),
                        "rank_b": int(ranks[b]),
                        "paired_tasks": n,
                        "gap_pp": 100 * obs if not np.isnan(obs) else np.nan,
                        "paired_se_pp": 100 * se if not np.isnan(se) else np.nan,
                        "point_lo_pp": point_lo,
                        "point_hi_pp": point_hi,
                        "mde80_pp": mde80_pp,
                        "p_value": p_value,
                        "a_wins": int((diff > 0).sum()) if n else 0,
                        "ties": int((diff == 0).sum()) if n else 0,
                        "a_losses": int((diff < 0).sum()) if n else 0,
                    }
                )
        pair_table = _multiplicity_flags(pd.DataFrame(rows))
        boot_pairs = pd.DataFrame(boot_columns)
        if t_columns:
            max_t = pd.DataFrame(t_columns).max(axis=1)
            max_t_critical = float(max_t.quantile(0.95))
        else:
            max_t_critical = float("nan")
        if not pair_table.empty:
            pair_table["max_t_critical"] = max_t_critical
            pair_table["max_t_lo_pp"] = pair_table["gap_pp"] - max_t_critical * pair_table["paired_se_pp"]
            pair_table["max_t_hi_pp"] = pair_table["gap_pp"] + max_t_critical * pair_table["paired_se_pp"]
            pair_table["pointwise_significant"] = (pair_table["point_lo_pp"] > 0) | (pair_table["point_hi_pp"] < 0)
            pair_table["familywise_significant"] = (pair_table["max_t_lo_pp"] > 0) | (pair_table["max_t_hi_pp"] < 0)
            pair_table["within_practical_band"] = pair_table["gap_pp"].abs() <= PRACTICAL_EQUIVALENCE_PP
            pair_table["practically_equivalent"] = (pair_table["max_t_lo_pp"] >= -PRACTICAL_EQUIVALENCE_PP) & (
                pair_table["max_t_hi_pp"] <= PRACTICAL_EQUIVALENCE_PP
            )
        adjacent = pair_table[pair_table["rank_b"] == pair_table["rank_a"] + 1].copy()
        first_vs_rest = pair_table[pair_table["rank_a"] == 1].copy()
        return pair_table, adjacent, first_vs_rest, max_t_critical

    def nonseparation_bands(adjacent):
        if adjacent.empty:
            return pd.DataFrame()
        rows = []
        band = 1
        ordered = adjacent.sort_values("rank_a")
        for item in ordered.to_dict("records"):
            if not rows:
                rows.append({"rank": item["rank_a"], "system_id": item["system_a"], "non_separation_band": band})
            if item.get("familywise_significant"):
                band += 1
            rows.append({"rank": item["rank_b"], "system_id": item["system_b"], "non_separation_band": band})
        return pd.DataFrame(rows).drop_duplicates(["rank", "system_id"]).sort_values("rank")

    def mde_power_table(pair_table):
        if pair_table.empty:
            return pd.DataFrame(), pd.DataFrame()
        adjacent = pair_table[pair_table["rank_b"] == pair_table["rank_a"] + 1]
        summary = pd.DataFrame(
            [
                {
                    "scope": "all_pairs_top_n",
                    "pairs": len(pair_table),
                    "median_paired_tasks": pair_table["paired_tasks"].median(),
                    "median_mde80_pp": pair_table["mde80_pp"].median(),
                    "min_mde80_pp": pair_table["mde80_pp"].min(),
                    "max_mde80_pp": pair_table["mde80_pp"].max(),
                },
                {
                    "scope": "adjacent_pairs",
                    "pairs": len(adjacent),
                    "median_paired_tasks": adjacent["paired_tasks"].median() if len(adjacent) else np.nan,
                    "median_mde80_pp": adjacent["mde80_pp"].median() if len(adjacent) else np.nan,
                    "min_mde80_pp": adjacent["mde80_pp"].min() if len(adjacent) else np.nan,
                    "max_mde80_pp": adjacent["mde80_pp"].max() if len(adjacent) else np.nan,
                },
            ]
        )
        median_mde = float(pair_table["mde80_pp"].median())
        median_n = float(pair_table["paired_tasks"].median())
        required = []
        for target in [1, 2, 5, 10]:
            required.append(
                {
                    "target_gap_pp": target,
                    "current_median_paired_tasks": median_n,
                    "approx_required_tasks": math.ceil(median_n * (median_mde / target) ** 2) if target and not np.isnan(median_mde) else np.nan,
                }
            )
        return summary, pd.DataFrame(required)

    def run_group_bootstrap_sensitivity(data, spec, draws=N_BOOT, seed=RNG_SEED):
        if data.empty or "run_group" not in data.columns:
            return pd.DataFrame(), pd.DataFrame()
        group_cells = (
            data.groupby([spec.system_col, "run_group", spec.task_col], dropna=False)["score"]
            .mean()
            .reset_index()
        )
        run_groups = group_cells.groupby(spec.system_col)["run_group"].unique().to_dict()
        if not any(len(values) > 1 for values in run_groups.values()):
            return pd.DataFrame(), pd.DataFrame()
        rng = np.random.default_rng(seed + 33)
        systems = sorted(group_cells[spec.system_col].unique())
        rows = []
        for draw in range(draws):
            for system in systems:
                subset = group_cells[group_cells[spec.system_col] == system]
                groups = np.array(sorted(subset["run_group"].unique()))
                if len(groups) == 0:
                    continue
                sampled_groups = rng.choice(groups, size=len(groups), replace=True)
                sampled = pd.concat([subset[subset["run_group"] == group] for group in sampled_groups], ignore_index=True)
                task_means = sampled.groupby(spec.task_col)["score"].mean()
                rows.append({"draw": draw, spec.system_col: system, "mean_score": task_means.mean()})
        boot = pd.DataFrame(rows)
        if boot.empty:
            return boot, pd.DataFrame()
        summary = (
            boot.groupby(spec.system_col)
            .agg(
                run_group_boot_mean=("mean_score", "mean"),
                run_group_boot_p05=("mean_score", lambda values: np.nanquantile(values, 0.05)),
                run_group_boot_p95=("mean_score", lambda values: np.nanquantile(values, 0.95)),
            )
            .reset_index()
        )
        return boot, summary

    def method_routing_table(data, cells, coverage, run_group_map, spec):
        n_tasks = int(data[spec.task_col].nunique()) if not data.empty else 0
        n_systems = int(data[spec.system_col].nunique()) if not data.empty else 0
        has = lambda column: column in data.columns and data[column].notna().any()
        repeated = bool(run_group_map is not None and not run_group_map.empty and (run_group_map["n_run_groups"] > 1).any())
        incomplete = int(coverage["incomplete"].sum()) if coverage is not None and not coverage.empty and "incomplete" in coverage else 0
        dims = [column for column in spec.dimension_cols if has(column)]
        reliability = [column for column in spec.reliability_cols if has(column)]
        rows = [
            ("Observed leaderboard", "task_id, system_id, Any Medal", "competition task cell", "task-cell mean", "included", f"{n_tasks} competitions x {n_systems} systems", "Ranks systems by mean Any Medal", "public runs may have uneven resources/dates"),
            ("Adjacent ranks", "paired task cells", "shared competitions", "paired deltas, bootstrap intervals, max-T", "included", f"{incomplete} systems incomplete", "Tests whether neighboring ranks separate", "thin shared coverage weakens claims"),
            ("All-pairs top-N", "paired task cells among selected systems", "shared competitions", "all-pairs paired deltas, Holm/BH, max-T", "included", f"top-N from {n_systems} systems", "Audits leaderboard over-ordering", "family is selected by displayed top-N"),
            ("Rank stability", "task ids and system scores", "competition", "task bootstrap rank intervals and heatmap", "included", f"{n_tasks} resampling units", "Shows rank movement under task composition", "competitions are heterogeneous, not literal iid"),
            ("Top-K boundary", "bootstrap ranks", "competition", "P(top-K), observed overlap", "included", f"top-K over {n_systems} systems", "Audits cutoff fragility", "top-K is a reporting choice"),
            ("Repeated runs", "run_group and repeated reports", "run group / competition", "run-group variance, run-group bootstrap, pass@k diagnostic", "included with caveat" if repeated else "unsupported by public MLE data", f"{int(run_group_map['n_run_groups'].sum()) if run_group_map is not None and not run_group_map.empty else 0} mapped run groups", "Estimates repeat sensitivity", "run groups may be changed systems, not clean seeds"),
            ("Difficulty/category slices", ", ".join(dims) if dims else "none", "slice competition", "slice leaderboards", "included with caveat" if dims else "unsupported by public MLE data", ", ".join(dims) if dims else "no dimensions", "Shows where results differ", "small slices are descriptive"),
            ("Metric-direction slices", "is_lower_better, raw score margins", "competition", "direction and margin diagnostics", "included" if has("is_lower_better") else "unsupported by public MLE data", "raw scores normalized only by thresholds", "Checks score-direction handling", "raw Kaggle metrics are not cross-task comparable"),
            ("Validity failures", "valid_submission, submission_exists", "competition report", "failure-rate slices", "included" if reliability else "unsupported by public MLE data", ", ".join(reliability) if reliability else "no reliability fields", "Separates score from operational failure", "validity definitions come from grading reports"),
            ("Raw score margins", "bronze/median thresholds", "competition", "margin distributions", "included" if has("margin_to_bronze") else "unsupported by public MLE data", "threshold-relative margins", "Measures near misses", "thresholds are competition-specific"),
            ("Task influence", "task ids and scores", "competition", "leave-one-competition-out", "included", f"{n_tasks} competitions", "Finds leverage competitions", "descriptive sensitivity only"),
            ("Equivalence", "paired task deltas", "shared competitions", "practical-band flags, non-separation bands", "included with caveat", f"+/- {PRACTICAL_EQUIVALENCE_PP:g} pp band", "Prevents reading non-significance as equality", "equivalence only if interval inside band"),
            ("Temporal drift", "snapshot/run dates", "time snapshot", "availability only", "roadmap", "created_at mostly unavailable", "No drift claim", "needs repeated benchmark snapshots"),
            ("Model-vs-agent attribution", "system/run-group names only", "agent config", "confounding audit", "unsupported by public MLE data", "system labels are coarse", "Cannot isolate model ability", "needs crossed model/scaffold/resource design"),
        ]
        return pd.DataFrame(rows, columns=["question", "available_data", "effective_unit", "supported_method", "status", "evidence", "claim_allowed", "caveat"])

    def yuri_resampling_checklist():
        return pd.DataFrame(
            [
                ("paired task bootstrap", "included", "task bootstrap and paired pair tables"),
                ("adjacent-rank distinguishability", "included", "adjacent paired deltas, pointwise/max-T intervals, MDE"),
                ("all-pairs top-K correction", "included", "top-N pair family with max-T, Holm, BH"),
                ("first-vs-rest", "included", "leader against every displayed top-N system"),
                ("rank bootstrap stability", "included", "rank intervals, rank heatmap, P(top-K)"),
                ("top-K boundary sensitivity", "included", "boundary probabilities and observed top-K overlap"),
                ("non-separation tiers", "included with caveat", "adjacent max-T bands, not equality proof"),
                ("equivalence testing", "included with caveat", "practical-band containment only; no overclaim"),
                ("MDE / power", "included with caveat", "paired-task SE approximation over heterogeneous competitions"),
                ("nested repeated-run bootstrap", "included with caveat", "run-group bootstrap; run groups are not guaranteed iid seeds"),
                ("repo/domain stratified bootstrap", "unsupported by public MLE data", "MLE categories are sparse and not a sampling frame"),
                ("factorial model x harness interaction", "unsupported by public MLE data", "no crossed model/scaffold/resource design"),
                ("score-column sensitivity", "included", "Any Medal, Above Median, Valid Submission, Submission Exists"),
                ("cost/token layer", "roadmap", "public grading reports lack complete cost/token/duration fields"),
                ("task/domain influence", "included", "leave-one-competition-out and category/difficulty slices"),
            ],
            columns=["analysis", "status", "where_or_caveat"],
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

    def run_group_mapping(data):
        if data.empty or "run_group" not in data.columns:
            return pd.DataFrame()
        return (
            data.groupby("system_id")
            .agg(
                run_groups=("run_group", lambda values: ", ".join(sorted(set(map(str, values))))),
                n_run_groups=("run_group", "nunique"),
                reports=("run_id", "nunique"),
                rows=("task_id", "size"),
                tasks=("task_id", "nunique"),
            )
            .reset_index()
            .sort_values(["tasks", "n_run_groups", "system_id"], ascending=[False, False, True])
        )

    def coverage_table(data, spec):
        if data.empty:
            return pd.DataFrame()
        total_tasks = data[spec.task_col].nunique()
        return (
            data.groupby(spec.system_col)
            .agg(tasks=(spec.task_col, "nunique"), rows=(spec.task_col, "size"), run_groups=("run_group", "nunique"))
            .reset_index()
            .assign(
                total_tasks=total_tasks,
                coverage_pct=lambda frame: 100 * frame["tasks"] / total_tasks,
                incomplete=lambda frame: frame["tasks"] < total_tasks,
            )
            .sort_values(["coverage_pct", spec.system_col], ascending=[False, True])
        )

    def duplicate_task_system_group_table(data, spec):
        if data.empty or "run_group" not in data.columns:
            return pd.DataFrame()
        counts = (
            data.groupby([spec.task_col, spec.system_col, "run_group"])
            .size()
            .reset_index(name="rows")
            .query("rows > 1")
            .sort_values(["rows", spec.system_col, spec.task_col], ascending=[False, True, True])
        )
        return counts

    def aggregation_impact_table(data, cells, spec):
        if data.empty or cells.empty:
            return pd.DataFrame()
        raw = (
            data.groupby(spec.system_col)
            .agg(raw_mean=("score", "mean"), raw_rows=("score", "size"), raw_tasks=(spec.task_col, "nunique"))
            .reset_index()
        )
        agg = (
            cells.groupby(spec.system_col)
            .agg(aggregated_mean=("score", "mean"), task_cells=("score", "size"))
            .reset_index()
        )
        return (
            raw.merge(agg, on=spec.system_col, how="outer")
            .assign(mean_delta_pp=lambda frame: 100 * (frame["raw_mean"] - frame["aggregated_mean"]))
            .sort_values("aggregated_mean", ascending=False)
        )

    def scoring_sanity_table(data):
        if data.empty:
            return pd.DataFrame()
        medal_or = (
            data[["gold_medal", "silver_medal", "bronze_medal"]]
            .fillna(0)
            .astype(float)
            .max(axis=1)
        )
        mismatch = data["any_medal"].fillna(0).astype(float) != medal_or
        invalid_medal_stack = data[["gold_medal", "silver_medal", "bronze_medal"]].fillna(0).sum(axis=1) > 1
        return pd.DataFrame(
            [
                {"check": "any_medal == gold OR silver OR bronze", "failures": int(mismatch.sum()), "rows": len(data)},
                {"check": "at most one medal tier is set", "failures": int(invalid_medal_stack.sum()), "rows": len(data)},
                {"check": "valid submissions with null raw score", "failures": int(((data["valid_submission"] == 1) & data["raw_score"].isna()).sum()), "rows": len(data)},
            ]
        )

    def metric_comparison_table(data, spec):
        if data.empty:
            return pd.DataFrame()
        rows = []
        for metric in ["score", "above_median", "valid_submission", "submission_exists"]:
            if metric not in data.columns:
                continue
            tmp = data.copy()
            tmp["score"] = tmp[metric]
            board = leaderboard_from_cells(aggregate_task_cells(tmp, spec), spec)
            for item in board.to_dict("records"):
                rows.append(
                    {
                        "metric": "any_medal" if metric == "score" else metric,
                        spec.system_col: item[spec.system_col],
                        "mean_pct": 100 * item["mean_score"],
                        "tasks": item["tasks"],
                        "rank": item["rank"],
                    }
                )
        return pd.DataFrame(rows)

    def raw_direction_table(data, spec):
        if data.empty:
            return pd.DataFrame()
        rows = []
        for direction, group in data.groupby("is_lower_better", dropna=False):
            rows.append(
                {
                    "is_lower_better": direction,
                    "rows": len(group),
                    "tasks": group[spec.task_col].nunique(),
                    "median_margin_to_bronze": group["margin_to_bronze"].median(),
                    "median_margin_to_median": group["margin_to_median"].median(),
                    "any_medal_rate_pct": 100 * group["any_medal"].mean(),
                    "valid_submission_pct": 100 * group["valid_submission"].mean(),
                }
            )
        return pd.DataFrame(rows)

    def coverage_matrix(data, spec):
        if data.empty:
            return pd.DataFrame()
        return (
            data.assign(present=1)
            .pivot_table(index=spec.system_col, columns=spec.task_col, values="present", aggfunc="max", fill_value=0)
            .sort_index()
        )

    def pairwise_win_loss(cells, spec):
        if cells.empty:
            return pd.DataFrame(), pd.DataFrame()
        systems = sorted(cells[spec.system_col].dropna().unique())
        pivot = cells.pivot_table(index=spec.task_col, columns=spec.system_col, values="score", aggfunc="mean")
        matrix = pd.DataFrame(index=systems, columns=systems, dtype=float)
        rows = []
        for left in systems:
            for right in systems:
                if left == right:
                    matrix.loc[left, right] = np.nan
                    continue
                paired = pivot[[left, right]].dropna()
                diff = paired[left] - paired[right]
                wins = int((diff > 0).sum())
                ties = int((diff == 0).sum())
                losses = int((diff < 0).sum())
                matrix.loc[left, right] = 100 * wins / len(diff) if len(diff) else np.nan
                rows.append(
                    {
                        "system_a": left,
                        "system_b": right,
                        "paired_tasks": len(diff),
                        "a_wins": wins,
                        "ties": ties,
                        "a_losses": losses,
                        "a_win_pct": 100 * wins / len(diff) if len(diff) else np.nan,
                        "mean_gap_pp": 100 * diff.mean() if len(diff) else np.nan,
                    }
                )
        return matrix, pd.DataFrame(rows)

    def adjacent_task_details(board, cells, spec, practical_equivalence_pp=PRACTICAL_EQUIVALENCE_PP):
        if board.empty or len(board) < 2:
            return pd.DataFrame()
        ordered = board.sort_values("rank")
        rows = []
        for left, right in zip(ordered.iloc[:-1].to_dict("records"), ordered.iloc[1:].to_dict("records")):
            a = left[spec.system_col]
            b = right[spec.system_col]
            pivot = cells[cells[spec.system_col].isin([a, b])].pivot_table(
                index=spec.task_col, columns=spec.system_col, values="score", aggfunc="mean"
            )
            for task_id, item in pivot.dropna().iterrows():
                diff_pp = 100 * (item[a] - item[b])
                rows.append(
                    {
                        "higher_ranked": a,
                        "lower_ranked": b,
                        "task_id": task_id,
                        "higher_score": item[a],
                        "lower_score": item[b],
                        "gap_pp": diff_pp,
                        "call": "practically tied"
                        if abs(diff_pp) <= practical_equivalence_pp
                        else ("higher wins" if diff_pp > 0 else "lower wins"),
                    }
                )
        return pd.DataFrame(rows)

    def split_slice_table(data, spec):
        if data.empty or "difficulty" not in data.columns:
            return pd.DataFrame()
        rows = []
        for split_name in ["low", "medium", "high"]:
            group = data[data["difficulty"] == split_name]
            if group.empty:
                continue
            board = leaderboard_from_cells(aggregate_task_cells(group, spec), spec)
            for item in board.to_dict("records"):
                rows.append(
                    {
                        "split": split_name,
                        spec.system_col: item[spec.system_col],
                        "any_medal_pct": 100 * item["mean_score"],
                        "tasks": item["tasks"],
                        "rank": item["rank"],
                    }
                )
        return pd.DataFrame(rows)

    def category_summary(data, spec):
        if data.empty or "category" not in data.columns:
            return pd.DataFrame()
        return (
            data.groupby(["category", spec.system_col], dropna=False)
            .agg(
                any_medal_pct=("any_medal", lambda values: 100 * values.mean()),
                valid_submission_pct=("valid_submission", lambda values: 100 * values.mean()),
                tasks=(spec.task_col, "nunique"),
                rows=(spec.task_col, "size"),
            )
            .reset_index()
            .sort_values(["category", "any_medal_pct", spec.system_col], ascending=[True, False, True])
        )

    def failure_summary(data):
        if data.empty:
            return pd.DataFrame()
        group_cols = [column for column in ["difficulty", "category"] if column in data.columns]
        if not group_cols:
            return pd.DataFrame()
        return (
            data.groupby(group_cols, dropna=False)
            .agg(
                rows=("task_id", "size"),
                tasks=("task_id", "nunique"),
                missing_submission_pct=("submission_exists", lambda values: 100 * (1 - values.mean())),
                invalid_submission_pct=("valid_submission", lambda values: 100 * (1 - values.mean())),
                any_medal_pct=("any_medal", lambda values: 100 * values.mean()),
            )
            .reset_index()
            .sort_values(["invalid_submission_pct", "missing_submission_pct"], ascending=[False, False])
        )

    def run_group_variance(data, spec):
        if data.empty or "run_group" not in data.columns:
            return pd.DataFrame()
        group_scores = (
            data.groupby([spec.system_col, "run_group", spec.task_col])
            .agg(score=("score", "mean"), above_median=("above_median", "mean"), valid_submission=("valid_submission", "mean"))
            .reset_index()
        )
        run_groups = (
            group_scores.groupby([spec.system_col, "run_group"])
            .agg(any_medal_rate=("score", "mean"), above_median_rate=("above_median", "mean"), valid_submission_rate=("valid_submission", "mean"), tasks=(spec.task_col, "nunique"))
            .reset_index()
        )
        summary = (
            run_groups.groupby(spec.system_col)
            .agg(
                run_groups=("run_group", "nunique"),
                mean_any_medal=("any_medal_rate", "mean"),
                sd_any_medal=("any_medal_rate", "std"),
                min_any_medal=("any_medal_rate", "min"),
                max_any_medal=("any_medal_rate", "max"),
                mean_valid_submission=("valid_submission_rate", "mean"),
                mean_tasks=("tasks", "mean"),
            )
            .reset_index()
        )
        return run_groups, summary

    return (
        MLE_SPEC,
        PRACTICAL_EQUIVALENCE_PP,
        adjacent_resolution,
        aggregate_task_cells,
        all_pairs_rank_resolution,
        bootstrap_summary,
        coverage_matrix,
        coverage_table,
        adjacent_task_details,
        aggregation_impact_table,
        category_summary,
        duplicate_task_system_group_table,
        filtered_rows,
        failure_summary,
        leaderboard_from_cells,
        load_mle_bench_results,
        mde_power_table,
        method_routing_table,
        metric_comparison_table,
        mo,
        nonseparation_bands,
        np,
        pass_at_k,
        pairwise_win_loss,
        pd,
        plt,
        raw_direction_table,
        rank_stability_summary,
        run_group_bootstrap_sensitivity,
        run_group_mapping,
        run_group_variance,
        scoring_sanity_table,
        slice_table,
        split_slice_table,
        task_bootstrap,
        task_influence,
        yuri_resampling_checklist,
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
def _(MLE_SPEC, coverage_table, mo, pd, rows, run_group_mapping):
    if rows.empty:
        _display = mo.md("## Data Audit / Run-Group Mapping\n\nNo rows loaded.")
        run_group_map = pd.DataFrame()
        coverage = pd.DataFrame()
    else:
        run_group_map = run_group_mapping(rows)
        coverage = coverage_table(rows, MLE_SPEC)
        incomplete = coverage[coverage["incomplete"]]
        _display = mo.vstack(
            [
                mo.md(
                    """
    ## Data Audit / Run-Group Mapping

    MLE-bench public reports include multiple run groups for some agents. This notebook normalizes timestamped `run-group_*` folders and `*_groupN` folders into base `system_id`s, then treats those run groups as repeated observations.
    """
                ),
                mo.callout(
                    mo.md(
                        f"`{len(incomplete)}` system(s) have incomplete competition coverage after normalization."
                    ),
                    kind="warn" if len(incomplete) else "success",
                ),
                mo.md("### Run Groups Collapsed Into Systems"),
                mo.ui.table(run_group_map, selection=None),
                mo.md("### System Coverage"),
                mo.ui.table(
                    coverage.assign(coverage_pct=lambda frame: frame["coverage_pct"].round(1)),
                    selection=None,
                ),
            ]
        )
    _display
    return coverage, run_group_map


@app.cell
def _(mo, pd, rows):
    if rows.empty:
        _display = mo.md("## Taxonomy And Capability Matrix\n\nNo rows loaded.")
    else:
        fields = []
        for _column in [
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
                    "column": _column,
                    "present": _column in rows.columns,
                    "non_null": int(rows[_column].notna().sum()) if _column in rows.columns else 0,
                    "unique": int(rows[_column].nunique(dropna=True)) if _column in rows.columns else 0,
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
    return cells, cells_for_rank, leaderboard, metric_col


@app.cell
def _(
    MLE_SPEC,
    cells_for_rank,
    coverage,
    method_routing_table,
    mo,
    rows,
    run_group_map,
    yuri_resampling_checklist,
):
    if rows.empty:
        _display = mo.md("## Method Routing / Capability Matrix\n\nNo rows loaded.")
        method_routing = yuri_coverage = None
    else:
        method_routing = method_routing_table(rows, cells_for_rank, coverage, run_group_map, MLE_SPEC)
        yuri_coverage = yuri_resampling_checklist()
        _display = mo.vstack(
            [
                mo.md(
                    """
    ## Method Routing / Capability Matrix

    This is the audit contract: available MLE-bench dimensions determine which statistical claims the notebook is allowed to make. Unsupported rows are kept visible so missing data does not silently become a missing caveat.
    """
                ),
                mo.ui.table(method_routing, selection=None),
                mo.md(
                    """
    ## Yuri / Resampling Coverage Checklist

    This maps the notebook against the resampling and rank-resolution patterns used in the SWE-bench, Terminal-Bench, Harness-Bench, and DeepSWE notebooks.
    """
                ),
                mo.ui.table(yuri_coverage, selection=None),
            ]
        )
    _display
    return method_routing, yuri_coverage


@app.cell
def _(
    MLE_SPEC,
    aggregation_impact_table,
    cells,
    duplicate_task_system_group_table,
    mo,
    rows,
):
    if rows.empty:
        _display = mo.md("## Aggregation Effects\n\nNo rows loaded.")
        duplicate_cells = duplicate_task_system_group_table(rows, MLE_SPEC)
        aggregation_impact = aggregation_impact_table(rows, cells, MLE_SPEC)
    else:
        duplicate_cells = duplicate_task_system_group_table(rows, MLE_SPEC)
        aggregation_impact = aggregation_impact_table(rows, cells, MLE_SPEC)
        _display = mo.vstack(
            [
                mo.md(
                    """
    ## Aggregation Effects

    The official-style estimand here is a task-cell mean: repeated `(competition, system, source, eval_scope)` observations are averaged before leaderboard means and bootstrap draws. This prevents systems with more repeated reports from receiving extra task weight.
    """
                ),
                mo.md("### Raw Row Mean vs Task-Cell Mean"),
                mo.ui.table(
                    aggregation_impact.assign(
                        raw_mean=lambda frame: (100 * frame["raw_mean"]).round(2),
                        aggregated_mean=lambda frame: (100 * frame["aggregated_mean"]).round(2),
                        mean_delta_pp=lambda frame: frame["mean_delta_pp"].round(2),
                    ),
                    selection=None,
                ),
                mo.md("### Duplicate `(task, system, run_group)` Rows"),
                mo.ui.table(duplicate_cells.head(50), selection=None),
            ]
        )
    _display
    return aggregation_impact, duplicate_cells


@app.cell
def _(
    MLE_SPEC,
    metric_comparison_table,
    mo,
    plt,
    raw_direction_table,
    rows,
    scoring_sanity_table,
):
    if rows.empty:
        _display = mo.md("## Scoring Sanity Checks\n\nNo rows loaded.")
        metric_comparison = metric_comparison_table(rows, MLE_SPEC)
        raw_direction = raw_direction_table(rows, MLE_SPEC)
        scoring_sanity = scoring_sanity_table(rows)
    else:
        scoring_sanity = scoring_sanity_table(rows)
        metric_comparison = metric_comparison_table(rows, MLE_SPEC)
        raw_direction = raw_direction_table(rows, MLE_SPEC)

        _fig, _axes = plt.subplots(1, 2, figsize=(12, max(3, 0.4 * rows["system_id"].nunique())))
        for _ax, _column, _title in [
            (_axes[0], "margin_to_bronze", "Margin to bronze"),
            (_axes[1], "margin_to_median", "Margin to median"),
        ]:
            _plot_data = rows[["system_id", _column]].dropna()
            _labels = sorted(_plot_data["system_id"].unique())
            _values = [_plot_data.loc[_plot_data["system_id"] == _label, _column] for _label in _labels]
            _ax.boxplot(_values, tick_labels=_labels, orientation="horizontal", showfliers=False)
            _ax.axvline(0, color="black", linewidth=1)
            _ax.set_title(_title)
            _ax.set_xlabel("Positive means above threshold")
        _fig.tight_layout()

        _display = mo.vstack(
            [
                mo.md(
                    """
    ## Scoring Sanity Checks

    MLE-bench's default leaderboard metric is `any_medal`. Raw Kaggle scores are not directly comparable across competitions, so threshold-relative margins are diagnostics rather than ranking metrics.
    """
                ),
                mo.ui.table(scoring_sanity, selection=None),
                mo.md("### Metric Leaderboards"),
                mo.ui.table(metric_comparison.round(2), selection=None),
                mo.md("### Raw Score Direction Handling"),
                mo.ui.table(raw_direction.round(2), selection=None),
                _fig,
            ]
        )
    _display
    return metric_comparison, raw_direction, scoring_sanity


@app.cell
def _(leaderboard, metric_col, mo, plt, top_n):
    if leaderboard.empty:
        _display = mo.md("## Observed Leaderboard\n\nNo leaderboard rows.")
    else:
        shown = leaderboard.head(top_n.value).copy()
        _fig, _ax = plt.subplots(figsize=(9, max(3, 0.35 * len(shown))))
        _ax.barh(shown["system_id"][::-1], 100 * shown["mean_score"][::-1])
        _ax.set_xlabel(f"Mean {metric_col} (%)")
        _ax.set_title("Observed MLE-bench leaderboard")
        _display = mo.vstack(
            [
                mo.md(
                    f"""
    ## Observed Leaderboard

    Rows are competition-level grading outcomes. Repeated runs for the same `(competition, system)` are mean-aggregated before ranking.
    """
                ),
                _fig,
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
    all_pairs_rank_resolution,
    cells_for_rank,
    leaderboard,
    mo,
    plt,
    top_n,
):
    pairwise_resolution, adjacent, first_vs_rest, max_t_critical = all_pairs_rank_resolution(
        leaderboard, cells_for_rank, MLE_SPEC, top_n=top_n.value
    )
    if pairwise_resolution.empty:
        _display = mo.md("## All-Pairs Rank Resolution\n\nNo pairwise rank-resolution output.")
    else:
        heat = pairwise_resolution.pivot_table(
            index="system_a",
            columns="system_b",
            values="familywise_significant",
            aggfunc="max",
            fill_value=False,
        )
        _fig, _ax = plt.subplots(figsize=(7, max(4, 0.45 * len(heat))))
        _image = _ax.imshow(heat.astype(float), vmin=0, vmax=1, cmap="Greens")
        _ax.set_xticks(range(len(heat.columns)), heat.columns, rotation=45, ha="right")
        _ax.set_yticks(range(len(heat.index)), heat.index)
        _ax.set_title(f"Top-{top_n.value} family-wise separated pairs")
        plt.colorbar(_image, ax=_ax, label="Separated after max-T")
        _fig.tight_layout()
        _display = mo.vstack(
            [
                mo.md(
                    f"""
    ## All-Pairs Rank Resolution

    All displayed top-{top_n.value} systems are compared on shared competitions. Pointwise intervals describe each pair alone; max-T controls the family of selected top-N pairwise comparisons. Holm/BH are secondary diagnostics, not the headline.

    Max-T critical value: `{max_t_critical:.2f}`.
    """
                ),
                _fig,
                mo.ui.table(
                    pairwise_resolution.drop(columns=["pair_key"], errors="ignore").round(3),
                    selection=None,
                ),
            ]
        )
    _display
    return adjacent, first_vs_rest, max_t_critical, pairwise_resolution


@app.cell
def _(first_vs_rest, mo, plt):
    if first_vs_rest is None or first_vs_rest.empty:
        _display = mo.md("## First-vs-Rest\n\nNo first-vs-rest comparisons.")
    else:
        plot_data = first_vs_rest.sort_values("gap_pp")
        _fig, _ax = plt.subplots(figsize=(8, max(3, 0.35 * len(plot_data))))
        _ax.barh(plot_data["system_b"], plot_data["gap_pp"], color="#72B7B2")
        _ax.errorbar(
            plot_data["gap_pp"],
            plot_data["system_b"],
            xerr=[
                plot_data["gap_pp"] - plot_data["max_t_lo_pp"],
                plot_data["max_t_hi_pp"] - plot_data["gap_pp"],
            ],
            fmt="none",
            color="black",
            capsize=3,
        )
        _ax.axvline(0, color="black", linewidth=1)
        _ax.set_xlabel("Leader gap vs system (pp), max-T interval")
        _ax.set_title("First-vs-rest rank resolution")
        _display = mo.vstack(
            [
                mo.md(
                    """
    ## First-vs-Rest

    The observed leader is compared to each other displayed system on shared competitions. A positive interval fully above zero is evidence that the leader separates from that system within the selected family.
    """
                ),
                _fig,
                mo.ui.table(first_vs_rest.drop(columns=["pair_key"], errors="ignore").round(3), selection=None),
            ]
        )
    _display
    return


@app.cell
def _(
    MLE_SPEC,
    cells_for_rank,
    leaderboard,
    mo,
    np,
    plt,
    rank_stability_summary,
    task_bootstrap,
    top_k,
):
    boot = task_bootstrap(cells_for_rank, MLE_SPEC)
    boot_summary, rank_probs, topk_overlap = rank_stability_summary(boot, leaderboard, MLE_SPEC, top_k.value)
    if boot_summary.empty:
        _display = mo.md("## Rank Stability And Top-K Boundary\n\nNo bootstrap output.")
    else:
        table = leaderboard.merge(boot_summary, on="system_id", how="left")
        plot_table = table.sort_values("mean_score")
        _fig, _ax = plt.subplots(figsize=(9, max(3, 0.45 * len(plot_table))))
        _ax.barh(plot_table["system_id"], 100 * plot_table["mean_score"], color="#6b8fbf")
        _ax.errorbar(
            100 * plot_table["mean_score"],
            plot_table["system_id"],
            xerr=[
                100 * (plot_table["mean_score"] - plot_table["score_p05"]),
                100 * (plot_table["score_p95"] - plot_table["mean_score"]),
            ],
            fmt="none",
            color="black",
            capsize=3,
        )
        _ax.set_xlabel("Mean Any Medal (%) with 90% task-bootstrap interval")
        _ax.set_title("Bootstrap score intervals")
        _fig_heat, _heat_ax = plt.subplots(figsize=(8, max(3, 0.45 * len(rank_probs))))
        _image = _heat_ax.imshow(100 * rank_probs.to_numpy(dtype=float), aspect="auto", cmap="Blues", vmin=0, vmax=100)
        _heat_ax.set_yticks(range(len(rank_probs.index)), rank_probs.index)
        _heat_ax.set_xticks(range(len(rank_probs.columns)), rank_probs.columns)
        _heat_ax.set_xlabel("Bootstrap rank")
        _heat_ax.set_title("Bootstrap rank distribution")
        plt.colorbar(_image, ax=_heat_ax, label="Probability (%)")
        _fig_heat.tight_layout()
        prob_col = f"p_top{top_k.value}"
        boundary = table.sort_values("rank").iloc[max(top_k.value - 4, 0) : top_k.value + 4].copy()
        _fig_boundary, _boundary_ax = plt.subplots(figsize=(8, max(3, 0.35 * len(boundary))))
        _boundary_ax.barh(boundary["system_id"][::-1], 100 * boundary[prob_col][::-1], color="#F58518")
        _boundary_ax.set_xlim(0, 100)
        _boundary_ax.set_xlabel(f"P(in top {top_k.value}) (%)")
        _boundary_ax.set_title(f"Boundary stability around top {top_k.value}")
        overlap_summary = pd.DataFrame(
            [
                {
                    "top_k": top_k.value,
                    "draws": len(topk_overlap),
                    "mean_overlap": topk_overlap["top_k_overlap"].mean() if len(topk_overlap) else np.nan,
                    "full_overlap_pct": 100 * (topk_overlap["top_k_overlap"] == top_k.value).mean() if len(topk_overlap) else np.nan,
                }
            ]
        )
        _display = mo.vstack(
            [
                mo.md(
                    f"""
    ## Rank Stability And Top-K Boundary

    Bootstrap draws resample competitions with replacement. Intervals describe how much ranks move if this competition set is treated as a sample from a broader MLE task population. Top-K probabilities audit cutoff fragility for the selected `K={top_k.value}`.
    """
                ),
                _fig,
                _fig_heat,
                _fig_boundary,
                mo.ui.table(
                    table.assign(
                        mean_score=lambda frame: (100 * frame["mean_score"]).round(2),
                        score_p05=lambda frame: (100 * frame["score_p05"]).round(2),
                        score_p95=lambda frame: (100 * frame["score_p95"]).round(2),
                        rank_p05=lambda frame: frame["rank_p05"].round(1),
                        rank_p50=lambda frame: frame["rank_p50"].round(1),
                        rank_p95=lambda frame: frame["rank_p95"].round(1),
                        **{prob_col: lambda frame: (100 * frame[prob_col]).round(1)},
                    ),
                    selection=None,
                ),
                mo.md("### Observed Top-K Overlap"),
                mo.ui.table(overlap_summary.round(2), selection=None),
            ]
        )
    _display
    return boot, boot_summary, rank_probs, topk_overlap


@app.cell
def _(adjacent, mo):
    if adjacent.empty:
        _display = mo.md("## Paired Adjacent-Rank Checks\n\nNo adjacent pairs.")
    else:
        _display = mo.vstack(
            [
                mo.md(
                    """
    ## Paired Adjacent-Rank Checks

    Adjacent systems are compared only on competitions where both have a task cell. This table is the adjacent-rank slice of the all-pairs family, so pointwise intervals, max-T intervals, Holm/BH flags, practical-band flags, and MDE fields are directly comparable.
    """
                ),
                mo.ui.table(adjacent.drop(columns=["pair_key"], errors="ignore").round(3), selection=None),
            ]
        )
    _display
    return


@app.cell
def _(mde_power_table, mo, pairwise_resolution):
    mde_summary, required_tasks = mde_power_table(pairwise_resolution)
    if mde_summary.empty:
        _display = mo.md("## MDE / Power\n\nNo MDE output.")
    else:
        _display = mo.vstack(
            [
                mo.md(
                    """
    ## MDE / Power

    MDE is computed from paired competition-level differences. Required competition counts are rough extrapolations from the median observed paired-task SE, useful for scale intuition rather than study-design guarantees.
    """
                ),
                mo.ui.table(mde_summary.round(2), selection=None),
                mo.md("### Approximate Competitions Needed For Target Gaps"),
                mo.ui.table(required_tasks.round(2), selection=None),
            ]
        )
    _display
    return mde_summary, required_tasks


@app.cell
def _(PRACTICAL_EQUIVALENCE_PP, adjacent, mo, nonseparation_bands):
    bands = nonseparation_bands(adjacent)
    if adjacent is None or adjacent.empty:
        _display = mo.md("## Non-Separation Is Not Equivalence\n\nNo adjacent pairs.")
    else:
        equivalent = int(adjacent["practically_equivalent"].sum()) if "practically_equivalent" in adjacent else 0
        not_separated = int((~adjacent["familywise_significant"]).sum()) if "familywise_significant" in adjacent else 0
        _display = mo.vstack(
            [
                mo.md(
                    f"""
    ## Non-Separation Is Not Equivalence

    `{not_separated}` adjacent pair(s) are not separated after max-T. That does **not** prove equality. Practical equivalence is only flagged when the full max-T interval sits inside the +/- `{PRACTICAL_EQUIVALENCE_PP:g}` pp band; `{equivalent}` adjacent pair(s) meet that stricter condition.
    """
                ),
                mo.md("### Non-Separation Bands"),
                mo.ui.table(bands, selection=None),
                mo.md("### Adjacent Equivalence Flags"),
                mo.ui.table(
                    adjacent[
                        [
                            "system_a",
                            "system_b",
                            "gap_pp",
                            "max_t_lo_pp",
                            "max_t_hi_pp",
                            "familywise_significant",
                            "within_practical_band",
                            "practically_equivalent",
                        ]
                    ].round(3),
                    selection=None,
                ),
            ]
        )
    _display
    return (bands,)


@app.cell
def _(
    MLE_SPEC,
    adjacent_task_details,
    cells_for_rank,
    coverage_matrix,
    leaderboard,
    mo,
    pairwise_win_loss,
    plt,
):
    if cells_for_rank.empty:
        _display = mo.md("## Pairwise And Coverage Diagnostics\n\nNo task cells available.")
        pairwise_matrix = coverage_grid = adjacent_details = None
        pairwise_long = None
    else:
        pairwise_matrix, pairwise_long = pairwise_win_loss(cells_for_rank, MLE_SPEC)
        coverage_grid = coverage_matrix(cells_for_rank, MLE_SPEC)
        adjacent_details = adjacent_task_details(leaderboard, cells_for_rank, MLE_SPEC)

        _fig1, _ax1 = plt.subplots(figsize=(6, 5))
        _image = _ax1.imshow(pairwise_matrix.astype(float), vmin=0, vmax=100, cmap="RdYlGn")
        _ax1.set_xticks(range(len(pairwise_matrix.columns)), pairwise_matrix.columns, rotation=45, ha="right")
        _ax1.set_yticks(range(len(pairwise_matrix.index)), pairwise_matrix.index)
        _ax1.set_title("Pairwise win rate: row beats column")
        plt.colorbar(_image, ax=_ax1, label="Win rate (%)")
        _fig1.tight_layout()

        _fig2, _ax2 = plt.subplots(figsize=(12, max(3, 0.45 * len(coverage_grid))))
        _ax2.imshow(coverage_grid.astype(float), aspect="auto", cmap="Greys", vmin=0, vmax=1)
        _ax2.set_yticks(range(len(coverage_grid.index)), coverage_grid.index)
        _ax2.set_xticks([])
        _ax2.set_xlabel("Competitions")
        _ax2.set_title("Task coverage matrix")

        _display = mo.vstack(
            [
                mo.md(
                    """
    ## Pairwise And Coverage Diagnostics

    These checks separate actual paired competition evidence from headline means. Pairwise win rates use only shared competitions; the coverage matrix shows where systems lack task cells.
    """
                ),
                _fig1,
                mo.ui.table(pairwise_long.round(2), selection=None),
                _fig2,
                mo.md("### Adjacent Pair Task-Level Calls"),
                mo.ui.table(adjacent_details.head(100).round(2), selection=None),
            ]
        )
    _display
    return adjacent_details, coverage_grid, pairwise_long, pairwise_matrix


@app.cell
def _(MLE_SPEC, mo, rows, slice_dimension, slice_table):
    slices = slice_table(rows, MLE_SPEC, slice_dimension.value)
    if slices.empty:
        _display = mo.md("## Slice Sensitivity\n\nNo slice output for the selected dimension.")
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
    return (slices,)


@app.cell
def _(
    MLE_SPEC,
    category_summary,
    failure_summary,
    mo,
    rows,
    split_slice_table,
):
    if rows.empty:
        _display = mo.md("## Official Splits, Categories, And Failure Modes\n\nNo rows loaded.")
        official_splits = category_slices = failure_slices = None
    else:
        official_splits = split_slice_table(rows, MLE_SPEC)
        category_slices = category_summary(rows, MLE_SPEC)
        failure_slices = failure_summary(rows)
        _display = mo.vstack(
            [
                mo.md(
                    """
    ## Official Splits, Categories, And Failure Modes

    The Low / Medium / High splits are official MLE-bench complexity buckets. Category and failure tables are diagnostics for where medal rates are driven by task type or submission validity rather than pure score quality.
    """
                ),
                mo.md("### Low / Medium / High Split Leaderboards"),
                mo.ui.table(official_splits.round(2), selection=None),
                mo.md("### Category-Level Medal And Validity Rates"),
                mo.ui.table(category_slices.round(2), selection=None),
                mo.md("### Missing / Invalid Submission Failure Rates"),
                mo.ui.table(failure_slices.round(2), selection=None),
            ]
        )
    _display
    return category_slices, failure_slices, official_splits


@app.cell
def _(MLE_SPEC, mo, pass_at_k, pd, plt, rows, run_group_bootstrap_sensitivity, run_group_variance, top_k):
    passk = pass_at_k(rows, MLE_SPEC, max_k=top_k.value)
    if passk.empty:
        _display = mo.md("## Repeated-Trial Variance And Pass@k Diagnostics\n\nNo repeated-trial output.")
        run_group_boot = run_group_boot_summary = run_group_scores = run_group_summary = None
    else:
        run_group_scores, run_group_summary = run_group_variance(rows, MLE_SPEC)
        run_group_boot, run_group_boot_summary = run_group_bootstrap_sensitivity(rows, MLE_SPEC)
        _fig, _ax = plt.subplots(figsize=(8, 4))
        for system, group in passk.groupby("system_id"):
            _ax.plot(group["k"], 100 * group["pass_at_k"], marker="o", label=system)
        _ax.set_xlabel("k observed run groups")
        _ax.set_ylabel("Empirical pass@k Any Medal (%)")
        _ax.set_title("Observed repeated-run pass@k")
        _ax.legend(loc="best", fontsize="small")
        _display = mo.vstack(
            [
                mo.md(
                    """
    ## Repeated-Trial Variance And Pass@k Diagnostics

    Pass@k here is empirical over observed repeated reports: for each competition, did any of the first `k` observed trials earn any medal? It is a diagnostic, not the official leaderboard metric. Run-group bootstrap treats mapped run groups as repeat-like observations, but public MLE-bench run groups may also contain system changes.
    """
                ),
                _fig,
                mo.ui.table(passk.assign(pass_at_k=lambda frame: (100 * frame["pass_at_k"]).round(2)), selection=None),
                mo.md("### Run-Group Variance By System"),
                mo.ui.table(
                    run_group_summary.assign(
                        mean_any_medal=lambda frame: (100 * frame["mean_any_medal"]).round(2),
                        sd_any_medal=lambda frame: (100 * frame["sd_any_medal"]).round(2),
                        min_any_medal=lambda frame: (100 * frame["min_any_medal"]).round(2),
                        max_any_medal=lambda frame: (100 * frame["max_any_medal"]).round(2),
                        mean_valid_submission=lambda frame: (100 * frame["mean_valid_submission"]).round(2),
                    ),
                    selection=None,
                ),
                mo.md("### Individual Run-Group Scores"),
                mo.ui.table(
                    run_group_scores.assign(
                        any_medal_rate=lambda frame: (100 * frame["any_medal_rate"]).round(2),
                        above_median_rate=lambda frame: (100 * frame["above_median_rate"]).round(2),
                        valid_submission_rate=lambda frame: (100 * frame["valid_submission_rate"]).round(2),
                    ),
                    selection=None,
                ),
                mo.md("### Run-Group Bootstrap Sensitivity"),
                mo.ui.table(
                    run_group_boot_summary.assign(
                        run_group_boot_mean=lambda frame: (100 * frame["run_group_boot_mean"]).round(2),
                        run_group_boot_p05=lambda frame: (100 * frame["run_group_boot_p05"]).round(2),
                        run_group_boot_p95=lambda frame: (100 * frame["run_group_boot_p95"]).round(2),
                    )
                    if run_group_boot_summary is not None and not run_group_boot_summary.empty
                    else pd.DataFrame(),
                    selection=None,
                ),
            ]
        )
    _display
    return passk, run_group_boot, run_group_boot_summary, run_group_scores, run_group_summary


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
    return (influence,)


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
def _(
    adjacent,
    boot_summary,
    coverage,
    influence,
    leaderboard,
    load_status,
    method_routing,
    mo,
    pairwise_resolution,
    rank_probs,
    rows,
):
    if rows.empty:
        _display = mo.md("## Audit Summary\n\nNo data loaded, so no statistical claims are supported.")
    else:
        leader = leaderboard.iloc[0]["system_id"] if not leaderboard.empty else "n/a"
        source_kind = load_status.get("source_kind", "unknown")
        incomplete_systems = []
        if coverage is not None and not coverage.empty and "incomplete" in coverage.columns:
            incomplete_systems = coverage.loc[coverage["incomplete"], "system_id"].tolist()
        top_unstable_task = "n/a"
        top_unstable_shift = "n/a"
        if influence is not None and not influence.empty:
            top_unstable_task = influence.iloc[0]["task_id"]
            top_unstable_shift = f"{influence.iloc[0]['max_abs_shift_pp']:.2f} pp"
        sensitive_pair = "n/a"
        if adjacent is not None and not adjacent.empty:
            pair = adjacent.assign(abs_gap=lambda frame: frame["gap_pp"].abs()).sort_values("abs_gap").iloc[0]
            sensitive_pair = f"{pair['system_a']} vs {pair['system_b']} ({pair['gap_pp']:.2f} pp paired gap)"
        leader_boot = "n/a"
        if boot_summary is not None and not boot_summary.empty and leader in set(boot_summary["system_id"]):
            item = boot_summary[boot_summary["system_id"] == leader].iloc[0]
            leader_boot = f"rank {item['rank_p05']:.1f}-{item['rank_p95']:.1f}, score {100 * item['score_p05']:.1f}-{100 * item['score_p95']:.1f}%"
        separated_pairs = "n/a"
        if pairwise_resolution is not None and not pairwise_resolution.empty:
            separated_pairs = f"{int(pairwise_resolution['familywise_significant'].sum())}/{len(pairwise_resolution)} top-N pairs max-T separated"
        adjacent_separated = "n/a"
        if adjacent is not None and not adjacent.empty:
            adjacent_separated = f"{int(adjacent['familywise_significant'].sum())}/{len(adjacent)} adjacent pairs max-T separated"
        widest_rank = "n/a"
        if boot_summary is not None and not boot_summary.empty:
            wide = boot_summary.assign(width=lambda frame: frame["rank_p95"] - frame["rank_p05"]).sort_values("width", ascending=False).iloc[0]
            widest_rank = f"{wide['system_id']} ({wide['rank_p05']:.1f}-{wide['rank_p95']:.1f})"
        unsupported = "n/a"
        if method_routing is not None and not method_routing.empty:
            blocked = method_routing[method_routing["status"].isin(["unsupported by public MLE data", "roadmap"])]
            unsupported = ", ".join(blocked["question"].head(5).tolist()) if len(blocked) else "none"
        _display = mo.md(
            f"""
    ## Audit Summary

    - Data source: `{source_kind}`; public grading reports: `{len(load_status.get("paths", []))}`.
    - Rows: `{len(rows):,}`; competitions: `{rows["task_id"].nunique():,}`; `{rows["system_id"].nunique():,}` normalized systems; run groups: `{rows["run_group"].nunique() if "run_group" in rows.columns else "n/a"}`.
    - Count check phrase: {rows["system_id"].nunique():,} normalized systems.
    - Observed leaderboard claim: `{leader}` has the highest mean Any Medal rate after repeated `(competition, system, source, eval_scope)` aggregation.
    - Rank-resolution claim: `{separated_pairs}`; adjacent-rank claim: `{adjacent_separated}`.
    - Bootstrap stability claim: observed leader `{leader_boot}`; widest rank interval: `{widest_rank}`.
    - Coverage and slice limitation: `{len(incomplete_systems)}` system(s) have incomplete competition coverage: `{", ".join(incomplete_systems) if incomplete_systems else "none"}`.
    - Most rank-sensitive adjacent pair: `{sensitive_pair}`.
    - Most influential competition by leave-one-out shift: `{top_unstable_task}` (`{top_unstable_shift}`).
    - Unsupported/roadmap claims kept visible: `{unsupported}`.

    Interpretation caveat: MLE-bench reports public grading outcomes for heterogeneous Kaggle competitions. Rank intervals quantify competition-sampling sensitivity, not uncertainty in Kaggle scoring itself; public leaderboard comparability also depends on agent resources, dates, and whether run groups are genuinely repeated seeds rather than changed systems.
    """
        )
    _display
    return


if __name__ == "__main__":
    app.run()
