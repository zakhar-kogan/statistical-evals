from __future__ import annotations

import os

import pandas as pd

try:
    import panel as pn
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError as exc:  # pragma: no cover - exercised by dashboard smoke checks.
    raise SystemExit(
        "Install dashboard dependencies with: uv sync --extra dev --extra dashboard"
    ) from exc

from deepswe_rank_stability.analysis.resampling import (
    aggregate_task_model_scores,
    bootstrap_by_dimension,
    bootstrap_rank_stability,
    filter_trials,
    model_task_coverage_by_dimension,
    score_matrix,
    task_influence_table,
)
from deepswe_rank_stability.dashboard.state import (
    DashboardSelection,
    coerce_source_for_mode,
    contender_model_order,
    eligible_variance_dimensions,
    friendly_empty_message,
    order_pairwise_by_strength,
    plotly_top_first_categoryarray,
    rank_axis_range,
    rank_model_order,
    slice_values_with_summaries,
    source_options,
    submit_selection,
    top_model_summary,
    variance_empty_message,
)
from deepswe_rank_stability.data.deepswe import load_dataset

pn.extension("tabulator", "perspective", "plotly", sizing_mode="stretch_width")

DATASET = load_dataset()
TRIALS = DATASET.trials

ALL_SOURCES = sorted(TRIALS["source"].dropna().unique())
SCOPE_OPTIONS = ["All", *sorted(TRIALS["eval_scope"].dropna().unique())]
OUTCOME_OPTIONS = ["All", *sorted(TRIALS["outcome"].dropna().unique())]
LANGUAGE_OPTIONS = ["All", *sorted(TRIALS.get("language", pd.Series(dtype=str)).dropna().unique())]
REPOSITORY_OPTIONS = ["All", *sorted(TRIALS.get("repository", pd.Series(dtype=str)).dropna().unique())]
MODEL_OPTIONS = sorted(TRIALS["model_key"].dropna().unique())

include_cross_benchmark = pn.widgets.Checkbox(name="Include cross-benchmark sources", value=False)
source = pn.widgets.Select(
    name="Source",
    options=source_options(ALL_SOURCES, include_cross_benchmark=False),
    value="deep-swe",
)
eval_scope = pn.widgets.Select(name="Eval scope", options=SCOPE_OPTIONS, value="full")
included = pn.widgets.Select(
    name="Included in score",
    options={"Included only": True, "All": None, "Excluded only": False},
    value=True,
)
outcome = pn.widgets.Select(name="Outcome", options=OUTCOME_OPTIONS, value="All")
language = pn.widgets.Select(name="Language", options=LANGUAGE_OPTIONS, value="All")
repository = pn.widgets.Select(name="Repository", options=REPOSITORY_OPTIONS, value="All")
models = pn.widgets.MultiChoice(name="Model + effort", options=MODEL_OPTIONS, value=[])
DEFAULT_DRAWS = int(os.environ.get("DEEPSWE_DEFAULT_DRAWS", "2_000"))

draws = pn.widgets.IntInput(name="Bootstrap draws", value=DEFAULT_DRAWS, start=100, end=20_000, step=100)
seed = pn.widgets.IntInput(name="Random seed", value=0, start=0, end=1_000_000)
variance_dimension = pn.widgets.Select(
    name="Variance dimension",
    options=["language"],
    value="language",
)
variance_metric = pn.widgets.RadioButtonGroup(
    name="Variance heatmap metric",
    options={"Observed score": "observed_score", "Top-1 probability": "top1_probability"},
    value="observed_score",
    button_type="default",
)
run = pn.widgets.Button(name="Recompute", button_type="primary", icon="refresh")

_submitted_selection: DashboardSelection | None = None
_last_clicks = -1


def _sync_source_options(event: object | None = None) -> None:
    del event
    options = source_options(ALL_SOURCES, include_cross_benchmark=include_cross_benchmark.value)
    source.options = options
    source.value = coerce_source_for_mode(
        source.value,
        ALL_SOURCES,
        include_cross_benchmark=include_cross_benchmark.value,
    )


include_cross_benchmark.param.watch(_sync_source_options, "value")


def _none_if_all(value: str) -> str | None:
    return None if value == "All" else value


def _pending_selection() -> DashboardSelection:
    return DashboardSelection(
        source=source.value,
        eval_scope=eval_scope.value,
        included_in_score=included.value,
        outcome=outcome.value,
        language=language.value,
        repository=repository.value,
        model_keys=tuple(models.value or ()),
        draws=int(draws.value),
        seed=int(seed.value),
        include_cross_benchmark=bool(include_cross_benchmark.value),
    )


def _selection_for_clicks(clicks: int) -> DashboardSelection:
    global _last_clicks, _submitted_selection
    _submitted_selection, _last_clicks = submit_selection(
        current=_submitted_selection,
        trigger_count=clicks,
        last_trigger_count=_last_clicks,
        pending=_pending_selection(),
    )
    return _submitted_selection


def _filtered_trials(selection: DashboardSelection) -> pd.DataFrame:
    return filter_trials(
        TRIALS,
        source=_none_if_all(selection.source),
        eval_scope=_none_if_all(selection.eval_scope),
        included_in_score=selection.included_in_score,
        outcome=_none_if_all(selection.outcome),
        language=_none_if_all(selection.language),
        repository=_none_if_all(selection.repository),
        model_keys=selection.model_keys or None,
    )


def _metric_strip(filtered: pd.DataFrame, selection: DashboardSelection) -> pn.pane.HTML:
    chips = [
        ("Trials", f"{len(filtered):,}"),
        ("Tasks", f"{filtered['task_name'].nunique():,}"),
        ("Models", f"{filtered['model_key'].nunique():,}"),
        ("Draws", f"{selection.draws:,}"),
    ]
    html = "".join(
        f"""
        <div style="border:1px solid #d1d5db;border-radius:6px;padding:10px 12px;background:#fff;min-width:120px">
          <div style="font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:#6b7280">{label}</div>
          <div style="font-size:22px;line-height:1.25;font-weight:650;color:#111827;white-space:nowrap">{value}</div>
        </div>
        """
        for label, value in chips
    )
    html += f"""
    <div style="border:1px solid #d1d5db;border-radius:6px;padding:10px 12px;background:#fff;min-width:360px">
      <div style="font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:#6b7280">Last run</div>
      <div style="font-size:14px;line-height:1.35;font-weight:600;color:#111827">{selection.label()}</div>
    </div>
    """
    return pn.pane.HTML(f'<div style="display:flex;gap:10px;flex-wrap:wrap;margin:4px 0 12px 0">{html}</div>')


def _run_anchor_strip(leaderboard: pd.DataFrame, selection: DashboardSelection) -> pn.pane.HTML:
    top_model = top_model_summary(leaderboard)
    score = "n/a" if top_model["observed_score"] is None else f"{top_model['observed_score']:.3f}"
    top1 = "n/a" if top_model["top1_probability"] is None else f"{top_model['top1_probability']:.1%}"
    rank_interval = (
        "n/a"
        if top_model["rank_p05"] is None
        else f"{top_model['rank_p05']:.0f}-{top_model['rank_p95']:.0f}"
    )
    chips = [
        ("Top model", top_model["model_key"]),
        ("Observed rank", top_model["observed_rank"] or "n/a"),
        ("Observed score", score),
        ("Top-1 probability", top1),
        ("Rank p05-p95", rank_interval),
        ("Draws", f"{selection.draws:,}"),
    ]
    html = "".join(
        f"""
        <div style="border:1px solid #d1d5db;border-radius:6px;padding:8px 10px;background:#f8fafc;min-width:120px">
          <div style="font-size:10px;text-transform:uppercase;letter-spacing:.05em;color:#64748b">{label}</div>
          <div style="font-size:15px;line-height:1.25;font-weight:650;color:#0f172a;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{value}</div>
        </div>
        """
        for label, value in chips
    )
    return pn.pane.HTML(f'<div style="display:flex;gap:8px;flex-wrap:wrap;margin:0 0 12px 0">{html}</div>')


def _with_anchor(anchor: pn.pane.HTML, *items: pn.viewable.Viewable) -> pn.Column:
    return pn.Column(anchor.clone(), *items)


def _rank_interval_figure(leaderboard: pd.DataFrame) -> go.Figure:
    order = rank_model_order(leaderboard)
    plot_frame = leaderboard.set_index("model_key").loc[order].reset_index()
    y_positions = list(range(len(plot_frame)))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=plot_frame["rank_p50"],
            y=y_positions,
            error_x={
                "type": "data",
                "array": plot_frame["rank_p75"] - plot_frame["rank_p50"],
                "arrayminus": plot_frame["rank_p50"] - plot_frame["rank_p25"],
                "thickness": 8,
                "width": 0,
                "color": "rgba(37, 99, 235, 0.42)",
            },
            mode="markers",
            marker={"symbol": "line-ns-open", "size": 18, "color": "#2563eb"},
            name="median rank with p25-p75 band",
            customdata=plot_frame[
                [
                    "model_key",
                    "observed_rank",
                    "rank_p05",
                    "rank_p25",
                    "rank_p75",
                    "rank_p95",
                    "top1_probability",
                    "top3_probability",
                ]
            ],
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Median rank: %{x:.1f}<br>"
                "Observed rank: %{customdata[1]}<br>"
                "p05-p95: %{customdata[2]:.1f} - %{customdata[5]:.1f}<br>"
                "p25-p75: %{customdata[3]:.1f} - %{customdata[4]:.1f}<br>"
                "Top-1 probability: %{customdata[6]:.1%}<br>"
                "Top-3 probability: %{customdata[7]:.1%}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=plot_frame["observed_rank"],
            y=y_positions,
            mode="markers",
            marker={"symbol": "diamond", "size": 9, "color": "#111827"},
            name="observed rank",
            customdata=plot_frame["model_key"],
            hovertemplate="<b>%{customdata}</b><br>Observed rank: %{x}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=plot_frame["rank_p95"],
            y=y_positions,
            error_x={
                "type": "data",
                "array": plot_frame["rank_p95"] - plot_frame["rank_p95"],
                "arrayminus": plot_frame["rank_p95"] - plot_frame["rank_p05"],
                "thickness": 2,
                "width": 0,
                "color": "rgba(17, 24, 39, 0.28)",
            },
            mode="markers",
            marker={"size": 1, "color": "rgba(0,0,0,0)"},
            name="p05-p95 interval",
            hoverinfo="skip",
        )
    )
    axis_start, axis_end = rank_axis_range(leaderboard)
    fig.update_layout(
        height=max(460, 34 * len(plot_frame)),
        margin={"l": 8, "r": 24, "t": 52, "b": 88},
        title="Rank interval chart: rank 1 is best and leftmost",
        xaxis_title="Rank",
        yaxis_title="",
        legend_orientation="h",
        legend_y=-0.16,
        legend_x=0,
    )
    fig.update_xaxes(range=[axis_start, axis_end], dtick=1, autorange=False)
    fig.update_yaxes(
        tickmode="array",
        tickvals=y_positions,
        ticktext=plot_frame["model_key"],
        autorange="reversed",
    )
    return fig


def _rank_heatmap_figure(rank_distribution: pd.DataFrame, leaderboard: pd.DataFrame) -> go.Figure:
    order = rank_model_order(leaderboard)
    lookup = leaderboard.set_index("model_key")[["observed_rank", "rank_p50"]]
    heatmap = rank_distribution.merge(lookup, left_on="model_key", right_index=True, how="left")
    pivot = (
        heatmap.pivot_table(index="model_key", columns="rank", values="probability", fill_value=0, observed=False)
        .reindex(order)
        .sort_index(axis=1)
    )
    observed = lookup.reindex(pivot.index)["observed_rank"].to_numpy()
    median = lookup.reindex(pivot.index)["rank_p50"].to_numpy()
    customdata = [
        [[observed[row_index], median[row_index]] for _ in pivot.columns]
        for row_index in range(len(pivot.index))
    ]
    fig = go.Figure(
        data=go.Heatmap(
            x=pivot.columns,
            y=pivot.index,
            z=pivot.to_numpy(),
            customdata=customdata,
            colorscale="Blues",
            zmin=0,
            zmax=1,
            colorbar={"title": "Probability"},
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Rank: %{x}<br>"
                "Probability: %{z:.1%}<br>"
                "Observed rank: %{customdata[0]}<br>"
                "Median rank: %{customdata[1]:.1f}<extra></extra>"
            ),
        )
    )
    axis_start, axis_end = rank_axis_range(leaderboard)
    fig.update_layout(
        title="Rank distribution: top models first, rank 1 left",
        height=max(460, 28 * len(pivot.index)),
        margin={"l": 8, "r": 24, "t": 48, "b": 48},
        xaxis_title="Bootstrap rank",
        yaxis_title="",
    )
    fig.update_xaxes(range=[axis_start, axis_end], autorange=False)
    fig.update_yaxes(categoryorder="array", categoryarray=plotly_top_first_categoryarray(order), autorange="reversed")
    return fig


def _pairwise_figure(pairwise: pd.DataFrame, leaderboard: pd.DataFrame) -> tuple[go.Figure, pd.DataFrame]:
    order = rank_model_order(leaderboard)
    matrix = pairwise.reindex(index=order, columns=order)
    _, strengths = order_pairwise_by_strength(pairwise)
    hover_text = [
        [f"{row_model} beats {column_model} in {value:.1%} of bootstrap draws" for column_model, value in row.items()]
        for row_model, row in matrix.iterrows()
    ]
    fig = go.Figure(
        data=go.Heatmap(
            x=matrix.columns,
            y=matrix.index,
            z=matrix.to_numpy(),
            text=hover_text,
            colorscale=[
                [0.0, "#b91c1c"],
                [0.5, "#f8fafc"],
                [1.0, "#047857"],
            ],
            zmin=0,
            zmax=1,
            colorbar={"title": "Win probability"},
            hovertemplate="%{text}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Pairwise bootstrap win probability: top models first on both axes",
        height=max(520, 28 * len(matrix.index)),
        margin={"l": 8, "r": 24, "t": 48, "b": 90},
        xaxis_title="Opponent",
        yaxis_title="Model",
    )
    fig.update_xaxes(categoryorder="array", categoryarray=order)
    fig.update_yaxes(categoryorder="array", categoryarray=plotly_top_first_categoryarray(order), autorange="reversed")
    return fig, strengths


def _top_probability_figure(leaderboard: pd.DataFrame) -> go.Figure:
    order = contender_model_order(leaderboard)
    top_prob_data = leaderboard.melt(
        id_vars=["model_key"],
        value_vars=["top1_probability", "top3_probability"],
        var_name="metric",
        value_name="probability",
    )
    top_prob_data = top_prob_data[top_prob_data["model_key"].isin(order)]
    fig = px.bar(
        top_prob_data,
        y="model_key",
        x="probability",
        color="metric",
        orientation="h",
        barmode="group",
        title="Top-rank probabilities",
        range_x=(0, 1),
        hover_data={"probability": ":.1%", "model_key": True, "metric": True},
    )
    fig.update_layout(height=max(320, 30 * len(order)), margin={"l": 8, "r": 24, "t": 48, "b": 48})
    fig.update_yaxes(categoryorder="array", categoryarray=list(reversed(order)))
    return fig


def _task_influence_table(filtered: pd.DataFrame, leaderboard: pd.DataFrame, limit: int = 30) -> pd.DataFrame:
    return task_influence_table(
        filtered,
        contender_models=contender_model_order(leaderboard),
        limit=limit,
    )


def _task_swing_table(filtered: pd.DataFrame, limit: int = 30) -> pd.DataFrame:
    matrix = score_matrix(aggregate_task_model_scores(filtered))
    if matrix.empty:
        return pd.DataFrame()
    spread = matrix.max(axis=1, skipna=True) - matrix.min(axis=1, skipna=True)
    coverage = matrix.notna().sum(axis=1)
    task_metadata = (
        filtered[["task_name", "language", "repository", "problem_title"]]
        .drop_duplicates("task_name")
        .set_index("task_name")
    )
    table = pd.DataFrame(
        {
            "task_name": spread.index,
            "score_spread": spread.values,
            "models_with_result": coverage.values,
        }
    ).join(task_metadata, on="task_name")
    return table.sort_values(["score_spread", "models_with_result"], ascending=[False, False]).head(limit)


def _language_breakdown(filtered: pd.DataFrame) -> go.Figure | None:
    if "language" not in filtered.columns or filtered["language"].dropna().empty:
        return None
    summary = (
        filtered.groupby(["language", "model_key"], dropna=False, observed=False)
        .agg(score=("score_value", "mean"), trials=("trial_name", "count"))
        .reset_index()
    )
    fig = px.scatter(
        summary,
        x="language",
        y="score",
        color="model_key",
        size="trials",
        hover_data={"model_key": True, "language": True, "score": ":.3f", "trials": ":,"},
        title="Score by language and model",
    )
    fig.update_layout(height=420, margin={"l": 8, "r": 24, "t": 48, "b": 80}, showlegend=False)
    return fig


def _score_stability_figure(leaderboard: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        leaderboard,
        x="observed_score",
        y="rank_interval_width",
        color="top1_probability",
        size="top3_probability",
        hover_name="model_key",
        hover_data={
            "observed_rank": True,
            "rank_p05": ":.1f",
            "rank_p95": ":.1f",
            "top1_probability": ":.1%",
            "top3_probability": ":.1%",
        },
        title="Score versus rank instability",
        color_continuous_scale="Viridis",
    )
    fig.update_layout(height=420, margin={"l": 8, "r": 24, "t": 48, "b": 48})
    fig.update_yaxes(title="p95-p05 rank interval width")
    return fig


def _slice_summary_table(summary: pd.DataFrame) -> pn.widgets.Tabulator:
    if summary.empty:
        table = summary
    else:
        table = (
            summary.sort_values(["slice_value", "observed_rank", "observed_score"], ascending=[True, True, False])
            .groupby("slice_value", as_index=False, observed=False)
            .first()
            [
                [
                    "slice_value",
                    "n_tasks",
                    "n_trials",
                    "n_models",
                    "model_key",
                    "observed_score",
                    "top1_probability",
                    "rank_interval_width",
                ]
            ]
            .rename(columns={"model_key": "top_model"})
        )
    return pn.widgets.Tabulator(
        table,
        height=260,
        pagination="remote",
        page_size=20,
        sorters=[{"field": "slice_value", "dir": "asc"}],
        formatters={
            "observed_score": {"type": "progress", "max": 1},
            "top1_probability": {"type": "progress", "max": 1},
        },
    )


def _slice_leaderboard_heatmap(summary: pd.DataFrame, metric: str, top_models: list[str]) -> go.Figure:
    if summary.empty:
        return go.Figure().update_layout(title="Slice leaderboard heatmap: no eligible slices")
    heatmap = summary[summary["model_key"].isin(top_models)]
    pivot = heatmap.pivot_table(
        index="slice_value",
        columns="model_key",
        values=metric,
        aggfunc="mean",
        observed=False,
    )
    slice_order = (
        summary.groupby("slice_value", observed=False)["n_tasks"]
        .max()
        .sort_values(ascending=False)
        .index.astype(str)
        .tolist()
    )
    pivot = pivot.reindex(index=slice_order, columns=top_models)
    title_metric = "observed score" if metric == "observed_score" else "top-1 probability"
    fig = go.Figure(
        data=go.Heatmap(
            x=pivot.columns,
            y=pivot.index,
            z=pivot.to_numpy(),
            colorscale="Viridis",
            zmin=0,
            zmax=1,
            colorbar={"title": title_metric},
            hovertemplate="Slice: %{y}<br>Model: %{x}<br>Value: %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Slice leaderboard heatmap by {title_metric}",
        height=max(360, 42 * len(pivot.index)),
        margin={"l": 8, "r": 24, "t": 48, "b": 96},
        xaxis_title="Model",
        yaxis_title="Slice",
    )
    fig.update_yaxes(categoryorder="array", categoryarray=list(pivot.index), autorange="reversed")
    return fig


def _language_rank_interval_figure(summary: pd.DataFrame, slice_value: str, top_models: list[str]) -> go.Figure:
    plot_frame = summary[(summary["slice_value"] == slice_value) & (summary["model_key"].isin(top_models))].copy()
    plot_frame = plot_frame.sort_values(["observed_rank", "observed_score"], ascending=[True, False])
    y_positions = list(range(len(plot_frame)))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=plot_frame["rank_p50"],
            y=y_positions,
            error_x={
                "type": "data",
                "array": plot_frame["rank_p95"] - plot_frame["rank_p50"],
                "arrayminus": plot_frame["rank_p50"] - plot_frame["rank_p05"],
                "thickness": 2,
                "width": 0,
                "color": "rgba(37, 99, 235, 0.45)",
            },
            mode="markers",
            marker={"size": 7, "color": "#2563eb"},
            customdata=plot_frame[["model_key", "observed_rank", "top1_probability"]],
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Median rank: %{x:.1f}<br>"
                "Observed rank: %{customdata[1]}<br>"
                "Top-1 probability: %{customdata[2]:.1%}<extra></extra>"
            ),
        )
    )
    max_rank = float(max(plot_frame["rank_p95"].max(), plot_frame["observed_rank"].max()))
    fig.update_layout(
        title=f"{slice_value}: rank p05-p95 for contenders",
        height=max(300, 28 * len(plot_frame)),
        margin={"l": 8, "r": 24, "t": 48, "b": 48},
        xaxis_title="Rank, rank 1 is best",
        yaxis_title="",
        showlegend=False,
    )
    fig.update_xaxes(range=[0.5, max_rank + 0.8], dtick=1, autorange=False)
    fig.update_yaxes(tickmode="array", tickvals=y_positions, ticktext=plot_frame["model_key"], autorange="reversed")
    return fig


def _language_rank_small_multiples(summary: pd.DataFrame, top_models: list[str]) -> list[pn.viewable.Viewable]:
    sections: list[pn.viewable.Viewable] = []
    for slice_value in slice_values_with_summaries(summary):
        sections.append(
            pn.pane.Plotly(
                _language_rank_interval_figure(summary, slice_value, top_models),
                config={"responsive": True, "displaylogo": False},
            )
        )
    return sections


def _coverage_table(coverage: pd.DataFrame, top_models: list[str]) -> pn.widgets.Tabulator:
    table = coverage[coverage["model_key"].isin(top_models)].copy() if not coverage.empty else coverage
    if not table.empty:
        table = table.sort_values(["slice_value", "coverage", "model_key"], ascending=[True, True, True])
    return pn.widgets.Tabulator(
        table,
        height=300,
        pagination="remote",
        page_size=25,
        sorters=[{"field": "slice_value", "dir": "asc"}, {"field": "coverage", "dir": "asc"}],
        formatters={"coverage": {"type": "progress", "max": 1}},
    )


def _repository_underpowered_note(filtered: pd.DataFrame) -> pn.pane.Alert | None:
    if "repository" not in filtered.columns or filtered["repository"].dropna().empty:
        return None
    max_tasks = int(filtered.groupby("repository", dropna=False, observed=False)["task_name"].nunique().max())
    return pn.pane.Alert(
        f"Repository is not offered as a bootstrap dimension because repository slices are underpowered here. "
        f"The largest repository slice has {max_tasks} tasks under the current filters.",
        alert_type="light",
    )


def _variance_controls(filtered: pd.DataFrame) -> pn.Row | pn.pane.Alert:
    options = eligible_variance_dimensions(filtered, min_tasks=10, min_models=2)
    empty_message = variance_empty_message(options)
    if empty_message is not None:
        return pn.pane.Alert(empty_message, alert_type="warning")
    variance_dimension.options = options
    if variance_dimension.value not in options:
        variance_dimension.value = options[0]
    return pn.Row(variance_dimension, variance_metric)


def _variance_view(filtered: pd.DataFrame, selection: DashboardSelection, leaderboard: pd.DataFrame) -> pn.Column:
    controls = _variance_controls(filtered)
    repository_note = _repository_underpowered_note(filtered)
    if isinstance(controls, pn.pane.Alert):
        sections: list[pn.viewable.Viewable] = [controls]
        if repository_note is not None:
            sections.append(repository_note)
        return pn.Column(*sections)

    dimension = variance_dimension.value
    top_models = contender_model_order(leaderboard)
    try:
        sliced = bootstrap_by_dimension(
            filtered,
            dimension=dimension,
            draws=selection.draws,
            seed=selection.seed,
            min_tasks=10,
            min_models=2,
        )
    except ValueError as exc:
        return pn.Column(pn.pane.Alert(str(exc), alert_type="danger"))

    summary = sliced.summaries.copy()
    skipped = sliced.skipped_slices.copy()
    coverage = model_task_coverage_by_dimension(filtered, dimension=dimension)

    sections: list[pn.viewable.Viewable] = [
        controls,
    ]
    if repository_note is not None:
        sections.append(repository_note)
    sections.extend(
        [
        pn.pane.Markdown(
            f"### Variance by `{dimension}`\n"
            "Eligible slices are bootstrapped independently with at least 10 tasks and 2 models. "
            "Skipped slices stay visible so small-sample gaps are explicit."
        ),
        _slice_summary_table(summary),
        pn.Row(
            pn.pane.Plotly(
                _slice_leaderboard_heatmap(summary, variance_metric.value, top_models),
                config={"responsive": True, "displaylogo": False},
            ),
        ),
        pn.pane.Markdown("### Rank intervals by language"),
        *_language_rank_small_multiples(summary, top_models),
        pn.pane.Markdown("### Missing-cell coverage"),
        _coverage_table(coverage, top_models),
        pn.pane.Markdown("### Skipped slices"),
        pn.widgets.Tabulator(skipped, height=220, pagination="remote", page_size=20),
        ]
    )
    return pn.Column(*sections)


def _leaderboard_table(leaderboard: pd.DataFrame) -> pn.widgets.Tabulator:
    visible_columns = [
        "model_key",
        "observed_rank",
        "observed_score",
        "rank_p50",
        "rank_p05",
        "rank_p95",
        "rank_interval_width",
        "top1_probability",
        "top3_probability",
        "score_p05",
        "score_p95",
    ]
    table = leaderboard[visible_columns].copy()
    formatters = {
        "observed_score": {"type": "progress", "max": 1},
        "top1_probability": {"type": "progress", "max": 1},
        "top3_probability": {"type": "progress", "max": 1},
        "rank_interval_width": {"type": "progress", "max": max(1, float(table["rank_interval_width"].max()))},
    }
    return pn.widgets.Tabulator(
        table,
        height=360,
        pagination="remote",
        page_size=25,
        sorters=[{"field": "observed_rank", "dir": "asc"}],
        formatters=formatters,
    )


def _strength_table(strengths: pd.DataFrame, leaderboard: pd.DataFrame) -> pn.widgets.Tabulator:
    table = strengths.merge(
        leaderboard[
            [
                "model_key",
                "observed_rank",
                "top1_probability",
                "top3_probability",
                "rank_interval_width",
            ]
        ],
        on="model_key",
        how="left",
    )
    return pn.widgets.Tabulator(
        table,
        height=320,
        pagination="remote",
        page_size=25,
        sorters=[{"field": "pairwise_strength", "dir": "desc"}],
        formatters={
            "pairwise_strength": {"type": "progress", "max": 1},
            "top1_probability": {"type": "progress", "max": 1},
            "top3_probability": {"type": "progress", "max": 1},
        },
    )


def _trial_explorer(filtered: pd.DataFrame) -> pn.viewable.Viewable:
    trial_columns = [
        column
        for column in [
            "trial_name",
            "task_name",
            "model_key",
            "source",
            "eval_scope",
            "included_in_score",
            "outcome",
            "score_value",
            "language",
            "repository",
            "cost_usd",
            "n_agent_steps",
            "n_input_tokens",
            "n_output_tokens",
            "peak_context_tokens",
        ]
        if column in filtered.columns
    ]
    try:
        return pn.pane.Perspective(
            filtered[trial_columns],
            columns=trial_columns,
            height=560,
            plugin="datagrid",
            settings=True,
            theme="pro",
        )
    except Exception:
        return pn.widgets.Tabulator(filtered[trial_columns], pagination="remote", page_size=25, height=560)


def _analysis_view(clicks: int) -> pn.Column:
    selection = _selection_for_clicks(clicks)
    filtered = _filtered_trials(selection)
    empty_message = friendly_empty_message(filtered)
    metrics = _metric_strip(filtered, selection)
    if empty_message is not None:
        return pn.Column(metrics, pn.pane.Alert(empty_message, alert_type="warning"))

    try:
        result = bootstrap_rank_stability(filtered, draws=selection.draws, seed=selection.seed)
    except ValueError as exc:
        return pn.Column(metrics, pn.pane.Alert(str(exc), alert_type="danger"))

    leaderboard = result.leaderboard.copy()
    rank_distribution = result.rank_distribution.copy()
    anchor = _run_anchor_strip(leaderboard, selection)
    pairwise_fig, strengths = _pairwise_figure(result.pairwise_win_probability.copy(), leaderboard)
    task_influence = _task_influence_table(filtered, leaderboard)
    swing_tasks = _task_swing_table(filtered)
    language_figure = _language_breakdown(filtered)

    overview = pn.Column(
        metrics,
        anchor,
        pn.pane.Markdown(
            """
### How to read this dashboard

Defaults use score-relevant full DeepSWE rows. SWE-Bench Pro only appears when cross-benchmark sources are enabled.
Rank `1` is best. Pairwise values near `50%` are too close to call.
"""
        ),
        _leaderboard_table(leaderboard),
    )
    who_is_one = _with_anchor(
        anchor,
        pn.pane.Plotly(_rank_interval_figure(leaderboard), config={"responsive": True, "displaylogo": False}),
        pn.Row(
            pn.pane.Plotly(_top_probability_figure(leaderboard), config={"responsive": True, "displaylogo": False}),
            pn.pane.Plotly(_score_stability_figure(leaderboard), config={"responsive": True, "displaylogo": False}),
        ),
    )
    rank_instability = _with_anchor(
        anchor,
        pn.pane.Plotly(_rank_heatmap_figure(rank_distribution, leaderboard), config={"responsive": True, "displaylogo": False})
    )
    pairwise = _with_anchor(
        anchor,
        pn.pane.Plotly(pairwise_fig, config={"responsive": True, "displaylogo": False}),
        pn.pane.Markdown("### Pairwise strength ranking"),
        _strength_table(strengths, leaderboard),
    )
    task_sections: list[pn.viewable.Viewable] = [
        pn.pane.Markdown("### Rank movement tasks"),
        pn.widgets.Tabulator(task_influence, height=320, pagination="remote", page_size=20),
        pn.pane.Markdown("### Score-spread tasks"),
        pn.widgets.Tabulator(swing_tasks, height=360, pagination="remote", page_size=20)
    ]
    if language_figure is not None:
        task_sections.append(pn.pane.Markdown("### Descriptive language score scatter"))
        task_sections.append(pn.pane.Plotly(language_figure, config={"responsive": True, "displaylogo": False}))
    tasks = _with_anchor(anchor, *task_sections)
    explorer = _with_anchor(anchor, _trial_explorer(filtered))
    variance = _with_anchor(anchor, _variance_view(filtered, selection, leaderboard))

    return pn.Column(
        pn.Tabs(
            ("Overview", overview),
            ("Who is #1?", who_is_one),
            ("Rank instability", rank_instability),
            ("Pairwise", pairwise),
            ("Variance", variance),
            ("Tasks", tasks),
            ("Explorer", explorer),
            dynamic=True,
        )
    )


controls = pn.WidgetBox(
    "### Filters",
    include_cross_benchmark,
    source,
    eval_scope,
    included,
    outcome,
    language,
    repository,
    models,
    "### Bootstrap",
    draws,
    seed,
    run,
    pn.pane.Markdown(
        "Filters are pending until **Recompute** is clicked. Enable cross-benchmark sources only when you want SWE-Bench Pro rows."
    ),
    width=350,
)

analysis = pn.panel(
    pn.bind(_analysis_view, run.param.clicks),
    loading_indicator=True,
)

app = pn.template.FastListTemplate(
    title="DeepSWE Rank Stability Lab",
    sidebar=[controls],
    main=[analysis],
    accent_base_color="#2563eb",
    header_background="#111827",
)

app.servable()
