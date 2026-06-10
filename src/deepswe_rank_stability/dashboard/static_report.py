from __future__ import annotations

import argparse
import html
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from deepswe_rank_stability.analysis.resampling import (
    aggregate_task_model_scores,
    bootstrap_rank_stability,
    filter_trials,
    score_matrix,
)
from deepswe_rank_stability.dashboard.state import (
    plotly_top_first_categoryarray,
    rank_axis_range,
    rank_model_order,
    top_model_summary,
)
from deepswe_rank_stability.data.deepswe import load_dataset


def _plot_html(fig: go.Figure, *, include_plotlyjs: bool = False) -> str:
    return pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs="cdn" if include_plotlyjs else False,
        config={"responsive": True, "displaylogo": False},
    )


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
        title="Bootstrap rank distribution",
        height=max(520, 28 * len(pivot.index)),
        margin={"l": 8, "r": 24, "t": 54, "b": 54},
        xaxis_title="Bootstrap rank, rank 1 is best",
        yaxis_title="",
    )
    fig.update_xaxes(range=[axis_start, axis_end], autorange=False)
    fig.update_yaxes(categoryorder="array", categoryarray=plotly_top_first_categoryarray(order), autorange="reversed")
    return fig


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
        title="Rank intervals",
        height=max(520, 34 * len(plot_frame)),
        margin={"l": 8, "r": 24, "t": 54, "b": 96},
        xaxis_title="Rank, rank 1 is best",
        yaxis_title="",
        legend_orientation="h",
        legend_y=-0.16,
        legend_x=0,
    )
    fig.update_xaxes(range=[axis_start, axis_end], dtick=1, autorange=False)
    fig.update_yaxes(tickmode="array", tickvals=y_positions, ticktext=plot_frame["model_key"], autorange="reversed")
    return fig


def _pairwise_figure(pairwise: pd.DataFrame, leaderboard: pd.DataFrame) -> go.Figure:
    order = rank_model_order(leaderboard)
    matrix = pairwise.reindex(index=order, columns=order)
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
            colorscale=[[0.0, "#b91c1c"], [0.5, "#f8fafc"], [1.0, "#047857"]],
            zmin=0,
            zmax=1,
            colorbar={"title": "Win probability"},
            hovertemplate="%{text}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Pairwise bootstrap win probability",
        height=max(560, 28 * len(matrix.index)),
        margin={"l": 8, "r": 24, "t": 54, "b": 106},
        xaxis_title="Opponent",
        yaxis_title="Model",
    )
    fig.update_xaxes(categoryorder="array", categoryarray=order)
    fig.update_yaxes(categoryorder="array", categoryarray=plotly_top_first_categoryarray(order), autorange="reversed")
    return fig


def _language_figure(filtered: pd.DataFrame) -> go.Figure:
    top_models = (
        filtered.groupby("model_key", observed=False)["score_value"]
        .mean()
        .sort_values(ascending=False)
        .head(8)
        .index
        .tolist()
    )
    summary = (
        filtered[filtered["model_key"].isin(top_models)]
        .groupby(["language", "model_key"], observed=False)
        .agg(score=("score_value", "mean"), trials=("trial_name", "count"))
        .reset_index()
    )
    fig = go.Figure()
    for model_key in top_models:
        frame = summary[summary["model_key"] == model_key]
        fig.add_trace(
            go.Scatter(
                x=frame["language"],
                y=frame["score"],
                mode="lines+markers",
                name=model_key,
                customdata=frame[["trials"]],
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "Language: %{x}<br>"
                    "Mean score: %{y:.3f}<br>"
                    "Trials: %{customdata[0]:,}<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title="Language breakdown, top observed models",
        height=460,
        margin={"l": 8, "r": 24, "t": 54, "b": 80},
        yaxis_title="Mean score",
        xaxis_title="Language",
        legend_title="Model",
    )
    fig.update_yaxes(range=[0, 1])
    return fig


def _close_pairs(pairwise: pd.DataFrame, limit: int = 12) -> pd.DataFrame:
    rows = []
    for left in pairwise.index:
        for right in pairwise.columns:
            if left >= right:
                continue
            win_probability = float(pairwise.loc[left, right])
            closeness = abs(win_probability - 0.5)
            if 0.4 <= win_probability <= 0.6:
                rows.append(
                    {
                        "model_a": left,
                        "model_b": right,
                        "a_beats_b_probability": win_probability,
                        "distance_from_50_50": closeness,
                    }
                )
    return pd.DataFrame(rows).sort_values("distance_from_50_50").head(limit)


def _swing_tasks(filtered: pd.DataFrame, limit: int = 20) -> pd.DataFrame:
    matrix = score_matrix(aggregate_task_model_scores(filtered))
    spread = matrix.max(axis=1, skipna=True) - matrix.min(axis=1, skipna=True)
    coverage = matrix.notna().sum(axis=1)
    task_metadata = (
        filtered[["task_name", "language", "repository", "problem_title"]]
        .drop_duplicates("task_name")
        .set_index("task_name")
    )
    return (
        pd.DataFrame(
            {
                "task_name": spread.index,
                "score_spread": spread.values,
                "models_with_result": coverage.values,
            }
        )
        .join(task_metadata, on="task_name")
        .sort_values(["score_spread", "models_with_result"], ascending=[False, False])
        .head(limit)
    )


def _table_html(frame: pd.DataFrame, *, columns: list[str] | None = None) -> str:
    table = frame if columns is None else frame[columns]
    return table.to_html(index=False, classes="data-table", border=0, escape=True)


def build_report(draws: int, seed: int) -> str:
    dataset = load_dataset()
    filtered = filter_trials(dataset.trials, source="deep-swe", eval_scope="full", included_in_score=True)
    result = bootstrap_rank_stability(filtered, draws=draws, seed=seed)
    leaderboard = result.leaderboard.copy()
    top_model = top_model_summary(leaderboard)
    top_rows = leaderboard[
        [
            "model_key",
            "observed_rank",
            "observed_score",
            "rank_p05",
            "rank_p50",
            "rank_p95",
            "top1_probability",
            "top3_probability",
        ]
    ].head(12)
    close_pairs = _close_pairs(result.pairwise_win_probability)
    swing_tasks = _swing_tasks(filtered)
    findings = [
        (
            f"{top_model['model_key']} is the observed leader and is rank 1 in "
            f"{top_model['top1_probability']:.1%} of bootstrap draws."
        ),
        (
            "The #2 through #6 band is materially less settled: several models have overlapping "
            "p05-p95 rank intervals and trade nearby ranks across resampled task mixes."
        ),
        (
            f"Only {len(close_pairs)} pairwise comparisons land between 40% and 60%, so most model pairs "
            "are well separated under this task-bootstrap view."
        ),
        (
            "Language and repository slices are useful next variance dimensions, but they should be treated "
            "as descriptive until each slice has enough task coverage."
        ),
    ]
    figures = [
        _plot_html(_rank_interval_figure(leaderboard), include_plotlyjs=True),
        _plot_html(_rank_heatmap_figure(result.rank_distribution.copy(), leaderboard)),
        _plot_html(_pairwise_figure(result.pairwise_win_probability.copy(), leaderboard)),
        _plot_html(_language_figure(filtered)),
    ]
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DeepSWE Rank Stability Snapshot</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f8fafc;
      --panel: #fbfdff;
      --ink: #0f172a;
      --muted: #64748b;
      --line: #d9e2ec;
      --accent: #2563eb;
    }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
      line-height: 1.45;
    }}
    main {{
      max-width: 1320px;
      margin: 0 auto;
      padding: 28px 24px 48px;
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 30px;
      letter-spacing: 0;
    }}
    h2 {{
      margin: 34px 0 12px;
      font-size: 20px;
    }}
    p {{
      color: var(--muted);
      max-width: 76ch;
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 10px;
      margin: 22px 0;
    }}
    .chip, section {{
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 8px;
    }}
    .chip {{
      padding: 12px;
    }}
    .label {{
      color: var(--muted);
      font-size: 11px;
      font-weight: 700;
      letter-spacing: .05em;
      text-transform: uppercase;
    }}
    .value {{
      margin-top: 4px;
      font-size: 18px;
      font-weight: 700;
      overflow-wrap: anywhere;
    }}
    section {{
      padding: 16px;
      margin: 16px 0;
    }}
    ul {{
      padding-left: 20px;
      max-width: 90ch;
    }}
    li {{
      margin: 8px 0;
    }}
    .data-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    .data-table th, .data-table td {{
      border-bottom: 1px solid var(--line);
      padding: 8px 10px;
      text-align: left;
      vertical-align: top;
    }}
    .data-table th {{
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: .04em;
    }}
    .note {{
      color: var(--muted);
      font-size: 13px;
    }}
  </style>
</head>
<body>
<main>
  <h1>DeepSWE Rank Stability Snapshot</h1>
  <p>Static export of the task-bootstrap analysis. Defaults: source=deep-swe, eval_scope=full, included_in_score=true.</p>
  <div class="summary">
    <div class="chip"><div class="label">Top model</div><div class="value">{html.escape(str(top_model["model_key"]))}</div></div>
    <div class="chip"><div class="label">Observed score</div><div class="value">{top_model["observed_score"]:.3f}</div></div>
    <div class="chip"><div class="label">Top-1 probability</div><div class="value">{top_model["top1_probability"]:.1%}</div></div>
    <div class="chip"><div class="label">Trials</div><div class="value">{len(filtered):,}</div></div>
    <div class="chip"><div class="label">Tasks</div><div class="value">{filtered["task_name"].nunique():,}</div></div>
    <div class="chip"><div class="label">Draws</div><div class="value">{draws:,}</div></div>
  </div>
  <section>
    <h2>Findings</h2>
    <ul>{"".join(f"<li>{html.escape(finding)}</li>" for finding in findings)}</ul>
    <p class="note">Caveat: task-bootstrap estimates sensitivity to benchmark task mix. It does not cover judge errors, prompt variance, infrastructure effects, or benchmark construction uncertainty.</p>
  </section>
  <section>
    <h2>Leaderboard</h2>
    {_table_html(top_rows)}
  </section>
  <section>{figures[0]}</section>
  <section>{figures[1]}</section>
  <section>{figures[2]}</section>
  <section>
    <h2>Too-close pairwise comparisons</h2>
    {_table_html(close_pairs) if not close_pairs.empty else "<p>No pairwise comparisons fall between 40% and 60%.</p>"}
  </section>
  <section>{figures[3]}</section>
  <section>
    <h2>Swing tasks</h2>
    {_table_html(swing_tasks)}
  </section>
</main>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a static DeepSWE rank-stability report.")
    parser.add_argument("--draws", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("dist/deepswe-rank-stability"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "index.html").write_text(build_report(draws=args.draws, seed=args.seed), encoding="utf-8")
    print(args.out / "index.html")


if __name__ == "__main__":
    main()
