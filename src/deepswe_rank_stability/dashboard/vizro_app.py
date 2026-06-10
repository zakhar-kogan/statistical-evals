from __future__ import annotations

try:
    from vizro import Vizro
    import vizro.plotly.express as px
    import vizro.models as vm
except ImportError as exc:  # pragma: no cover - exercised by dashboard smoke checks.
    raise SystemExit(
        "Install dashboard dependencies with: uv sync --extra dev --extra dashboard"
    ) from exc

from deepswe_rank_stability.analysis.resampling import bootstrap_rank_stability, filter_trials
from deepswe_rank_stability.data.deepswe import load_dataset


def build_dashboard(draws: int = 500, seed: int = 0) -> vm.Dashboard:
    dataset = load_dataset()
    trials = filter_trials(
        dataset.trials,
        source="deep-swe",
        eval_scope="full",
        included_in_score=True,
    )
    result = bootstrap_rank_stability(trials, draws=draws, seed=seed)
    leaderboard = result.leaderboard
    rank_distribution = result.rank_distribution

    leaderboard_fig = px.bar(
        leaderboard.sort_values("observed_rank"),
        x="observed_score",
        y="model_key",
        color="top1_probability",
        orientation="h",
        title="Observed score with top-1 bootstrap probability",
    )
    rank_fig = px.density_heatmap(
        rank_distribution,
        x="rank",
        y="model_key",
        z="probability",
        histfunc="avg",
        title="Bootstrap rank probability",
    )

    page = vm.Page(
        title="Rank stability spike",
        components=[
            vm.Graph(figure=leaderboard_fig),
            vm.Graph(figure=rank_fig),
        ],
        controls=[
            vm.Filter(column="model_key"),
        ],
    )
    return vm.Dashboard(pages=[page], title="DeepSWE Rank Stability")


def main() -> None:
    Vizro().build(build_dashboard()).run()


if __name__ == "__main__":
    main()
