from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from deepswe_rank_stability.analysis.resampling import (
    aggregate_task_model_scores,
    bootstrap_by_dimension,
    bootstrap_rank_stability,
    cost_diagnostics_by_dimension,
    filter_trials,
    model_task_coverage_by_dimension,
    observed_leaderboard,
    rank_scores,
    score_matrix,
    swing_tasks_by_dimension,
    task_influence_table,
)
from deepswe_rank_stability.dashboard.state import friendly_empty_message
from deepswe_rank_stability.dashboard.state import (
    DashboardSelection,
    contender_model_order,
    eligible_variance_dimensions,
    order_pairwise_by_strength,
    pairwise_strength,
    plotly_top_first_categoryarray,
    rank_axis_range,
    rank_model_order,
    slice_values_with_summaries,
    source_options,
    submit_selection,
    variance_empty_message,
)


def sample_trials() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "trial_name": "a-1",
                "task_name": "task-1",
                "model_key": "model-a [high]",
                "source": "deep-swe",
                "eval_scope": "full",
                "included_in_score": True,
                "outcome": "pass",
                "language": "python",
                "repository": "repo/a",
                "score_value": 1.0,
                "passed": True,
            },
            {
                "trial_name": "a-1-repeat",
                "task_name": "task-1",
                "model_key": "model-a [high]",
                "source": "deep-swe",
                "eval_scope": "full",
                "included_in_score": True,
                "outcome": "fail",
                "language": "python",
                "repository": "repo/a",
                "score_value": 0.0,
                "passed": False,
            },
            {
                "trial_name": "b-1",
                "task_name": "task-1",
                "model_key": "model-b",
                "source": "deep-swe",
                "eval_scope": "full",
                "included_in_score": True,
                "outcome": "pass",
                "language": "python",
                "repository": "repo/a",
                "score_value": 1.0,
                "passed": True,
            },
            {
                "trial_name": "a-2",
                "task_name": "task-2",
                "model_key": "model-a [high]",
                "source": "deep-swe",
                "eval_scope": "full",
                "included_in_score": True,
                "outcome": "fail",
                "language": "go",
                "repository": "repo/b",
                "score_value": 0.0,
                "passed": False,
            },
            {
                "trial_name": "b-2",
                "task_name": "task-2",
                "model_key": "model-b",
                "source": "deep-swe",
                "eval_scope": "full",
                "included_in_score": True,
                "outcome": "pass",
                "language": "go",
                "repository": "repo/b",
                "score_value": 1.0,
                "passed": True,
            },
            {
                "trial_name": "excluded",
                "task_name": "task-2",
                "model_key": "model-c",
                "source": "deep-swe",
                "eval_scope": "full",
                "included_in_score": False,
                "outcome": "pass",
                "language": "go",
                "repository": "repo/b",
                "score_value": 1.0,
                "passed": True,
            },
        ]
    )


def test_filter_trials_applies_default_dashboard_filters() -> None:
    filtered = filter_trials(
        sample_trials(),
        source="deep-swe",
        eval_scope="full",
        included_in_score=True,
        language="python",
    )

    assert set(filtered["trial_name"]) == {"a-1", "a-1-repeat", "b-1"}


def test_aggregate_task_model_scores_uses_cell_mean_for_repeats() -> None:
    aggregated = aggregate_task_model_scores(sample_trials())
    cell = aggregated[
        (aggregated["task_name"] == "task-1") & (aggregated["model_key"] == "model-a [high]")
    ].iloc[0]

    assert cell["score_value"] == 0.5
    assert cell["pass_rate"] == 0.5
    assert cell["n_trials"] == 2


def test_rank_scores_uses_min_rank_for_ties() -> None:
    ranks = rank_scores(pd.Series({"a": 1.0, "b": 1.0, "c": 0.0}))

    assert ranks.to_dict() == {"a": 1, "b": 1, "c": 3}


def test_observed_leaderboard_handles_ties() -> None:
    matrix = pd.DataFrame({"a": [1.0, 0.0], "b": [0.0, 1.0], "c": [0.0, 0.0]})
    leaderboard = observed_leaderboard(matrix)

    ranks = dict(zip(leaderboard["model_key"], leaderboard["observed_rank"], strict=False))
    assert ranks == {"a": 1, "b": 1, "c": 3}


def test_bootstrap_rank_stability_is_deterministic_with_seed() -> None:
    trials = filter_trials(sample_trials(), included_in_score=True)

    left = bootstrap_rank_stability(trials, draws=50, seed=7)
    right = bootstrap_rank_stability(trials, draws=50, seed=7)

    pd.testing.assert_frame_equal(left.leaderboard, right.leaderboard)
    pd.testing.assert_frame_equal(left.rank_distribution, right.rank_distribution)
    pd.testing.assert_frame_equal(left.pairwise_win_probability, right.pairwise_win_probability)


def test_bootstrap_by_dimension_is_deterministic_with_seed() -> None:
    trials = filter_trials(sample_trials(), included_in_score=True)

    left = bootstrap_by_dimension(trials, dimension="language", draws=30, seed=11, min_tasks=1, min_models=2)
    right = bootstrap_by_dimension(trials, dimension="language", draws=30, seed=11, min_tasks=1, min_models=2)

    pd.testing.assert_frame_equal(left.summaries, right.summaries)
    pd.testing.assert_frame_equal(left.skipped_slices, right.skipped_slices)


def test_bootstrap_by_dimension_skips_underpowered_slices() -> None:
    trials = filter_trials(sample_trials(), included_in_score=True)

    result = bootstrap_by_dimension(trials, dimension="language", draws=10, seed=1, min_tasks=2, min_models=2)

    assert result.summaries.empty
    assert set(result.skipped_slices["slice_value"]) == {"go", "python"}
    assert set(result.skipped_slices["reason"]) == {"fewer than 2 tasks"}


def test_bootstrap_by_dimension_summarizes_language_slices() -> None:
    trials = filter_trials(sample_trials(), included_in_score=True)

    result = bootstrap_by_dimension(trials, dimension="language", draws=30, seed=2, min_tasks=1, min_models=2)

    assert set(result.summaries["slice_value"]) == {"go", "python"}
    assert {"dimension", "slice_value", "n_trials", "n_tasks", "n_models", "model_key"}.issubset(
        result.summaries.columns
    )
    assert result.skipped_slices.empty


def test_bootstrap_by_dimension_keeps_skipped_language_diagnostics() -> None:
    trials = pd.concat(
        [
            filter_trials(sample_trials(), included_in_score=True),
            pd.DataFrame(
                [
                    {
                        "trial_name": "js-1",
                        "task_name": "task-js",
                        "model_key": "model-a [high]",
                        "source": "deep-swe",
                        "eval_scope": "full",
                        "included_in_score": True,
                        "outcome": "pass",
                        "language": "javascript",
                        "repository": "repo/js",
                        "score_value": 1.0,
                        "passed": True,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    result = bootstrap_by_dimension(trials, dimension="language", draws=10, seed=1, min_tasks=2, min_models=2)

    skipped = result.skipped_slices.set_index("slice_value")
    assert skipped.loc["javascript", "n_tasks"] == 1
    assert skipped.loc["javascript", "reason"] == "fewer than 2 tasks"


def test_score_matrix_excludes_missing_cells_from_model_mean() -> None:
    aggregated = aggregate_task_model_scores(sample_trials())
    matrix = score_matrix(aggregated)

    assert pd.isna(matrix.loc["task-1", "model-c"])
    assert matrix["model-c"].mean(skipna=True) == 1.0


def test_model_task_coverage_by_dimension_handles_sparse_matrix() -> None:
    coverage = model_task_coverage_by_dimension(sample_trials(), dimension="language")
    model_c_go = coverage[(coverage["slice_value"] == "go") & (coverage["model_key"] == "model-c")].iloc[0]

    assert model_c_go["observed_tasks"] == 1
    assert model_c_go["n_tasks"] == 1
    assert model_c_go["coverage"] == 1.0


def test_cost_diagnostics_by_dimension_tolerates_absent_optional_columns() -> None:
    diagnostics = cost_diagnostics_by_dimension(sample_trials(), dimension="language")

    assert diagnostics.columns.tolist() == ["dimension", "slice_value", "model_key", "n_trials"]
    assert diagnostics.empty


def test_swing_tasks_by_dimension_groups_by_slice() -> None:
    swing_tasks = swing_tasks_by_dimension(sample_trials(), dimension="language", limit=10)

    assert {"dimension", "slice_value", "task_name", "score_spread", "models_with_result"}.issubset(
        swing_tasks.columns
    )
    assert set(swing_tasks["slice_value"]) == {"go", "python"}


def test_task_influence_table_ranks_high_spread_tasks_first_with_metadata() -> None:
    influence = task_influence_table(
        filter_trials(sample_trials(), included_in_score=True),
        contender_models=["model-a [high]", "model-b"],
        limit=5,
    )

    assert influence.iloc[0]["contender_score_spread"] >= influence.iloc[-1]["contender_score_spread"]
    assert {"language", "repository", "best_contender", "worst_contender"}.issubset(influence.columns)


def test_bootstrap_missing_cells_do_not_emit_runtime_warnings() -> None:
    trials = filter_trials(sample_trials(), included_in_score=None)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = bootstrap_rank_stability(trials, draws=50, seed=3)

    runtime_warnings = [warning for warning in caught if issubclass(warning.category, RuntimeWarning)]
    assert runtime_warnings == []
    assert np.isnan(result.boot_scores["model-c"]).any()


def test_leaderboard_includes_rank_interval_fields() -> None:
    result = bootstrap_rank_stability(filter_trials(sample_trials(), included_in_score=True), draws=50, seed=4)

    expected = {"rank_p05", "rank_p25", "rank_p50", "rank_p75", "rank_p95", "rank_interval_width"}
    assert expected.issubset(result.leaderboard.columns)
    assert (result.leaderboard["rank_interval_width"] >= 0).all()


def test_pairwise_matrix_shape_and_diagonal() -> None:
    result = bootstrap_rank_stability(filter_trials(sample_trials(), included_in_score=True), draws=50, seed=5)
    pairwise = result.pairwise_win_probability

    assert pairwise.shape == (2, 2)
    assert list(pairwise.index) == list(pairwise.columns)
    assert np.diag(pairwise).tolist() == [1.0, 1.0]


def test_friendly_empty_dashboard_state_helper() -> None:
    assert "No trial rows" in friendly_empty_message(sample_trials().iloc[0:0])

    one_model = sample_trials()[sample_trials()["model_key"] == "model-b"]
    assert "at least two" in friendly_empty_message(one_model)

    assert friendly_empty_message(filter_trials(sample_trials(), included_in_score=True)) is None


def test_eligible_variance_dimensions_hide_one_value_source_and_scope() -> None:
    trials = filter_trials(sample_trials(), included_in_score=True)

    assert eligible_variance_dimensions(trials, min_tasks=1, min_models=2) == ["language"]


def test_eligible_variance_dimensions_excludes_repository() -> None:
    trials = filter_trials(sample_trials(), included_in_score=True)

    dimensions = eligible_variance_dimensions(
        trials,
        candidates=("language", "repository", "source", "eval_scope"),
        min_tasks=1,
        min_models=2,
    )

    assert "repository" not in dimensions
    assert dimensions == ["language"]


def test_eligible_variance_dimensions_keeps_language_with_eligible_slice() -> None:
    trials = filter_trials(sample_trials(), included_in_score=True)

    dimensions = eligible_variance_dimensions(trials, min_tasks=1, min_models=2)

    assert "language" in dimensions


def test_variance_empty_message_reports_no_eligible_dimensions() -> None:
    assert "No bootstrap variance dimensions" in variance_empty_message([])
    assert variance_empty_message(["language"]) is None


def test_rank_ordering_places_observed_rank_one_first() -> None:
    leaderboard = pd.DataFrame(
        {
            "model_key": ["b", "a", "c"],
            "observed_rank": [2, 1, 3],
            "observed_score": [0.7, 0.8, 0.1],
            "rank_p95": [3, 2, 3],
        }
    )

    assert rank_model_order(leaderboard) == ["a", "b", "c"]


def test_contender_model_order_returns_top_ten_plus_nonzero_top3() -> None:
    leaderboard = pd.DataFrame(
        {
            "model_key": [f"m{i}" for i in range(1, 13)],
            "observed_rank": list(range(1, 13)),
            "observed_score": [1 / i for i in range(1, 13)],
            "top3_probability": [1.0, 0.8, 0.3, 0.1, 0.01, 0, 0, 0, 0, 0, 0.2, 0],
        }
    )

    assert contender_model_order(leaderboard, top_n=5) == ["m1", "m2", "m3", "m4", "m5", "m11"]


def test_slice_values_with_summaries_orders_by_task_count() -> None:
    summary = pd.DataFrame(
        {
            "slice_value": ["python", "go", "typescript"],
            "n_tasks": [34, 34, 35],
        }
    )

    assert slice_values_with_summaries(summary) == ["typescript", "go", "python"]


def test_plotly_top_first_categoryarray_preserves_data_order() -> None:
    order = ["a", "b", "c"]

    assert plotly_top_first_categoryarray(order) == order


def test_rank_axis_range_starts_at_one_and_is_not_reversed() -> None:
    leaderboard = pd.DataFrame({"observed_rank": [1, 2, 3], "rank_p95": [1.0, 2.5, 3.0]})

    axis_start, axis_end = rank_axis_range(leaderboard)

    assert axis_start == 0.5
    assert axis_end > axis_start


def test_pairwise_strength_excludes_diagonal_and_sorts_descending() -> None:
    pairwise = pd.DataFrame(
        [[1.0, 0.9, 0.8], [0.1, 1.0, 0.7], [0.2, 0.3, 1.0]],
        index=["a", "b", "c"],
        columns=["a", "b", "c"],
    )

    strengths = pairwise_strength(pairwise)
    ordered, ordered_strengths = order_pairwise_by_strength(pairwise)

    assert strengths["model_key"].tolist() == ["a", "b", "c"]
    assert strengths["pairwise_strength"].round(2).tolist() == [0.85, 0.40, 0.25]
    assert ordered.index.tolist() == ["a", "b", "c"]
    assert ordered.columns.tolist() == ["a", "b", "c"]
    pd.testing.assert_frame_equal(strengths, ordered_strengths)


def test_source_options_hide_swebenchpro_until_cross_benchmark_enabled() -> None:
    all_sources = ["deep-swe", "swebenchpro"]

    assert source_options(all_sources, include_cross_benchmark=False) == ["All", "deep-swe"]
    assert source_options(all_sources, include_cross_benchmark=True) == ["All", "deep-swe", "swebenchpro"]


def test_submit_selection_changes_only_when_trigger_changes() -> None:
    first = DashboardSelection(
        source="deep-swe",
        eval_scope="full",
        included_in_score=True,
        outcome="All",
        language="All",
        repository="All",
        model_keys=(),
        draws=500,
        seed=1,
    )
    second = DashboardSelection(
        source="deep-swe",
        eval_scope="full",
        included_in_score=True,
        outcome="All",
        language="All",
        repository="All",
        model_keys=(),
        draws=2_000,
        seed=2,
    )

    submitted, last_clicks = submit_selection(
        current=None,
        trigger_count=0,
        last_trigger_count=-1,
        pending=first,
    )
    unchanged, unchanged_clicks = submit_selection(
        current=submitted,
        trigger_count=0,
        last_trigger_count=last_clicks,
        pending=second,
    )
    changed, changed_clicks = submit_selection(
        current=unchanged,
        trigger_count=1,
        last_trigger_count=unchanged_clicks,
        pending=second,
    )

    assert unchanged == first
    assert unchanged_clicks == 0
    assert changed == second
    assert changed_clicks == 1
