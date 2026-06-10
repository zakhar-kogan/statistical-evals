# AGENTS.md

## Project
This repo is an exploratory benchmark rank-stability analysis lab. Keep changes small, inspectable, and grounded in the existing DataFrame-first design.

## Commands
- Test: `uv run --extra dev pytest`
- Dashboard: `uv run --extra dashboard panel serve src/deepswe_rank_stability/dashboard/panel_app.py --show`
- Static report: `uv run --extra dashboard python -m deepswe_rank_stability.dashboard.static_report`

## Architecture
- `data/` downloads, normalizes, and registers eval datasets.
- `analysis/` contains reusable statistical methods and must not import dashboard libraries.
- `dashboard/` contains Panel UI and Plotly/Perspective presentation.
- `cli.py` is a thin entrypoint over data and analysis functions.

## Methodology
- Normalize eval rows to `eval_id`, `trial_id`, `task_id`, `system_id`, and a selected ranking metric.
- Preserve legacy aliases (`trial_name`, `task_name`, `model_key`, `score_value`) while DeepSWE-specific code is migrated.
- Repeated `(task, system, source, eval_scope)` cells are mean-aggregated before bootstrap.
- Bootstrap resamples tasks with replacement.
- Missing model-task cells are excluded from that model's mean.

## Deployment
- Hugging Face Space deploys from GitHub via `.github/workflows/sync-to-hub.yml`.
- Docker Space listens on port `7860`.
- Keep hosted defaults lightweight; local analysis can use larger draw counts.

## Style
- Keep analysis functions deterministic with explicit seeds.
- Add tests for ranking, tie handling, filtering, adapters, metrics, and bootstrap output changes.
- Do not put plotting logic into `analysis/`.
