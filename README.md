---
title: DeepSWE Rank Stability Lab
emoji: "📊"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
fullWidth: true
---

# DeepSWE Rank-Stability Side Experiment

Exploratory tools for downloading benchmark artifacts, bootstrapping task-level rankings, and comparing rank stability across eval datasets. DeepSWE is the default eval; SWE-Bench Pro rows from the same artifact source are available as a second eval.

## Setup

```bash
uv sync --extra dev
```

Dashboard dependencies are heavier:

```bash
uv sync --extra dev --extra dashboard
```

## Run The Analysis

```bash
uv run python -m deepswe_rank_stability.cli summarize
uv run python -m deepswe_rank_stability.cli bootstrap --draws 500
uv run python -m deepswe_rank_stability.cli summarize --eval swebench_pro
uv run python -m deepswe_rank_stability.cli bootstrap --eval swebench_pro --draws 500
```

Artifacts are cached under `.cache/deepswe_rank_stability/` by default.

τ-Bench / τ² current exports can be analyzed by pointing `TAU_BENCH_RESULTS` at a
monolithic `results.json`, a directory-format result folder with `simulations/`,
or a flat JSON/JSONL/CSV export:

```bash
TAU_BENCH_RESULTS=/path/to/results.json uv run python -m deepswe_rank_stability.cli summarize --eval tau_bench
TAU_BENCH_RESULTS=/path/to/results.json uv run python -m deepswe_rank_stability.cli bootstrap --eval tau_bench --draws 500
```

The reusable analysis functions operate on normalized result tables with `eval_id`, `trial_id`, `task_id`, `system_id`, and an explicit ranking metric column. DeepSWE aliases such as `trial_name`, `task_name`, `model_key`, and `score_value` are still preserved during the migration.

## Dashboards

Panel dashboard:

```bash
uv run panel serve src/deepswe_rank_stability/dashboard/panel_app.py --show
```

The Panel dashboard is the main app path. It has an eval selector, metric selector, eval-specific filters, bootstrap reruns, and trial/task inspection.

## Deploy To Hugging Face Spaces

This repo is set up to deploy as a Docker Space from GitHub. GitHub stays the source of truth; Hugging Face mirrors `main`.

### 1. Create the Space

Create a new Hugging Face Space and choose **Docker** as the SDK.

### 2. Add GitHub configuration

Add these in the GitHub repo settings:

- Secret: `HF_TOKEN`
- Variable: `HF_SPACE_REPO_ID`

`HF_SPACE_REPO_ID` should be the full Hub repo id, for example `your-hf-handle/deepswe-rank-stability`.

The token needs write access to Spaces.

### 3. Push to `main`

On each push to `main` or `master`, GitHub Actions runs `.github/workflows/sync-to-hub.yml` and syncs the repo to the Space using `huggingface/hub-sync`.

You can also trigger the workflow manually from the GitHub Actions UI.

### 4. Runtime

The Space runs the Panel app from this repo with:

```bash
uv run panel serve src/deepswe_rank_stability/dashboard/panel_app.py --address 0.0.0.0 --port 7860 --use-xheaders
```

The container image is defined in [`Dockerfile`](/Users/zakhar/Documents/statistical-evals/Dockerfile).
