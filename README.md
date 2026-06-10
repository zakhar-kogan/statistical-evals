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

Exploratory tools for downloading DeepSWE benchmark artifacts, bootstrapping task-level rankings, and trying small dashboard prototypes.

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
```

Artifacts are cached under `.cache/deepswe_rank_stability/` by default.

## Dashboards

Panel dashboard:

```bash
uv run panel serve src/deepswe_rank_stability/dashboard/panel_app.py --show
```

The Panel dashboard is the main app path. It is built for filter-heavy exploratory analysis, bootstrap reruns, and trial/task inspection.

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

On each push to `main`, GitHub Actions runs `.github/workflows/sync-to-hub.yml` and syncs the repo to the Space using `huggingface/hub-sync`.

You can also trigger the workflow manually from the GitHub Actions UI.

### 4. Runtime

The Space runs the Panel app from this repo with:

```bash
uv run panel serve src/deepswe_rank_stability/dashboard/panel_app.py --address 0.0.0.0 --port 7860 --use-xheaders
```

The container image is defined in [`Dockerfile`](/Users/zakhar/Documents/statistical-evals/Dockerfile).
