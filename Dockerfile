FROM python:3.11-slim

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    DEEPSWE_DEFAULT_DRAWS=500

RUN useradd -m -u 1000 user

USER user
WORKDIR $HOME/app

RUN pip install --no-cache-dir --upgrade pip uv

COPY --chown=user pyproject.toml uv.lock README.md ./
COPY --chown=user src ./src

RUN uv sync --frozen --extra dashboard --no-dev
RUN uv run python -c "from deepswe_rank_stability.data.deepswe import load_dataset; load_dataset()"

CMD ["uv", "run", "panel", "serve", "src/deepswe_rank_stability/dashboard/panel_app.py", "--address", "0.0.0.0", "--port", "7860", "--use-xheaders", "--allow-websocket-origin", "zakhar-kogan-deepswe-rank-stability.hf.space"]
