# api, worker-io and worker-index share this image; compose picks the command.
FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
 && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /usr/local/bin/uv

# The colab CLI drives indexing on Colab GPUs; pinned, since its output is parsed.
RUN UV_TOOL_DIR=/opt/uv-tools UV_TOOL_BIN_DIR=/usr/local/bin \
    uv tool install --python /usr/local/bin/python3 google-colab-cli==0.7.4

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv PATH=/opt/venv/bin:$PATH
WORKDIR /app

# Dependencies first, for layer caching.
COPY pyproject.toml uv.lock ./
COPY server/pyproject.toml server/pyproject.toml
COPY indexer/pyproject.toml indexer/pyproject.toml
RUN uv sync --frozen --no-dev --all-packages --all-extras --no-install-workspace --python /usr/local/bin/python3

COPY server server
COPY indexer indexer
RUN uv sync --frozen --no-dev --all-packages --all-extras --no-editable --python /usr/local/bin/python3

# ffmpeg and the workers run unprivileged.
RUN useradd --uid 1000 --create-home fh && mkdir -p /data && chown fh:fh /data
USER fh
ENV DATA_DIR=/data FH_MODELS_DIR=/data/models HF_HOME=/data/models HOST=0.0.0.0
VOLUME /data
EXPOSE 8000
CMD ["uvicorn", "framehound.api:app", "--host", "0.0.0.0", "--port", "8000"]
