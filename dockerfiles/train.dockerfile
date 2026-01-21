FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean &&  rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY README.md README.md
COPY LICENSE LICENSE
COPY src/ src/
COPY configs/ configs/

ENV UV_LINK_MODE=copy
RUN uv sync --dev --locked --no-install-project

# Use mounted GCS filesystem for data access
# Data is mounted at /gcs/<bucket-name> by Vertex AI
ENTRYPOINT ["uv", "run", "src/project_name/train.py"]
