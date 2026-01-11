FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean &&  rm -rf /var/lib/apt/lists/*

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY src/ src/
COPY data/ data/

WORKDIR /
ENV UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv uv sync --dev --locked --no-install-project

ENTRYPOINT ["uv", "run", "--no-project", "src/project_name/train.py"]
