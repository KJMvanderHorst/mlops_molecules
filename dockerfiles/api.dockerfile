FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base

EXPOSE $PORT
WORKDIR /app

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY src src
COPY README.md README.md
COPY LICENSE LICENSE

ENV UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked

CMD exec uv run uvicorn project_name.api:app --port ${PORT} --host 0.0.0.0 --workers 1

#ENV PYTHONPATH=/app/src
#ENTRYPOINT ["uv", "run", "uvicorn", "project_name.api:app", "--host", "0.0.0.0", "--port", "$PORT"]
