import os
from typing import Any, Dict

from invoke import Context, task
from omegaconf import OmegaConf

WINDOWS = os.name == "nt"
PROJECT_NAME = "project_name"
PYTHON_VERSION = "3.12"


def _load_loadtest_config(config_path: str) -> Dict[str, Any]:
    cfg = OmegaConf.load(config_path)
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


# Project commands
@task
def preprocess_data(ctx: Context) -> None:
    """Preprocess data."""
    ctx.run(f"uv run src/{PROJECT_NAME}/data.py", echo=True, pty=not WINDOWS)


@task
def train(ctx: Context) -> None:
    """Train model."""
    ctx.run(
        f"uv run src/{PROJECT_NAME}/train.py",
        echo=True,
        pty=not WINDOWS,
        env={"HYDRA_FULL_ERROR": "1"},
    )


@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("uv run coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("uv run coverage report -m -i", echo=True, pty=not WINDOWS)


@task
def train_docker(ctx: Context, verbose=False) -> None:
    """Train model."""
    models_path = "$(pwd)/models:/models/"
    if verbose:
        ctx.run(f"docker run --rm -t --name train -v {models_path} train:latest", echo=True, pty=not WINDOWS)
    else:
        ctx.run(f"docker run --rm -v {models_path} train:latest", echo=True, pty=not WINDOWS)


@task
def docker_build(ctx: Context, progress: str = "plain") -> None:
    """Build docker images."""
    ctx.run(
        f"docker build -t train:latest . -f dockerfiles/train.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )


#    ctx.run(
#        f"docker build -t api:latest . -f dockerfiles/api.dockerfile --progress={progress}",
#        echo=True,
#        pty=not WINDOWS
#    )


# Documentation commands
@task
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("uv run mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)


@task
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("uv run mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)


@task
def loadtest(ctx: Context, config_path: str = "configs/loadtest/loadtest.yaml", host: str | None = None) -> None:
    """Run Locust load test using parameters from config and optional host override."""

    cfg = _load_loadtest_config(config_path)
    resolved_host = host or os.getenv("LOCUST_HOST") or cfg.get("host")
    if not resolved_host:
        raise ValueError("Host is required; provide host, LOCUST_HOST, or config host.")

    locust_file = cfg.get("locust_file", "tests/performance_tests/load_test_api.py")
    users = cfg.get("users", 50)
    spawn_rate = cfg.get("spawn_rate", 5)
    run_time = cfg.get("run_time", "2m")
    csv_prefix = cfg.get("csv_prefix", "locust-results/results")
    extra_args = cfg.get("extra_args", "")

    output_dir = os.path.dirname(csv_prefix)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    cmd = (
        f"uv run locust -f {locust_file} "
        f"--headless -u {users} -r {spawn_rate} --run-time {run_time} "
        f'--host="{resolved_host}" --csv={csv_prefix} {extra_args}'
    ).strip()

    ctx.run(cmd, echo=True, pty=not WINDOWS)
