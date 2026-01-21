"""FastAPI application for drift monitoring with Evidently.

This service compares a reference dataset (typically a slice of the training
data) against the latest prediction logs. It produces an HTML report that
covers input drift, prediction drift, and performance drift (when ground truth
is available).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import anyio
import pandas as pd
from evidently.metric_preset import DataDriftPreset, RegressionPerformancePreset, TargetDriftPreset
from evidently.report import Report
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from google.cloud import storage


REFERENCE_PATH = Path("data/processed/reference.parquet")
BUCKET_ENV = "PREDICTION_BUCKET"
PREFIX_ENV = "PREDICTION_PREFIX"
DEFAULT_PREFIX = "predictions/"
REPORT_PATH = Path("drift_report.html")
TMP_DIR = Path("tmp_predictions")


def _ensure_reference() -> pd.DataFrame:
    """Load the reference dataset for drift comparison."""

    if not REFERENCE_PATH.exists():
        raise FileNotFoundError(
            f"Reference file not found at {REFERENCE_PATH}. Export a training sample before running monitoring."
        )

    if REFERENCE_PATH.suffix == ".parquet":
        return pd.read_parquet(REFERENCE_PATH)
    return pd.read_csv(REFERENCE_PATH)


def _to_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer numeric features per graph for monitoring."""

    records: list[dict[str, Any]] = []
    for row in df.itertuples():
        node_features = getattr(row, "node_features", None)
        edge_index = getattr(row, "edge_index", None)
        if node_features is None or edge_index is None:
            raise ValueError("Both node_features and edge_index must be present in prediction logs.")

        node_df = pd.DataFrame(node_features)
        record: dict[str, Any] = {
            "num_nodes": len(node_df),
            "num_edges": len(edge_index[0]) if len(edge_index) > 0 else 0,
        }

        for col in node_df.columns:
            record[f"node_mean_f{col}"] = float(node_df[col].mean())
            record[f"node_std_f{col}"] = float(node_df[col].std())

        if hasattr(row, "prediction"):
            record["prediction"] = float(getattr(row, "prediction"))
        if hasattr(row, "target") and getattr(row, "target") is not None:
            record["target"] = float(getattr(row, "target"))

        records.append(record)

    return pd.DataFrame.from_records(records)


def download_files(n: int = 5) -> list[Path]:
    """Download the N latest prediction files from GCS."""

    bucket_name = os.environ.get(BUCKET_ENV)
    if bucket_name is None:
        raise HTTPException(status_code=500, detail=f"Environment variable {BUCKET_ENV} is not set.")

    prefix = os.environ.get(PREFIX_ENV) or DEFAULT_PREFIX

    client = storage.Client()
    bucket = client.bucket(str(bucket_name))

    blobs = sorted(
        bucket.list_blobs(prefix=prefix),
        key=lambda b: b.time_created,
        reverse=True,
    )

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for blob in blobs[:n]:
        dest = TMP_DIR / Path(blob.name).name
        blob.download_to_filename(dest)
        paths.append(dest)

    if not paths:
        raise HTTPException(status_code=404, detail="No prediction files found in bucket.")

    return paths


def load_latest_files(directory: Path, n: int = 5) -> pd.DataFrame:
    """Fetch latest prediction data into a single DataFrame."""

    _ = directory  # directory kept for interface parity; downloads go to TMP_DIR
    paths = download_files(n=n)

    frames: list[pd.DataFrame] = []
    for path in paths:
        if path.suffix == ".parquet":
            frames.append(pd.read_parquet(path))
        else:
            frames.append(pd.read_csv(path))

    return pd.concat(frames, ignore_index=True)


def run_analysis(reference_data: pd.DataFrame, current_data: pd.DataFrame) -> None:
    """Run Evidently analysis and save the report."""

    ref_features = _to_features(reference_data)
    cur_features = _to_features(current_data)

    if set(ref_features.columns) != set(cur_features.columns):
        raise HTTPException(
            status_code=500,
            detail="Reference and current features have different columns. Ensure logging schema matches the reference snapshot.",
        )

    metrics = [DataDriftPreset()]

    has_targets = "target" in cur_features.columns and cur_features["target"].notna().any()
    if has_targets:
        metrics.append(RegressionPerformancePreset())
    else:
        metrics.append(TargetDriftPreset(columns=["prediction"]))

    report = Report(metrics=metrics)
    report.run(reference_data=ref_features, current_data=cur_features)
    report.save_html(REPORT_PATH)


def lifespan(app: FastAPI):
    """Load reference data at startup and release on shutdown."""

    del app  # unused
    global reference_data
    reference_data = _ensure_reference()
    yield
    del reference_data


app = FastAPI(title="Molecule Drift Monitor", lifespan=lifespan)


@app.get("/report")
async def get_report(n: int = 5):
    """Generate and return the drift report for the latest N prediction files."""

    try:
        prediction_data = load_latest_files(Path("."), n=n)
        run_analysis(reference_data, prediction_data)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    async with await anyio.open_file(REPORT_PATH, encoding="utf-8") as f:
        html_content = await f.read()

    return HTMLResponse(content=html_content, status_code=200)
