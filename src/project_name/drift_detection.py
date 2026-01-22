import json
from pathlib import Path

import anyio
import pandas as pd
import numpy as np
from evidently.legacy.metric_preset import TargetDriftPreset, DataDriftPreset
from evidently.legacy.report import Report
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from google.cloud import storage

from project_name.data import QM9Dataset

BUCKET_NAME = "molecules_bucket"
DATA_PATH = "gcs/data"


def run_analysis(reference_data: pd.DataFrame, current_data: pd.DataFrame) -> None:
    """Run the analysis and return the report."""
    report = Report(metrics=[DataDriftPreset(), TargetDriftPreset(columns=["prediction"])])
    report.run(reference_data=reference_data, current_data=current_data)
    report.save_html("report.html")


def lifespan(app: FastAPI):
    """Load the data and class names before the application starts."""
    global train_molecule_df, train_atoms_df

    dataset = QM9Dataset(DATA_PATH)
    nodes_features, edge_index, prediction = [], [], []
    for i, example in enumerate(dataset):
        nodes_features.append(example.x.tolist())
        edge_index.append(example.edge_index.tolist())
        prediction.append(example.y[:, 4].tolist())
        if i >= 1000:  # takes too much time to load all data
            break
    train_molecule_df, train_atoms_df = data2frames(nodes_features, edge_index, prediction)
    yield

    del train_molecule_df, train_atoms_df


app = FastAPI(lifespan=lifespan)


def load_latest_files(directory: Path, n: int) -> pd.DataFrame:
    """Load the N latest prediction files from the directory."""
    # Download the latest prediction files from the GCP bucket
    download_files(n=n)

    # Get all prediction files in the directory
    files = directory.glob("*.json")

    # Load or process the files as needed
    nodes_features, edge_index, prediction = [], [], []
    for file in files:
        with file.open() as f:
            data = json.load(f)
            nodes_features.append(data["node_features"])
            edge_index.append(data["edge_index"])
            prediction.append(data["prediction"])
    molecule_df, atoms_df = data2frames(nodes_features, edge_index, prediction)
    return molecule_df, atoms_df


def download_files(n: int = 5, output_dir: str = "predictions") -> None:
    """Download the N latest prediction files from the GCP bucket.

    Args:
        n: Number of latest files to download.
        output_dir: Local directory to save files to.
    """
    bucket = storage.Client().bucket(BUCKET_NAME)
    blobs = list(bucket.list_blobs(prefix="predictions/"))
    blobs.sort(key=lambda x: x.updated, reverse=True)
    latest_blobs = blobs[:n]

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for blob in latest_blobs:
        filename = Path(blob.name).name
        local_path = Path(output_dir) / filename
        blob.download_to_filename(str(local_path))


def data2frames(
    node_features: list[list[list[float]]],
    edge_index: list[list[list[int]]],
    predictions: list[list[float]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert raw data to molecule and atom DataFrames.

    Args:
        node_features: List of node feature matrices, each shape (N, 11).
        edge_index: List of edge index arrays, each shape (2, num_edges).
        predictions: List of prediction values.

    Returns:
        Tuple of (molecule_df, atoms_df) DataFrames.
    """
    molecule_data = []
    atom_data = []

    for nodes, edges, preds in zip(node_features, edge_index, predictions):
        # Calculate average node features
        nodes_array = np.array(nodes)  # Shape: (N, 11)
        avg_features = nodes_array.mean(axis=0)

        # Get number of nodes and edges
        num_nodes = len(nodes)
        num_edges = len(edges[0])  # edges is (2, num_edges)

        # Create molecule data
        mol_dict = {
            "prediction": preds[0],
            "num_nodes": num_nodes,
            "num_edges": num_edges,
        }
        # Add average features
        for i, avg_feat in enumerate(avg_features):
            mol_dict[f"avg_feature_{i}"] = avg_feat

        molecule_data.append(mol_dict)

        # Add atom data
        for features in nodes:
            atom_data.append({**{f"feature_{i}": feat for i, feat in enumerate(features)}})

    molecule_df = pd.DataFrame(molecule_data)
    atoms_df = pd.DataFrame(atom_data)

    return molecule_df, atoms_df


@app.get("/")
def health_check():
    """Health check."""
    return {"status": "healthy"}


@app.get("/report", response_class=HTMLResponse)
async def get_report(n: int = 5):
    """Generate and return the report."""
    molecule_df, atoms_df = load_latest_files(Path("predictions"), n=n)
    run_analysis(train_molecule_df, molecule_df)

    async with await anyio.open_file("report.html", encoding="utf-8") as f:
        html_content = await f.read()

    return HTMLResponse(content=html_content, status_code=200)
