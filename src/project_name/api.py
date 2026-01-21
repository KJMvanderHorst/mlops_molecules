import json
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import torch
from google.cloud import storage
from torch_geometric.data import Data
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, field_validator

from project_name.model import GraphNeuralNetwork


# Request/Response Models
class PredictionRequest(BaseModel):
    """Input for prediction."""

    node_features: list[list[float]]
    edge_index: list[list[int]]

    @field_validator("node_features")
    def validate_node_features(cls, v: list[list[float]]) -> list[list[float]]:
        """Validate node features have correct number of features.

        Args:
            v: Node features matrix.

        Returns:
            Validated node features.

        Raises:
            ValueError: If node features don't have correct number of dimensions.
        """
        if not v:
            raise ValueError("node_features cannot be empty")
        num_features = len(v[0])
        if num_features != 11:
            raise ValueError(f"Each node must have exactly 11 features, got {num_features}")
        if not all(len(features) == num_features for features in v):
            raise ValueError("All nodes must have the same number of features")
        return v


class PredictionResponse(BaseModel):
    """Prediction output."""

    prediction: list[float]


# Inference Service
class InferenceService:
    """Handles model loading and predictions."""

    def __init__(self, model_path: str | Path) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GraphNeuralNetwork(
            num_node_features=11,
            hidden_dim=128,
            num_layers=3,
            output_dim=1,
        )
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def predict(
        self,
        node_features: list[list[float]],
        edge_index: list[list[int]],
    ) -> list[float]:
        """Generate prediction."""
        with torch.no_grad():
            x = torch.tensor(node_features, dtype=torch.float32).to(self.device)
            edge_idx = torch.tensor(edge_index, dtype=torch.long).to(self.device)
            data = Data(x=x, edge_index=edge_idx)
            output = self.model(data)
            return [float(output.squeeze().cpu())]


# Global service instance
service: InferenceService | None = None
MODEL_FOLDER = "/gcs/models/"
PRED_FOLDER = "/gcs/predictions/"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global service
    model_path = "best_model.pt"
    service = InferenceService(MODEL_FOLDER + model_path)

    yield

    del service


# FastAPI App
app = FastAPI(title="Molecule Prediction API", lifespan=lifespan)


# Save prediction results to GCP
def save_prediction_to_gcp(node_features: list[list[float]], edge_index: list[list[int]], outputs: list[float]):
    """Save the prediction results to GCP bucket."""
    client = storage.Client()
    bucket = client.bucket(PRED_FOLDER)
    time = datetime.now(tz=datetime.UTC).isoformat()
    # Prepare prediction data
    data = {
        "node_features": node_features,
        "edge_index": edge_index,
        "prediction": outputs,
        "timestamp": datetime.now(tz=datetime.UTC).isoformat(),
    }
    blob = bucket.blob(f"prediction_{time}.json")
    blob.upload_from_string(json.dumps(data))
    print("Prediction saved to GCP bucket.")


@app.get("/")
def health_check():
    """Health check."""
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest, background_tasks: BackgroundTasks):
    """Generate prediction."""
    if service is None:
        raise HTTPException(status_code=503, detail="Model not ready")

    try:
        prediction = service.predict(
            request.node_features,
            request.edge_index,
        )
        background_tasks.add_task(save_prediction_to_gcp, request.node_features, request.edge_index, prediction)
        return PredictionResponse(prediction=prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
