from __future__ import annotations

from pathlib import Path
from typing import Sequence

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import NormalizeScale

from data import QM9Dataset
from model import GraphNeuralNetwork


@torch.no_grad()
def evaluate(
    model: GraphNeuralNetwork,
    loader: DataLoader,
    device: torch.device,
    target_indices: Sequence[int],
) -> float:
    """Evaluate model on a dataloader.

    Computes mean MSE loss per graph over the entire loader, matching train_epoch.

    Args:
        model: Trained GNN model.
        loader: DataLoader for validation/test set.
        device: Torch device.
        target_indices: Indices of target properties in batch.y.

    Returns:
        Mean MSE loss per graph.
    """
    model.eval()

    total_loss: float = 0.0
    num_samples: int = 0

    target_idx = list(target_indices)

    for batch in loader:
        batch = batch.to(device)

        pred: torch.Tensor = model(batch)
        target: torch.Tensor = batch.y[:, target_idx]

        loss: torch.Tensor = F.mse_loss(pred, target)
        total_loss += loss.item() * batch.num_graphs
        num_samples += batch.num_graphs

    if num_samples == 0:
        return 0.0

    return total_loss / num_samples


@torch.no_grad()
def evaluate_with_metrics(
    model: GraphNeuralNetwork,
    loader: DataLoader,
    device: torch.device,
    target_indices: Sequence[int],
) -> dict[str, float]:
    """Evaluate model on a dataloader with multiple metrics.

    Args:
        model: Trained GNN model.
        loader: DataLoader for validation/test set.
        device: Torch device.
        target_indices: Indices of target properties in batch.y.

    Returns:
        Dictionary with metrics: mse, rmse, mae, r2.
    """
    model.eval()

    all_preds: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    target_idx = list(target_indices)

    for batch in loader:
        batch = batch.to(device)

        pred: torch.Tensor = model(batch)
        target: torch.Tensor = batch.y[:, target_idx]

        all_preds.append(pred)
        all_targets.append(target)

    if len(all_preds) == 0:
        return {"mse": 0.0, "rmse": 0.0, "mae": 0.0, "r2": 0.0}

    predictions = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)

    # MSE
    mse = F.mse_loss(predictions, targets).item()

    # RMSE
    rmse = torch.sqrt(F.mse_loss(predictions, targets)).item()

    # MAE
    mae = F.l1_loss(predictions, targets).item()

    # R² score
    ss_res = torch.sum((targets - predictions) ** 2).item()
    ss_tot = torch.sum((targets - torch.mean(targets)) ** 2).item()
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


def get_device() -> torch.device:
    """Get the best available device for computation."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Load best model and evaluate on test set with comprehensive metrics."""
    device = get_device()

    # Load dataset
    dataset = QM9Dataset(cfg.training.data_path)
    dataset.transform = NormalizeScale()

    n = len(dataset)
    train_size = int(cfg.training.train_ratio * n)
    val_size = int(cfg.training.val_ratio * n)
    test_size = n - train_size - val_size

    _, _, test_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
    )

    target_indices = list(cfg.training.target_indices)
    num_targets = len(target_indices)

    # Build model
    model = GraphNeuralNetwork(
        num_node_features=cfg.model.num_node_features,
        hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers,
        output_dim=num_targets,
    ).to(device)

    # Load best model
    best_model_path = Path(cfg.training.model_dir) / "best_model.pt"
    print(f"Loading model from: {best_model_path}")

    try:
        state = torch.load(best_model_path, weights_only=True)
    except TypeError:
        state = torch.load(best_model_path)

    model.load_state_dict(state)

    # Evaluate with multiple metrics
    metrics = evaluate_with_metrics(model, test_loader, device, target_indices)

    print("\n" + "=" * 50)
    print("Test Set Evaluation (Best Model)")
    print("=" * 50)
    print(f"MSE:  {metrics['mse']:.6f}")
    print(f"RMSE: {metrics['rmse']:.6f}")
    print(f"MAE:  {metrics['mae']:.6f}")
    print(f"R²:   {metrics['r2']:.6f}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
