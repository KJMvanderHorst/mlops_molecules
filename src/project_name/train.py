from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.optim import Optimizer
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import NormalizeScale

from data import load_qm9_dataset
from model import GraphNeuralNetwork

if TYPE_CHECKING:
    from torch_geometric.data import Dataset

logger = logging.getLogger(__name__)

# Constants
TARGET_PROPERTY_IDX = 4  # Index of target property in QM9 dataset
LOG_INTERVAL = 10  # Print training stats every N epochs


def train_epoch(
    model: GraphNeuralNetwork,
    loader: DataLoader,
    optimizer: Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch.

    Args:
        model: The GNN model.
        loader: DataLoader for training data.
        optimizer: PyTorch optimizer.
        device: Device to train on (cuda/mps/cpu).

    Returns:
        Average MSE loss for the epoch.
    """
    model.train()
    total_loss: float = 0.0
    num_samples: int = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        pred: torch.Tensor = model(batch)
        target: torch.Tensor = batch.y[:, TARGET_PROPERTY_IDX].unsqueeze(1)

        loss: torch.Tensor = F.mse_loss(pred, target)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        num_samples += batch.num_graphs

    return total_loss / num_samples


@torch.no_grad()
def evaluate(
    model: GraphNeuralNetwork,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Evaluate the model on a dataset.

    Args:
        model: The GNN model.
        loader: DataLoader for validation/test data.
        device: Device to evaluate on.

    Returns:
        Average MSE loss over the dataset.
    """
    model.eval()
    total_loss: float = 0.0
    num_samples: int = 0

    for batch in loader:
        batch = batch.to(device)

        pred: torch.Tensor = model(batch)
        target: torch.Tensor = batch.y[:, TARGET_PROPERTY_IDX].unsqueeze(1)
        loss: torch.Tensor = F.mse_loss(pred, target)
        
        total_loss += loss.item() * batch.num_graphs
        num_samples += batch.num_graphs

    return total_loss / num_samples


def _get_device() -> torch.device:
    """Determine the best available device.

    Returns:
        PyTorch device (cuda/mps/cpu).
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def train(cfg: DictConfig) -> None:
    """Train the GNN model on QM9 dataset.

    Args:
        cfg: Hydra configuration object containing all parameters.
    """
    device: torch.device = _get_device()
    logger.info(f"Using device: {device}")

    model_dir: Path = Path(cfg.training.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading QM9 dataset...")
    dataset: Dataset = load_qm9_dataset(cfg.training.data_path).get_training_dataset()

    # Apply normalization transform
    dataset.transform = NormalizeScale()

    # Split dataset
    n: int = len(dataset)
    train_size: int = int(cfg.training.train_ratio * n)
    val_size: int = int(cfg.training.val_ratio * n)
    test_size: int = n - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(cfg.seed)
    )

    # Create data loaders
    train_loader: DataLoader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True)
    val_loader: DataLoader = DataLoader(val_dataset, batch_size=cfg.training.batch_size, shuffle=False)
    test_loader: DataLoader = DataLoader(test_dataset, batch_size=cfg.training.batch_size, shuffle=False)

    logger.info(f"Dataset split - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Initialize model
    model: GraphNeuralNetwork = GraphNeuralNetwork(
        num_node_features=cfg.model.num_node_features,
        hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers,
        output_dim=cfg.model.output_dim,
    ).to(device)

    optimizer: Optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)

    # Early stopping variables
    best_val_loss: float = float("inf")
    patience: int = cfg.training.patience
    patience_counter: int = 0

    logger.info(f"Starting training for {cfg.training.epochs} epochs...")
    
    # Training loop
    for epoch in range(1, cfg.training.epochs + 1):
        train_loss: float = train_epoch(model, train_loader, optimizer, device)
        val_loss: float = evaluate(model, val_loader, device)

        if epoch % LOG_INTERVAL == 0 or epoch == 1:
            logger.info(f"Epoch {epoch:3d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # Save best model and handle early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_path: Path = model_dir / "best_model.pt"
            torch.save(model.state_dict(), best_model_path)
            logger.debug(f"Saved best model to {best_model_path}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break

    # Load best model and evaluate on test set
    best_model_path = model_dir / "best_model.pt"
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    test_loss: float = evaluate(model, test_loader, device)
    logger.info(f"Final test loss: {test_loss:.6f}")

    # Save final model
    final_model_path: Path = model_dir / "final_model.pt"
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Training complete. Models saved to {model_dir}")


if __name__ == "__main__":
    train()
