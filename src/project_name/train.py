from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
import torch
import torch.nn.functional as F
import wandb

from omegaconf import DictConfig, OmegaConf
from torch.optim import Optimizer
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import NormalizeScale

from data import QM9Dataset
from evaluate import evaluate, evaluate_with_metrics
from model import GraphNeuralNetwork
from profiling import TrainingProfiler, timing_checkpoint
from utils import get_data_path

import multiprocessing
from torch import nn

def _num_workers(cfg: DictConfig) -> int:
    cores = multiprocessing.cpu_count()
    return min(4, cores)

if TYPE_CHECKING:
    from torch_geometric.data import Dataset

logger = logging.getLogger(__name__)

# Constants
LOG_INTERVAL = 10

# Determine config path - works both locally and in container
_CONFIG_PATH = str(Path(__file__).parent.parent.parent / "configs")


def _init_wandb(cfg: DictConfig) -> wandb.run.Run | None:
    """Initialize wandb run if enabled in config.

    Assumes authentication is handled externally via:
        wandb login
    """
    enabled = bool(OmegaConf.select(cfg, "wandb.enable", default=True))
    if not enabled:
        os.environ["WANDB_MODE"] = "disabled"
        logger.info("wandb logging disabled by config.")
        return None

    try:
        run = wandb.init(
            project=cfg.wandb.get("project", "mlops-molecules"),
            config=OmegaConf.to_container(cfg, resolve=True),
            settings=wandb.Settings(git_root=None),  # Disable git detection for containerized environments
        )
        logger.info("Initialized wandb run: %s", run.id)
        return run
    except Exception as e:
        logger.warning("Failed to initialize wandb: %s", e, exc_info=True)
        os.environ["WANDB_MODE"] = "disabled"
        return None


def train_epoch(
    model: GraphNeuralNetwork,
    loader: DataLoader,
    optimizer: Optimizer,
    device: torch.device,
    target_indices: list[int],
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss: float = 0.0
    num_samples: int = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)

        pred: torch.Tensor = model(batch)
        target: torch.Tensor = batch.y[:, target_indices]

        loss: torch.Tensor = F.mse_loss(pred, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        num_samples += batch.num_graphs

    return total_loss / num_samples


def _get_device() -> torch.device:
    """Pick device. If multiple GPUs exist, use cuda:0 as primary for DataParallel."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

@hydra.main(version_base=None, config_path=_CONFIG_PATH, config_name="config")
def train(cfg: DictConfig) -> None:
    """Train the GNN model on QM9 dataset.

    Args:
        cfg: Hydra configuration object containing all parameters.
    """
    device: torch.device = _get_device()
    logger.info("Using device: %s", device)

    model_dir: Path = get_data_path(
        cfg.training.model_dir,
        gcs_bucket=OmegaConf.select(cfg, "training.gcs_bucket"),
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    print(cfg)

    profile: bool = cfg.training.profile
    profiler_run_dir: str = cfg.training.profiler_run_dir
    run = _init_wandb(cfg)
    if run is not None:
        logger.info("wandb logging enabled (run: %s)", run.id)
    else:
        logger.info("wandb logging disabled")
    with timing_checkpoint("Load dataset", enabled=profile):
        logger.info("Loading QM9 dataset...")
        data_path = get_data_path(
            cfg.training.data_path,
            gcs_bucket=OmegaConf.select(cfg, "training.gcs_bucket"),
        )
        dataset: Dataset = QM9Dataset(data_path)

    # Apply normalization transform
    dataset.transform = NormalizeScale()

    # Split dataset
    n: int = len(dataset)
    train_size: int = int(cfg.training.train_ratio * n)
    val_size: int = int(cfg.training.val_ratio * n)
    test_size: int = n - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    # Create data loaders
    # Create data loaders (parallel loading)
    workers = _num_workers(cfg)
    logger.info("DataLoader num_workers=%d", workers)

    train_loader: DataLoader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(workers > 0),
    )

    val_loader: DataLoader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(workers > 0),
    )

    test_loader: DataLoader = DataLoader(
        test_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(workers > 0),
    )

    logger.info("Dataset split - Train: %d, Val: %d, Test: %d", len(train_dataset), len(val_dataset), len(test_dataset))

    # Get target indices and infer output dimension
    target_indices: list[int] = list(cfg.training.target_indices)
    num_targets: int = len(target_indices)
    logger.info("Predicting %d target(s): %s", num_targets, target_indices)

    # Initialize model
    model: GraphNeuralNetwork | nn.DataParallel = GraphNeuralNetwork(
        num_node_features=cfg.model.num_node_features,
        hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers,
        output_dim=num_targets,
    ).to(device)

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))

    optimizer: Optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)

    # Early stopping variables
    best_val_loss: float = float("inf")
    patience: int = cfg.training.patience
    patience_counter: int = 0

    logger.info(
        "Starting training for %d epochs (batch_size=%d, lr=%g, patience=%d)",
        cfg.training.epochs,
        cfg.training.batch_size,
        cfg.training.learning_rate,
        patience,
    )
    profiler = TrainingProfiler(enabled=profile, output_dir=Path(f"profiling_results/{profiler_run_dir}"))

    for epoch in range(1, cfg.training.epochs + 1):
        train_loss: float = train_epoch(model, train_loader, optimizer, device, target_indices)

        # compute validation metrics (mse/rmse/mae/r2)
        val_metrics = evaluate_with_metrics(model, val_loader, device, target_indices)
        val_loss: float = float(val_metrics["mse"])  # keep early-stopping tied to MSE

        if epoch % LOG_INTERVAL == 0 or epoch == 1:
            logger.info(
                "Epoch %3d | Train Loss: %.6f | Val MSE: %.6f | Val RMSE: %.6f | Val MAE: %.6f | Val R2: %.6f",
                epoch, train_loss, val_metrics["mse"], val_metrics["rmse"], val_metrics["mae"], val_metrics["r2"]
            )

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_path: Path = model_dir / "best_model.pt"
            torch.save(
                model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                best_model_path,
            )
            logger.debug("Saved best model to %s", best_model_path)
        else:
            patience_counter += 1

        if run is not None:
            wandb.log(
                {
                    "epoch": epoch,
                    "loss/train": train_loss,

                    # keep your existing val loss key if you want
                    "loss/val": val_loss,

                    # add full validation metrics
                    "val/mse": val_metrics["mse"],
                    "val/rmse": val_metrics["rmse"],
                    "val/mae": val_metrics["mae"],
                    "val/r2": val_metrics["r2"],

                    "early_stopping/patience_counter": patience_counter,
                    "early_stopping/best_val_loss": best_val_loss,
                }
            )

        if patience_counter >= patience:
            logger.info("Early stopping triggered at epoch %d (best_val_loss=%.6f)", epoch, best_val_loss)
            break

        profiler.step()
    profiler.finalize()

    # Load best model and evaluate on test set
    best_model_path = model_dir / "best_model.pt"

    try:
        state = torch.load(best_model_path, weights_only=True)
    except TypeError:
        state = torch.load(best_model_path)

    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(state)
    else:
        model.load_state_dict(state)

    test_metrics: dict[str, float] = evaluate_with_metrics(model, test_loader, device, target_indices)
    test_loss: float = float(test_metrics["mse"])
    logger.info("Final test loss: %.6f", test_loss)

    # Save final model
    final_model_path: Path = model_dir / "final_model.pt"
    torch.save(
    model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
    final_model_path,
    )  
    logger.info("Training complete. Models saved to %s", model_dir)

    # wandb: final logs
    if run is not None:
        # Log full test metrics (assumes you computed `test_metrics` as shown before)
        wandb.log(
            {
                "loss/test": test_loss,              # keep compatibility (MSE)
                "test/mse": test_metrics["mse"],
                "test/rmse": test_metrics["rmse"],
                "test/mae": test_metrics["mae"],
                "test/r2": test_metrics["r2"],
            }
        )

        # Optionally log model artifact
        artifact = None
        if bool(OmegaConf.select(cfg, "wandb.log_artifacts", default=True)):
            artifact = wandb.Artifact(
                name="qm9-gnn",
                type="model",
                description="Trained model",
                metadata={
                    "target_indices": target_indices,
                    "best_val_loss": best_val_loss,
                    "test_mse": test_metrics["mse"],
                    "test_rmse": test_metrics["rmse"],
                    "test_mae": test_metrics["mae"],
                    "test_r2": test_metrics["r2"],
                },
            )
            artifact.add_file(str(best_model_path))
            run.log_artifact(artifact)

            # Link only if we actually created an artifact
            run.link_artifact(
                artifact=artifact,
                target_path="model-registry/mlops-molecules",
                aliases=["latest"],
            )

        wandb.finish()


if __name__ == "__main__":
    train()
