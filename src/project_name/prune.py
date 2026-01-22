# prune.py
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

from torch import nn

import hydra
import torch
import torch.nn.utils.prune as prune
from omegaconf import DictConfig, OmegaConf
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import NormalizeScale

from data import QM9Dataset
from model import GraphNeuralNetwork

logger = logging.getLogger(__name__)

# Determine config path - works both locally and in container
_CONFIG_PATH = str(Path(__file__).parent.parent.parent / "configs")


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate_mse(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_indices: Sequence[int],
) -> float:
    """Mean MSE per graph (matches your train/eval convention)."""
    model.eval()

    total_loss: float = 0.0
    num_samples: int = 0
    target_idx = list(target_indices)

    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        target = batch.y[:, target_idx]
        loss = torch.nn.functional.mse_loss(pred, target)
        total_loss += float(loss.item()) * batch.num_graphs
        num_samples += batch.num_graphs

    return total_loss / max(1, num_samples)


@torch.no_grad()
def evaluate_with_metrics(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_indices: Sequence[int],
) -> dict[str, float]:
    model.eval()

    all_preds: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    target_idx = list(target_indices)

    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        target = batch.y[:, target_idx]
        all_preds.append(pred)
        all_targets.append(target)

    if not all_preds:
        return {"mse": 0.0, "rmse": 0.0, "mae": 0.0, "r2": 0.0}

    predictions = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)

    mse = torch.nn.functional.mse_loss(predictions, targets).item()
    rmse = torch.sqrt(torch.nn.functional.mse_loss(predictions, targets)).item()
    mae = torch.nn.functional.l1_loss(predictions, targets).item()

    ss_res = torch.sum((targets - predictions) ** 2).item()
    ss_tot = torch.sum((targets - torch.mean(targets)) ** 2).item()
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


def _iter_prunable_weight_params(model: torch.nn.Module) -> Iterable[tuple[torch.nn.Module, str]]:
    """
    Yield (module, param_name) pairs to prune, restricted to *fully connected* layers:
    - model.initial_embedding (nn.Linear)
    - all nn.Linear layers inside model.mlp
    """
    # initial embedding
    if hasattr(model, "initial_embedding") and isinstance(model.initial_embedding, nn.Linear):
        yield (model.initial_embedding, "weight")

    # mlp layers
    if hasattr(model, "mlp"):
        for m in model.mlp.modules():
            if isinstance(m, nn.Linear):
                yield (m, "weight")


def apply_unstructured_pruning(
    model: torch.nn.Module,
    amount: float,
) -> dict[str, Any]:
    """Apply unstructured L1 pruning to FC layers only, then make it permanent."""
    if not (0.0 <= amount < 1.0):
        raise ValueError(f"Prune amount must be in [0, 1). Got: {amount}")

    pruned_modules = list(_iter_prunable_weight_params(model))
    if not pruned_modules:
        logger.warning("No prunable fully-connected (nn.Linear) layers found.")
        return {"modules_pruned": 0, "global_sparsity": 0.0}

    for m, pname in pruned_modules:
        prune.l1_unstructured(m, name=pname, amount=amount)

    for m, pname in pruned_modules:
        prune.remove(m, pname)

    total_elems = 0
    zero_elems = 0
    for m, pname in pruned_modules:
        w = getattr(m, pname)
        total_elems += w.numel()
        zero_elems += int((w == 0).sum().item())

    global_sparsity = (zero_elems / total_elems) if total_elems > 0 else 0.0
    return {
        "modules_pruned": len(pruned_modules),
        "global_sparsity": global_sparsity,
        "zero_elems": zero_elems,
        "total_elems": total_elems,
    }


@torch.no_grad()
def measure_inference_latency(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    warmup_batches: int = 10,
    timed_batches: int = 50,
) -> dict[str, float]:
    """
    Measures average latency per batch (ms) over a fixed number of batches.

    Notes:
    - Uses torch.inference_mode() via @torch.no_grad() + model.eval()
    - Syncs CUDA for accurate timing
    """
    model.eval()

    def _sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize()

    it = iter(loader)

    # Warmup
    for _ in range(warmup_batches):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        batch = batch.to(device)
        _ = model(batch)
    _sync()

    # Timed
    times: list[float] = []
    for _ in range(timed_batches):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        batch = batch.to(device)

        _sync()
        t0 = time.perf_counter()
        _ = model(batch)
        _sync()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    if not times:
        return {"ms_per_batch": 0.0, "batches": 0}

    avg_s = sum(times) / len(times)
    return {"ms_per_batch": avg_s * 1000.0, "batches": float(len(times))}


def _resolve_path(cfg_path: str, gcs_bucket: str | None = None) -> Path:
    """
    Minimal path resolver.
    - If you're using mounted GCS (/gcs/<bucket>/...), you can pass absolute paths in cfg already.
    - Otherwise, keep it as local relative/absolute.
    """
    p = Path(cfg_path)
    if p.is_absolute():
        return p
    # If they pass something like "models/" it becomes relative to CWD.
    return Path.cwd() / p


@hydra.main(version_base=None, config_path=_CONFIG_PATH, config_name="config")
def main(cfg: DictConfig) -> None:
    device = _get_device()
    logger.info("Using device: %s", device)

    # --- config defaults (no need to edit your config.yaml) ---
    prune_amount: float = float(OmegaConf.select(cfg, "pruning.amount", default=0.10))
    warmup_batches: int = int(OmegaConf.select(cfg, "pruning.warmup_batches", default=10))
    timed_batches: int = int(OmegaConf.select(cfg, "pruning.timed_batches", default=50))

    target_indices = list(cfg.training.target_indices)
    num_targets = len(target_indices)

    # --- data ---
    data_path = _resolve_path(str(cfg.training.data_path), OmegaConf.select(cfg, "training.gcs_bucket"))
    dataset = QM9Dataset(data_path)
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
        num_workers=0,  # keep timing less noisy; adjust if you want
        pin_memory=torch.cuda.is_available(),
    )

    # --- model ---
    model = GraphNeuralNetwork(
        num_node_features=cfg.model.num_node_features,
        hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers,
        output_dim=num_targets,
        dropout=float(OmegaConf.select(cfg, "model.dropout", default=0.1)),
    ).to(device)

    model_dir = _resolve_path(str(cfg.training.model_dir), OmegaConf.select(cfg, "training.gcs_bucket"))
    model_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = model_dir / "best_model.pt"
    logger.info("Loading best model from: %s", best_model_path)

    try:
        state = torch.load(best_model_path, weights_only=True, map_location=device)
    except TypeError:
        state = torch.load(best_model_path, map_location=device)

    model.load_state_dict(state)

    # --- baseline ---
    base_metrics = evaluate_with_metrics(model, test_loader, device, target_indices)
    base_latency = measure_inference_latency(
        model,
        test_loader,
        device,
        warmup_batches=warmup_batches,
        timed_batches=timed_batches,
    )

    # --- prune ---
    logger.info("Applying unstructured L1 pruning: amount=%.2f", prune_amount)
    prune_stats = apply_unstructured_pruning(model, amount=prune_amount)

    # --- after prune ---
    pruned_metrics = evaluate_with_metrics(model, test_loader, device, target_indices)
    pruned_latency = measure_inference_latency(
        model,
        test_loader,
        device,
        warmup_batches=warmup_batches,
        timed_batches=timed_batches,
    )

    # --- report ---
    print("\n" + "=" * 70)
    print("PRUNING REPORT")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Targets: {target_indices}")
    print(f"Prune amount (requested): {prune_amount:.2f}")
    print(f"Modules pruned: {prune_stats.get('modules_pruned', 0)}")
    print(
        "Global sparsity (measured): "
        f"{100.0 * float(prune_stats.get('global_sparsity', 0.0)):.2f}% "
        f"({prune_stats.get('zero_elems', 0)}/{prune_stats.get('total_elems', 0)})"
    )
    print("-" * 70)
    print("Performance (Test set)")
    print(
        f"  Baseline: mse={base_metrics['mse']:.6f} rmse={base_metrics['rmse']:.6f} "
        f"mae={base_metrics['mae']:.6f} r2={base_metrics['r2']:.6f}"
    )
    print(
        f"  Pruned:   mse={pruned_metrics['mse']:.6f} rmse={pruned_metrics['rmse']:.6f} "
        f"mae={pruned_metrics['mae']:.6f} r2={pruned_metrics['r2']:.6f}"
    )
    print("-" * 70)
    print("Inference latency (average)")
    print(f"  Baseline: {base_latency['ms_per_batch']:.3f} ms/batch over {int(base_latency['batches'])} batches")
    print(f"  Pruned:   {pruned_latency['ms_per_batch']:.3f} ms/batch over {int(pruned_latency['batches'])} batches")
    speedup = (
        (base_latency["ms_per_batch"] / pruned_latency["ms_per_batch"]) if pruned_latency["ms_per_batch"] > 0 else 0.0
    )
    print(f"  Speedup:  {speedup:.3f}x")
    print("=" * 70 + "\n")

    # --- save pruned model ---
    pruned_model_path = model_dir / "pruned_model.pt"
    torch.save(model.state_dict(), pruned_model_path)
    logger.info("Saved pruned model to: %s", pruned_model_path)


if __name__ == "__main__":
    main()
