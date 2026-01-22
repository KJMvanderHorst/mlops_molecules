# quantize.py
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Sequence

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import NormalizeScale

from data import QM9Dataset
from model import GraphNeuralNetwork

logger = logging.getLogger(__name__)

_CONFIG_PATH = str(Path(__file__).parent.parent.parent / "configs")


def _resolve_path(cfg_path: str) -> Path:
    p = Path(cfg_path)
    return p if p.is_absolute() else (Path.cwd() / p)


def _build_model(cfg: DictConfig, num_targets: int) -> GraphNeuralNetwork:
    return GraphNeuralNetwork(
        num_node_features=cfg.model.num_node_features,
        hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers,
        output_dim=num_targets,
        dropout=float(OmegaConf.select(cfg, "model.dropout", default=0.1)),
    )


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
    idx = list(target_indices)

    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        target = batch.y[:, idx]
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
    Average latency per batch in ms.
    Note: for quantized CPU models this is the typical use case.
    """
    model.eval()
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

    times: list[float] = []
    for _ in range(timed_batches):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        batch = batch.to(device)
        t0 = time.perf_counter()
        _ = model(batch)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    if not times:
        return {"ms_per_batch": 0.0, "batches": 0.0}

    avg_s = sum(times) / len(times)
    return {"ms_per_batch": avg_s * 1000.0, "batches": float(len(times))}


def quantize_full_model(model: torch.nn.Module, scheme: str) -> torch.nn.Module:
    """
    Apply weight-only INT8 quantization to *all* linear layers in the model, including those
    nested inside GraphConv blocks. Uses torchao when available and falls back to the
    torch.ao dynamic quantization API otherwise.

    scheme:
      - "torchao_int8_weight_only" (default)
      - "torch_ao_dynamic" (fallback-style dynamic quantization)
    """
    if scheme == "torch_ao_dynamic":
        from torch.ao.quantization import quantize_dynamic  # older weights-only API
        return quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

    # default: torchao
    try:
        from torchao.quantization import quantize_
        from torchao.quantization import Int8WeightOnlyConfig
    except Exception as e:
        logger.warning("torchao not available (%s). Falling back to torch.ao.quantization.quantize_dynamic.", e)
        from torch.ao.quantization import quantize_dynamic
        return quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

    # torchao quantize_ is inplace; returns None or model depending on version
    quantize_(model, Int8WeightOnlyConfig())  #  [oai_citation:3‡PyTorch Documentation](https://docs.pytorch.org/ao/stable/generated/torchao.quantization.quantize_.html)
    return model


@hydra.main(version_base=None, config_path=_CONFIG_PATH, config_name="config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)

    # Quantization (both torchao eager and quantize_dynamic) is primarily CPU-oriented in practice.
    # Keep benchmark apples-to-apples on CPU.
    device = torch.device("cpu")
    logger.info("Benchmark device: %s", device)

    warmup_batches: int = int(OmegaConf.select(cfg, "quantization.warmup_batches", default=10))
    timed_batches: int = int(OmegaConf.select(cfg, "quantization.timed_batches", default=50))
    scheme: str = str(OmegaConf.select(cfg, "quantization.scheme", default="torchao_int8_weight_only"))

    target_indices = list(cfg.training.target_indices)
    num_targets = len(target_indices)

    # --- data ---
    data_path = _resolve_path(str(cfg.training.data_path))
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
        num_workers=0,  # less timing noise
    )

    # --- load baseline model ---
    model_dir = _resolve_path(str(cfg.training.model_dir))
    model_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = model_dir / "best_model.pt"
    logger.info("Loading best model from: %s", best_model_path)

    model_fp32 = _build_model(cfg, num_targets).to(device)

    try:
        state = torch.load(best_model_path, weights_only=True, map_location="cpu")
    except TypeError:
        state = torch.load(best_model_path, map_location="cpu")

    model_fp32.load_state_dict(state)

    # --- baseline eval + speed ---
    base_metrics = evaluate_with_metrics(model_fp32, test_loader, device, target_indices)
    base_latency = measure_inference_latency(
        model_fp32, test_loader, device, warmup_batches=warmup_batches, timed_batches=timed_batches
    )

    # --- quantize entire model (all Linear layers) ---
    model_q = quantize_full_model(model_fp32, scheme=scheme)
    model_q.eval()

    # --- quantized eval + speed ---
    q_metrics = evaluate_with_metrics(model_q, test_loader, device, target_indices)
    q_latency = measure_inference_latency(
        model_q, test_loader, device, warmup_batches=warmup_batches, timed_batches=timed_batches
    )

    # --- report ---
    print("\n" + "=" * 70)
    print("QUANTIZATION REPORT (Full model)")
    print("=" * 70)
    print(f"Scheme:  {scheme}")
    print(f"Device:  {device}")
    print(f"Targets: {target_indices}")
    print("-" * 70)
    print("Performance (Test set)")
    print(
        f"  FP32:  mse={base_metrics['mse']:.6f} rmse={base_metrics['rmse']:.6f} "
        f"mae={base_metrics['mae']:.6f} r2={base_metrics['r2']:.6f}"
    )
    print(
        f"  INT8:  mse={q_metrics['mse']:.6f} rmse={q_metrics['rmse']:.6f} "
        f"mae={q_metrics['mae']:.6f} r2={q_metrics['r2']:.6f}"
    )
    print("-" * 70)
    print("Inference latency (average)")
    print(f"  FP32:  {base_latency['ms_per_batch']:.3f} ms/batch over {int(base_latency['batches'])} batches")
    print(f"  INT8:  {q_latency['ms_per_batch']:.3f} ms/batch over {int(q_latency['batches'])} batches")
    speedup = (base_latency["ms_per_batch"] / q_latency["ms_per_batch"]) if q_latency["ms_per_batch"] > 0 else 0.0
    print(f"  Speedup: {speedup:.3f}x")
    print("=" * 70 + "\n")

    # --- save ---
    # With torchao quantize_ (inplace), easiest for *reloading* later is:
    #   build FP32 model -> apply same quantize_fc_only(...) -> load this state_dict
    quant_path = model_dir / "quantized_model.pt"
    torch.save(model_q.state_dict(), quant_path)
    logger.info("Saved quantized model state_dict to: %s", quant_path)


if __name__ == "__main__":
    main()