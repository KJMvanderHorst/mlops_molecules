import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import typer

from model import GraphNeuralNetwork


@torch.no_grad()
def evaluate(
    model: GraphNeuralNetwork,
    loader: DataLoader,
    device: torch.device,
    target_indices: list[int],
) -> float:
    """Evaluate the model on a dataset.

    Args:
        model: The GNN model.
        loader: DataLoader for validation/test data.
        device: Device to evaluate on.
        target_indices: List of target property indices to predict.

    Returns:
        Average MSE loss over the dataset.
    """
    model.eval()
    total_loss: float = 0.0
    num_samples: int = 0

    for batch in loader:
        batch = batch.to(device)

        pred: torch.Tensor = model(batch)
        target: torch.Tensor = batch.y[:, target_indices]
        loss: torch.Tensor = F.mse_loss(pred, target)

        total_loss += loss.item() * batch.num_graphs
        num_samples += batch.num_graphs

    return total_loss / num_samples


if __name__ == "__main__":
    typer.run(evaluate)
