from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import NormalizeScale
import typer

from project_name.model import GraphNeuralNetwork
from project_name.data import load_qm9_dataset


def evaluate_model(
    model_path: str = "models/best_model.pt",
    data_path: str = "data",
    batch_size: int = 32,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> None:
    """Evaluate the trained model on test set.

    Args:
        model_path: Path to saved model weights.
        data_path: Path to data directory.
        batch_size: Batch size for evaluation.
        train_ratio: Fraction of data used for training.
        val_ratio: Fraction of data used for validation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    print("Loading QM9 dataset...")
    dataset = load_qm9_dataset(data_path).get_training_dataset()
    dataset.transform = NormalizeScale()

    n = len(dataset)
    train_size = int(train_ratio * n)
    val_size = int(val_ratio * n)
    test_size = n - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = GraphNeuralNetwork(
        num_node_features=11,
        hidden_dim=128,
        num_layers=3,
        output_dim=1,
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print(f"Evaluating on {len(test_dataset)} test samples...")

    total_loss = 0
    predictions = []
    targets = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)

            pred = model(batch)
            target = batch.y[:, 0].unsqueeze(1)

            loss = F.mse_loss(pred, target)
            total_loss += loss.item() * batch.num_graphs

            predictions.extend(pred.cpu().numpy().flatten())
            targets.extend(target.cpu().numpy().flatten())

    mse_loss = total_loss / len(test_dataset)
    rmse_loss = torch.sqrt(torch.tensor(mse_loss)).item()

    mae_loss = sum(abs(p - t) for p, t in zip(predictions, targets)) / len(targets)

    print(f"\n{'='*50}")
    print(f"Test Set Evaluation Metrics")
    print(f"{'='*50}")
    print(f"MSE:  {mse_loss:.6f}")
    print(f"RMSE: {rmse_loss:.6f}")
    print(f"MAE:  {mae_loss:.6f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    typer.run(evaluate_model)
