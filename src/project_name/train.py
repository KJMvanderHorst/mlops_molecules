from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import NormalizeScale

import typer

from model import GraphNeuralNetwork
from data import load_qm9_dataset


def train_epoch(
    model: GraphNeuralNetwork,
    loader: DataLoader,
    optimizer: Adam,
    device: torch.device,
) -> float:
    """Train for one epoch.

    Args:
        model: The GNN model.
        loader: DataLoader for training data.
        optimizer: Adam optimizer.
        device: Device to train on (cuda/cpu).

    Returns:
        Average loss for the epoch.
    """
    model.train()
    total_loss = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        pred = model(batch)
        target = batch.y[:, 4].unsqueeze(1)

        loss = F.mse_loss(pred, target)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: GraphNeuralNetwork,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Evaluate the model.

    Args:
        model: The GNN model.
        loader: DataLoader for validation/test data.
        device: Device to evaluate on.

    Returns:
        Average MSE loss.
    """
    model.eval()
    total_loss = 0

    for batch in loader:
        batch = batch.to(device)

        pred = model(batch)
        target = batch.y[:, 4].unsqueeze(1)
        loss = F.mse_loss(pred, target)
        total_loss += loss.item() * batch.num_graphs

    return total_loss / len(loader.dataset)


def train(
    data_path: str = "data",
    model_dir: str = "models",
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    target_idx: int = 0,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> None:
    """Train the GNN model on QM9 dataset.

    Args:
        data_path: Path to data directory.
        model_dir: Directory to save trained model.
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        learning_rate: Learning rate for optimizer.
        target_idx: Index of target property in QM9.
        train_ratio: Fraction of data for training.
        val_ratio: Fraction of data for validation.
    """
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    model = GraphNeuralNetwork(
        num_node_features=11,
        hidden_dim=128,
        num_layers=3,
        output_dim=1,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")
    patience = 20
    patience_counter = 0

    print(f"Training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                model.state_dict(),
                model_dir / "best_model.pt",
            )
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(torch.load(model_dir / "best_model.pt"))
    test_loss = evaluate(model, test_loader, device)
    print(f"Test Loss: {test_loss:.6f}")

    torch.save(model.state_dict(), model_dir / "final_model.pt")
    print(f"Model saved to {model_dir}")


if __name__ == "__main__":
    typer.run(train)
