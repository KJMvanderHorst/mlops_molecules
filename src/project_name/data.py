from pathlib import Path

import typer
import torch
from torch.utils.data import Dataset
from torch_geometric.datasets import QM9


class QM9Dataset(Dataset):
    """QM9 dataset wrapper from torch_geometric."""

    def __init__(self, data_path: Path) -> None:
        """Initialize the QM9 dataset.

        Args:
            data_path: Path to the data directory where QM9 will be stored.
        """
        self.data_path = Path(data_path)
        self.dataset = self._load_dataset()

    def _load_dataset(self) -> QM9:
        """Load QM9 dataset, checking if it already exists locally.

        Returns:
            QM9 dataset from torch_geometric.
        """
        raw_path = self.data_path / "raw"
        raw_path.mkdir(parents=True, exist_ok=True)

        dataset = QM9(root=str(self.data_path))
        return dataset

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.dataset)

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""
        return self.dataset[index]

    def get_training_dataset(self) -> QM9:
        """Return the dataset for training.

        Returns:
            QM9 dataset ready for training.
        """
        return self.dataset

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""

def preprocess(data_path: str = "data", output_folder: str = "data/processed") -> None:
    """Load and preprocess the QM9 dataset.

    Args:
        data_path: Path to the data directory.
        output_folder: Path where preprocessed data will be saved.
    """
    print("Loading QM9 dataset...")
    dataset = QM9Dataset(Path(data_path))
    print(f"Dataset loaded with {len(dataset)} samples")
    dataset.preprocess(Path(output_folder))


def load_qm9_dataset(data_path: Path) -> QM9Dataset:
    """Load the QM9 dataset for training.

    Args:
        data_path: Path to the data directory.

    Returns:
        QM9Dataset instance with the loaded dataset.
    """
    return QM9Dataset(data_path)



if __name__ == "__main__":
    typer.run(preprocess)
