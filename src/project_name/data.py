from pathlib import Path

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

        Downloads the dataset on first instantiation if it doesn't exist.

        Returns:
            QM9 dataset from torch_geometric.
        """
        raw_path = self.data_path / "raw"
        raw_path.mkdir(parents=True, exist_ok=True)

        print("Loading QM9 dataset (downloading if not already present)...")
        dataset = QM9(root=str(self.data_path))
        print(f"Dataset ready at {self.data_path}")
        return dataset

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.dataset)

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""
        return self.dataset[index]

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""


if __name__ == "__main__":
    dataset = QM9Dataset(Path("data"))
    print(f"Dataset instance created with {len(dataset)} samples")
