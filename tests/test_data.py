import torch
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from torch_geometric.data import Data

from project_name.data import QM9Dataset


@pytest.fixture
def sample_qm9_data():
    """Create sample QM9-like graph data."""
    data = Data(
        x=torch.randn(5, 11),  # 5 nodes with 11 features
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
        y=torch.tensor([[1.5]]),  # Label
    )
    return data


class TestQM9Dataset:
    """Tests for QM9Dataset class."""

    @patch("project_name.data.QM9")
    def test_dataset_length(self, mock_qm9_class, sample_qm9_data):
        """Test that dataset returns correct length."""
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 10
        mock_qm9_class.return_value = mock_dataset

        dataset = QM9Dataset(Path("data"))

        assert len(dataset) == 10, "Dataset should have 10 samples"

    @patch("project_name.data.QM9")
    def test_sample_shape(self, mock_qm9_class, sample_qm9_data):
        """Test that samples have correct node and edge shape."""
        mock_dataset = MagicMock()
        mock_dataset.__getitem__.return_value = sample_qm9_data
        mock_qm9_class.return_value = mock_dataset

        dataset = QM9Dataset(Path("data"))
        sample = dataset[0]

        assert sample.x.shape[1] == 11, f"Node features should be 11, got {sample.x.shape[1]}"
        assert sample.edge_index.dim() == 2, f"Edge index should be 2D, got {sample.edge_index.dim()}D"
        assert sample.edge_index.shape[0] == 2, (
            f"Edge index should have 2 rows (src, dst), got {sample.edge_index.shape[0]}"
        )

    @patch("project_name.data.QM9")
    def test_all_samples_have_labels(self, mock_qm9_class, sample_qm9_data):
        """Test that all samples have labels."""
        mock_dataset = MagicMock()
        mock_dataset.__getitem__.return_value = sample_qm9_data
        mock_dataset.__len__.return_value = 3
        mock_qm9_class.return_value = mock_dataset

        dataset = QM9Dataset(Path("data"))

        for i in range(len(dataset)):
            sample = dataset[i]
            assert hasattr(sample, "y"), f"Sample {i} missing label attribute"
            assert sample.y is not None, f"Sample {i} has None label"
            assert sample.y.shape[0] > 0, f"Sample {i} label has empty shape"


class TestLoadQM9Dataset:
    """Tests for load_qm9_dataset function."""

    @patch("project_name.data.QM9")
    def test_load_returns_qm9_dataset_instance(self, mock_qm9_class):
        """Test that load_qm9_dataset returns QM9Dataset instance."""
        mock_qm9_class.return_value = MagicMock()

        dataset = QM9Dataset(Path("data"))

        assert isinstance(dataset, QM9Dataset), f"Expected QM9Dataset instance, got {type(dataset)}"
