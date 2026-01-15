import torch
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from project_name.data import MyDataset


class TestMyDataset:
    """Tests for MyDataset class."""

    def test_dataset_instantiation(self):
        """Test that dataset can be instantiated."""
        dataset = MyDataset(Path("data"))
        assert dataset is not None
        assert dataset.data_path == Path("data")

    def test_dataset_has_required_methods(self):
        """Test that dataset has required methods."""
        dataset = MyDataset(Path("data"))
        assert hasattr(dataset, "__len__")
        assert hasattr(dataset, "__getitem__")
        assert hasattr(dataset, "preprocess")
