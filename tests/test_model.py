# tests/test_model.py
import torch
import pytest

from project_name.model import Model


def test_model_instantiation():
    """Test that the model can be instantiated."""
    model = Model()
    assert model is not None


def test_model_forward_pass():
    """Test that the model can perform forward pass."""
    model = Model()
    x = torch.rand(1)
    output = model(x)
    assert output is not None
    assert output.shape == torch.Size([1])
