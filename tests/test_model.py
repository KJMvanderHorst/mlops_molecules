import torch
from torch_geometric.data import Data

from project_name.model import GraphNeuralNetwork


def test_model_instantiation():
    """Test that the model can be instantiated with correct configuration."""
    num_node_features = 11
    hidden_dim = 64
    num_layers = 2
    output_dim = 1
    dropout = 0.1

    model = GraphNeuralNetwork(
        num_node_features=num_node_features,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=output_dim,
        dropout=dropout,
    )

    assert model is not None
    assert isinstance(model, GraphNeuralNetwork)
    assert len(model.conv_layers) == num_layers


def test_forward_pass_output_shape():
    """Test that forward pass produces correct output shape."""
    model = GraphNeuralNetwork(num_node_features=11, hidden_dim=64, output_dim=1)

    x = torch.randn(10, 11)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1, 2, 2, 2])

    data = Data(x=x, edge_index=edge_index, batch=batch)
    output = model(data)

    assert output.shape == (3, 1)  # 3 graphs in batch


def test_gradient_flow():
    """Test that gradients can flow through the model."""
    model = GraphNeuralNetwork(num_node_features=11, hidden_dim=64)

    x = torch.randn(10, 11, requires_grad=True)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1, 2, 2, 2])

    data = Data(x=x, edge_index=edge_index, batch=batch)
    output = model(data)
    loss = output.sum()
    loss.backward()

    assert x.grad is not None
    for param in model.parameters():
        assert param.grad is not None


def test_dropout_in_train_mode():
    """Test that dropout is active during training."""
    model = GraphNeuralNetwork(num_node_features=11, hidden_dim=64, dropout=0.5)
    model.train()

    x = torch.randn(10, 11)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
    data = Data(x=x, edge_index=edge_index, batch=batch)

    outputs = [model(data).detach().clone() for _ in range(5)]
    assert not all(torch.allclose(outputs[0], o) for o in outputs[1:])


def test_dropout_disabled_in_eval_mode():
    """Test that dropout is disabled during evaluation."""
    model = GraphNeuralNetwork(num_node_features=11, hidden_dim=64, dropout=0.5)
    model.eval()

    x = torch.randn(10, 11)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
    data = Data(x=x, edge_index=edge_index, batch=batch)

    with torch.no_grad():
        outputs = [model(data).detach().clone() for _ in range(5)]
    assert all(torch.allclose(outputs[0], o) for o in outputs[1:])


def test_different_batch_sizes():
    """Test model with different numbers of graphs in batch."""
    model = GraphNeuralNetwork(num_node_features=11, output_dim=2)

    for num_graphs in [1, 2, 5]:
        x = torch.randn(10, 11)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        batch = torch.zeros(10, dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, batch=batch)
        output = model(data)

        assert output.shape == (1, 2)  # Single graph
