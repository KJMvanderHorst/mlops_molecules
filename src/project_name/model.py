from torch import nn
import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, global_mean_pool


class GraphNeuralNetwork(nn.Module):
    """Graph Neural Network for molecular property regression."""

    def __init__(
        self,
        num_node_features: int = 11,
        num_edge_features: int = 4,
        hidden_dim: int = 128,
        num_layers: int = 3,
        output_dim: int = 1,
        dropout: float = 0.1,
    ) -> None:
        """Initialize the GNN model.

        Args:
            num_node_features: Number of node (atom) features.
            num_edge_features: Number of edge (bond) features.
            hidden_dim: Number of hidden channels.
            num_layers: Number of GraphConv layers.
            output_dim: Output dimension (1 for single property regression).
            dropout: Dropout rate for regularization.
        """
        super().__init__()
        self.dropout_rate = dropout

        self.initial_embedding = nn.Linear(num_node_features, hidden_dim)

        self.conv_layers = nn.ModuleList([GraphConv(hidden_dim, hidden_dim) for _ in range(num_layers)])

        self.pool = global_mean_pool

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, data) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            data: PyTorch Geometric Data object with x, edge_index, and batch attributes.

        Returns:
            Predicted property values.
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.initial_embedding(x)
        x = F.relu(x)

        for conv in self.conv_layers:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        x = self.pool(x, batch)

        x = self.mlp(x)

        return x


if __name__ == "__main__":
    from torch_geometric.data import Data

    model = GraphNeuralNetwork()

    x = torch.randn(10, 11)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    batch = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    data = Data(x=x, edge_index=edge_index, batch=batch)

    output = model(data)
    print(f"Output shape: {output.shape}")
    print(f"Output value: {output.item():.4f}")
