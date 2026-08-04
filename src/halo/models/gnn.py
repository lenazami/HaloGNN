import torch
from torch import Tensor, nn
from typing import Sequence, Literal, Optional
from torch_geometric.nn import MessagePassing, global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.utils import scatter, softmax

from halo.models.utils import build_mlp
from halo.data import dense_pair_geometry


# --- Graph Neural Network ----

class EdgeUpdate(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int],
        use_batch_norm: bool,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        self.network = build_mlp(
            input_dim,
            output_dim,
            hidden_dims,
            use_batch_norm=use_batch_norm,
            dropout_rate=dropout_rate,
        )

    def forward(self, source: Tensor, target: Tensor, edge_attr: Tensor) -> Tensor:
        return self.network(torch.cat((source, target, edge_attr), dim=-1))


class AttentionAggregation(MessagePassing):
    def __init__(self, edge_dim: int, node_dim: int) -> None:
        super().__init__(aggr=None)
        self.attention = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, 1),
            nn.LeakyReLU(),
        )

    def forward(self, nodes: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        return self.propagate(edge_index, x=nodes, edge_attr=edge_attr)

    def aggregate(
        self, inputs: Tensor, index: Tensor, dim_size: Optional[int] = None
    ) -> Tensor:
        return scatter(inputs, index, dim=0, dim_size=dim_size, reduce="sum")

    def message(
        self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor, index: Tensor
    ) -> Tensor:
        inputs = torch.cat((x_i, x_j, edge_attr), dim=-1)
        return softmax(self.attention(inputs), index) * edge_attr


class NodeUpdate(MessagePassing):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        edge_dim: int,
        hidden_dims: Sequence[int],
        aggregation: Literal["add", "mean", "max", "attention"],
        use_batch_norm: bool,
        dropout_rate: float,
    ) -> None:
        super().__init__(aggr=aggregation if aggregation != "attention" else None)
        self.aggregation = aggregation
        if aggregation == "attention":
            self.attention = AttentionAggregation(edge_dim, input_dim)
        self.network = build_mlp(
            input_dim + edge_dim,
            output_dim,
            hidden_dims,
            use_batch_norm=use_batch_norm,
            dropout_rate=dropout_rate,
        )

    def forward(self, nodes: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        if self.aggregation == "attention":
            messages = self.attention(nodes, edge_index, edge_attr)
        else:
            messages = self.propagate(edge_index, edge_attr=edge_attr)
        return self.network(torch.cat((nodes, messages), dim=-1))

    def message(self, edge_attr: Tensor) -> Tensor:
        return edge_attr


class GraphLayer(nn.Module):
    def __init__(
        self,
        node_input_dim: int,
        node_output_dim: int,
        edge_input_dim: int,
        edge_output_dim: int,
        hidden_dims: Sequence[int],
        use_residual: bool,
        aggregation: Literal["add", "mean", "max", "attention"],
        use_batch_norm: bool,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        self.use_residual = use_residual and node_input_dim == node_output_dim
        self.edge_update = EdgeUpdate(
            2 * node_input_dim + edge_input_dim,
            edge_output_dim,
            hidden_dims,
            use_batch_norm,
            dropout_rate,
        )
        self.node_update = NodeUpdate(
            node_input_dim,
            node_output_dim,
            edge_output_dim,
            hidden_dims,
            aggregation,
            use_batch_norm,
            dropout_rate,
        )

    def forward(
        self, nodes: Tensor, edge_index: Tensor, edge_attr: Tensor
    ) -> tuple[Tensor, Tensor]:
        row, col = edge_index
        initial_nodes = nodes
        edge_attr = self.edge_update(nodes[row], nodes[col], edge_attr)
        nodes = self.node_update(nodes, edge_index, edge_attr)
        if self.use_residual:
            nodes = nodes + initial_nodes
        return nodes, edge_attr


class GraphNetwork(nn.Module):
    """Encode a PyG graph batch into a fixed-width flow context."""

    def __init__(
        self,
        gal_features_dim =  16,
        context_dim =  32,
        hidden_dims = (128, 128, 128),
    ) -> None:
        super().__init__()
        
        node_features_hidden_dim = 64
        edge_features_hidden_dim = 64
        global_features_dim = 1
        message_passing_steps = 2
        use_residual = True
        # you can use mean, max, add, attention
        aggregation_type  = "mean"
        pooling_type = "mean"
        use_batch_norm = True
        dropout_rate = 0.0
        
        self.context_dim = context_dim
        self.global_features_dim = global_features_dim
        self.pooling_type = pooling_type
        # self.boxsize = boxsize
        # self.radius = radius
        self.graph_layers = nn.ModuleList()

        node_input_dim = gal_features_dim
        edge_input_dim = 4
        for _ in range(message_passing_steps):
            self.graph_layers.append(
                GraphLayer(
                    node_input_dim,
                    node_features_hidden_dim,
                    edge_input_dim,
                    edge_features_hidden_dim,
                    hidden_dims,
                    use_residual,
                    aggregation_type,
                    use_batch_norm,
                    dropout_rate,
                )
            )
            node_input_dim = node_features_hidden_dim
            edge_input_dim = edge_features_hidden_dim

        self.readout = build_mlp(
            node_features_hidden_dim + global_features_dim,
            context_dim,
            hidden_dims,
            use_batch_norm=use_batch_norm,
            dropout_rate=dropout_rate,
        )

    @staticmethod
    def _require(data, name: str) -> Tensor:
        value = getattr(data, name, None)
        if not isinstance(value, Tensor):
            raise ValueError(f"graph batch must provide tensor data.{name}")
        return value

    def _pool(self, nodes: Tensor, batch: Tensor, data) -> Tensor:
        if self.pooling_type == "mean":
            return global_mean_pool(nodes, batch)
        if self.pooling_type == "max":
            return global_max_pool(nodes, batch)
        if self.pooling_type == "add":
            return global_add_pool(nodes, batch)

        central_mask = self._require(data, "central_mask").bool()
        graph_count = torch.bincount(batch).numel()
        central_counts = torch.bincount(batch[central_mask], minlength=graph_count)
        if not torch.equal(central_counts, torch.ones_like(central_counts)):
            raise ValueError("central pooling requires exactly one central node per graph")
        return nodes[central_mask]


    def forward(self, data) -> Tensor:
        nodes = self._require(data, "x")
        pos = self._require(data, "pos")
        batch = self._require(data, "batch").long()
        global_attr = self._require(data, "global_attr")
        if global_attr.ndim == 1:
            global_attr = global_attr.reshape(-1, 1)

        # computing on the fly
        radius = float(data.radius.flatten()[0])
        boxsize = float(data.boxsize.flatten()[0])
        edge_index, edge_attr = dense_pair_geometry(pos, batch, radius, boxsize)

        for layer in self.graph_layers:
            nodes, edge_attr = layer(nodes, edge_index, edge_attr)

        pooled = self._pool(nodes, batch, data)
        if global_attr.shape != (pooled.shape[0], self.global_features_dim):
            raise ValueError(
                f"global_attr must have shape [{pooled.shape[0]}, {self.global_features_dim}], "
                f"got {tuple(global_attr.shape)}"
            )
        return self.readout(torch.cat((pooled, global_attr), dim=-1))


