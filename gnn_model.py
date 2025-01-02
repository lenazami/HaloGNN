# ------------------
# Imports
# ------------------
# general -----
from typing import List, Optional, Union, Literal
# torch -----
import torch
from torch import Tensor
import torch.nn as nn 
from torch.nn.functional import mse_loss
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.utils import softmax
from torch_geometric.utils import scatter
import lightning as L
# norm flow -----
import zuko
# plotting -----
import wandb
import io
import matplotlib.pyplot as plt
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------
# Functions
# ------------------
def get_mlp(
    in_channels: int, 
    hidden_layers: List[int], 
    activation=nn.SiLU,
    use_batch_norm: bool = True,
    dropout_rate: float = 0.,
) -> nn.Sequential:
    """
    Creates a multi-layer perceptron (MLP) with optional batch normalization and dropout
    """
    if not hidden_layers:
        raise ValueError("hidden_layers list cannot be empty")
    
    layers = []
    input_dim = in_channels
    for i, hidden_dim in enumerate(hidden_layers):
        layers.append(nn.Linear(input_dim, hidden_dim))
        if i < len(hidden_layers) - 1:  # No activation/bn/dropout after last layer
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(activation())
            layers.append(nn.Dropout(dropout_rate))
        input_dim = hidden_dim # TODO what does this do
    return nn.Sequential(*layers)

def log_matplotlib_figure(figure_label: str):
    """log a matplotlib figure to wandb, avoiding plotly

    Args:
        figure_label (str): label for figure
    """
    # Save plot to a buffer, otherwise wandb does ugly plotly
    buf = io.BytesIO()
    plt.savefig(
        buf,
        format="png",
        dpi=300,
    )
    buf.seek(0)
    image = Image.open(buf)
    # Log the plot to wandb
    wandb.log({f"{figure_label}": wandb.Image(image)})
  
# ------------------
# Classes
# ------------------
class EdgeUpdate(torch.nn.Module):
    def __init__(
        self,
        edge_in_channels: int,
        edge_out_channels: int,
        hidden_layers: List[int],
        use_batch_norm: bool = True,
        dropout_rate: float = 0.
    ):
        """Update edge attributes

        Args:
            edge_in_channels (int): input channels
            edge_out_channels (int): output channels of MLP, determines the dimensionality of the messages
            hidden_layers (List[int]): hidden layers of MLP
            use_batch_norm (bool, optional): Batch normalization in MLP, defaults to true
            dropout_rate (float, optional): Dropout rate in MLP, defaults to 0
        """
        super().__init__()
        if edge_in_channels <= 0:
            raise ValueError("edge_in_channels must be positive")
            
        self.mlp = get_mlp(
            in_channels=edge_in_channels,
            hidden_layers=hidden_layers + [edge_out_channels],
            use_batch_norm=use_batch_norm,
            dropout_rate=dropout_rate
            )

    def forward(self, h_i: Optional[Tensor], h_j: Optional[Tensor], 
                edge_attr: Optional[Tensor], u: Optional[Tensor]) -> Tensor:
        """
        Args:
            h_i (Tensor): node features node i
            h_j (Tensor): node features node j
            edge_attr (Tensor): edge attributes
            u (Tensor): global attributes

        Returns:
            Tensor: updated edge attributes
        """
        inputs_to_concat = []
        if h_i is not None:
            inputs_to_concat.append(h_i)
        if h_j is not None:
            inputs_to_concat.append(h_j)
        if edge_attr is not None:
            inputs_to_concat.append(edge_attr)
        if u is not None:
            inputs_to_concat.append(u)
        
        if not inputs_to_concat:
            raise ValueError("At least one input tensor must be provided")
        inputs = torch.cat(inputs_to_concat, dim=-1)
        return self.mlp(inputs)

class AttentionAggregation(MessagePassing):
    def __init__(self, edge_dim: int, node_dim: int):
        """Attention-based message aggregation
        
        Args:
            edge_dim (int): Amount of edge features
            node_dim (int): Amount of node features
        """
        super().__init__(aggr=None)  # No default aggregation
        self.attention_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, 1),
            nn.LeakyReLU(),
        )

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message_and_aggregate(self, adj_t: Tensor) -> Tensor:
        return adj_t

    def aggregate(self, inputs: Tensor, index: Tensor, dim_size: Optional[int] = None) -> Tensor:
        # Sum up all the weighted messages for each target node
        return scatter(inputs, index, dim=0, dim_size=dim_size, reduce='sum')

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor, index:Tensor,) -> Tensor:
        # Compute attention coefficients
        inputs = torch.cat([x_i, x_j, edge_attr], dim=-1)
        alpha = self.attention_mlp(inputs)
        alpha = softmax(alpha, index)
        return alpha * edge_attr
    
# The node update would do all the work
class NodeUpdate(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_layers: List[int],
        aggr: Union[str, Literal["add", "mean", "max", "attention"]] = "mean",
        use_batch_norm: bool = True,
        dropout_rate: float = 0.,
        edge_dim: Optional[int] = None
    ):
        """Update nodes

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            hidden_layers (List[int]): hidden layers of MLP
            aggr (str, optional): Node aggregation. Defaults to 'mean'.
            use_batch_norm (bool, optional): Batch normalization in MLP, defaults to true
            dropout_rate (float, optional): Dropout rate in MLP, defaults to 0
            edge_dim (int, optional): Edge dimensions, defaults to None, must be used for attention aggregation
        """
        # Initialize with basic aggregation if not using attention
        super().__init__(aggr=aggr if aggr != "attention" else None)
        
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
            
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggr_type = aggr
        
        # Attention mechanism if specified
        if aggr == "attention":
            if edge_dim is None:
                raise ValueError("edge_dim must be specified when using attention aggregation")
            self.attention = AttentionAggregation(edge_dim, in_channels)
        
        self.mlp = get_mlp(
            in_channels=in_channels + edge_dim,
            hidden_layers=hidden_layers + [out_channels],
            use_batch_norm=use_batch_norm,
            dropout_rate=dropout_rate
        )

    def forward(
        self, 
        h: Optional[Tensor], 
        edge_index: Tensor, 
        edge_attr: Tensor, 
        u: Optional[Tensor]
    ) -> Tensor:
        """Update node features

        Args:
            h (Tensor): node features
            edge_index (Tensor): edge index
            edge_attr (Tensor): edge attributes
            u (Tensor): globals

        Returns:
            Tensor: updated node features
        """
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape [2, num_edges]")
            
        # Different aggregation strategies
        if self.aggr_type == "attention":
            msg = self.attention(h, edge_index, edge_attr)
        else:
            msg = self.propagate(edge_index, edge_attr=edge_attr)
            
        to_concat = [msg]
        
        if h is not None:
            to_concat.append(h)
        if u is not None:
            to_concat.append(u)
            
        inputs = torch.cat(to_concat, dim=-1)
        return self.mlp(inputs)

    def message(self, edge_attr: Tensor) -> Tensor:
        """Message function

        Args:
            edge_attr (Tensor): edge attributes

        Returns:
            Tensor: messages
        """
        return edge_attr

class GraphLayer(torch.nn.Module):
    def __init__(
        self,
        node_in_channels: Optional[int] = 2,
        node_out_channels: int = 1,
        edge_in_channels: Optional[int] = 2,
        hidden_layers: List[int] = [128, 128, 128],
        edge_out_channels: int = 16,
        global_in_channels: int = 0,
        use_residual: bool = True,
        aggr_type: str = "mean",
        use_batch_norm: bool = True,
        dropout_rate: float = 0.1
    ):
        # TODO figure out what use_residual is
        """Constructs a single graph Layer in the network

        Args:
            node_in_channels (int): Node input dimension, defaults to 2
            node_out_channels (int): Node output dimension, defaults to 1
            edge_in_channels (int): Edge input dimension, defaults to 2,
            hidden_layers (List[int]): Dimensions for hidden layers, defaults to [128,128,128]
            edge_out_channels (int): Edge output dimension, defaults to 16
            global_in_channels (int): Global attribute input dimension, defaults to 0
            use_residual (bool): ##### defaults to True 
            aggr_type (str): Graph aggregation type, defaults to 'mean'
            use_batch_norm (bool, optional): Batch normalization in node+edge MLPs, defaults to true
            dropout_rate (float, optional): Dropout rate in node+edge MLPs, defaults to 0.1
        """
        super().__init__()
        
        self.node_in_channels = node_in_channels if node_in_channels is not None else 0
        self.edge_in_channels = edge_in_channels if edge_in_channels is not None else 0
        self.use_residual = use_residual
        self.node_out_channels = node_out_channels
        
        # Check if residual connection is possible
        self.valid_residual = (use_residual and 
                             self.node_in_channels == node_out_channels)
        
        total_edge_input = (self.node_in_channels * 2 + 
                          self.edge_in_channels + 
                          global_in_channels)
        
        self.edge_update = EdgeUpdate(
            edge_in_channels=total_edge_input,
            edge_out_channels=edge_out_channels,
            hidden_layers=hidden_layers,
            use_batch_norm=use_batch_norm,
            dropout_rate=dropout_rate
        )
        
        self.node_update = NodeUpdate(
            in_channels=self.node_in_channels,
             # + edge_out_channels,
            out_channels=node_out_channels,
            hidden_layers=hidden_layers,
            aggr=aggr_type,
            use_batch_norm=use_batch_norm,
            dropout_rate=dropout_rate,
            edge_dim=edge_out_channels
        )

    def forward(self, 
                h: Optional[Tensor], 
                edge_index: Tensor, 
                edge_attr: Tensor, 
                global_attr: Optional[Tensor] = None, 
                batch: Optional[Tensor] = None
                ) -> tuple[Tensor, Tensor]:
        '''Updates edges and nodes
        '''
        if edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape [2, num_edges]")
            
        row, col = edge_index
        
        # Store initial node features for residual
        h_initial = h
        
        # Update edges
        edge_attr = self.edge_update(
            h_i=h[row] if h is not None else None,
            h_j=h[col] if h is not None else None,
            edge_attr=edge_attr,
            u=global_attr,
        )
        
        # Update nodes
        h = self.node_update(
            h=h, 
            edge_index=edge_index, 
            edge_attr=edge_attr, 
            u=global_attr,
        )
        
        # Add residual connection if possible
        if self.valid_residual and h_initial is not None:
            h = h + h_initial
            
        return h, edge_attr

class GraphNetwork(torch.nn.Module):
    def __init__(
            self,
            node_features_dim: Optional[int] = 16,
            edge_features_dim: Optional[int] = 3,
            global_features_dim: Optional[int] = 1,
            node_features_hidden_dim: int = 64,
            edge_features_hidden_dim: int = 64,
            global_output_dim: int = 32,
            message_passing_steps: int = 2,
            hidden_layers: Optional[List[int]] = [128, 128, 128],
            use_residual: bool = True,
            aggr_type: str = "mean",
            pooling_type: str = "mean",
            use_batch_norm: bool = True,
            dropout_rate: float = 0.
    ):
        super().__init__()
        
        if message_passing_steps < 1:
            raise ValueError("message_passing_steps must be at least 1")
            
        self.graph_layers = torch.nn.ModuleList()
        self.pooling_type = pooling_type
        
        # Build graph layers
        for idx in range(message_passing_steps):
            if idx == 0:
                node_in_channels = node_features_dim
                node_out_channels = node_features_hidden_dim
                edge_in_channels = edge_features_dim
                edge_out_channels = edge_features_hidden_dim
            else:
                node_in_channels = node_features_hidden_dim
                node_out_channels = node_features_hidden_dim
                edge_in_channels = edge_features_hidden_dim
                edge_out_channels = edge_features_hidden_dim
                
            self.graph_layers.append(
                GraphLayer(
                    node_in_channels=node_in_channels,
                    node_out_channels=node_out_channels,
                    edge_in_channels=edge_in_channels,
                    edge_out_channels=edge_out_channels,
                    hidden_layers=hidden_layers,
                    use_residual=use_residual,
                    aggr_type=aggr_type,
                    use_batch_norm=use_batch_norm,
                    dropout_rate=dropout_rate
                )
            )
            
        self.global_mlp = get_mlp(
            in_channels=node_features_hidden_dim + global_features_dim,
            hidden_layers=hidden_layers + [global_output_dim],
            use_batch_norm=use_batch_norm,
            dropout_rate=dropout_rate
        )

    def _pool_graph(self, h: Tensor, batch: Tensor, data) -> Tensor:
        """Apply the selected pooling operation"""
        if self.pooling_type == "mean":
            return global_mean_pool(h, batch)
        elif self.pooling_type == "max":
            return global_max_pool(h, batch)
        elif self.pooling_type == "add":
            return global_add_pool(h, batch)
        elif self.pooling_type == "central":
            central_mask = data.central_mask
            central_features = h[central_mask]
            return central_features 
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")

    def forward(self, data) -> Tensor:
        """Forward pass through the graph network"""
        if not hasattr(data, 'x'):
            raise ValueError("Input data must have node features (data.x)")
        if not hasattr(data, 'edge_index'):
            raise ValueError("Input data must have edge indices (data.edge_index)")
        if not hasattr(data, 'edge_attr'):
            raise ValueError("Input data must have edge features (data.edge_attr)")
        if not hasattr(data, 'batch'):
            raise ValueError("Input data must have batch indices (data.batch)")
        if self.pooling_type == "central" and not hasattr(data, 'central_mask'):
            raise ValueError("Data must have central_mask when using 'central' pooling")

            
        h, edge_index = data.x, data.edge_index
        edge_attr, batch = data.edge_attr, data.batch

        if hasattr(data, 'global_attr'):
            global_attr = data.global_attr
        else: 
            global_attr = None

        # Message passing layers
        for layer in self.graph_layers:
            h, edge_attr = layer(h, edge_index, edge_attr, global_attr=None, batch=batch)
            
        # Global pooling and final MLP
        h = self._pool_graph(h, batch, data)
        
        # concatenating global features with pooled feature
        if global_attr is not None:
            if len(global_attr.shape) == 1:
                global_attr = global_attr.unsqueeze(-1)
            h = torch.cat([h, global_attr], dim=-1)
        return self.global_mlp(h)

class GraphModel(L.LightningModule):
    def __init__(
            self, 
            #GNN
            batch_size=64,
            context=32, 
            node_features_dim = 7,
            node_features_hidden_dim=64,
            edge_features_hidden_dim=64,
            message_passing_steps=4,
            use_residual=True,
            aggregation_type='attention',
            pooling_type='mean',
            dropout_rate=0.,
            # Flow
            transforms=6, 
            hidden_features=[128,128,128],
    ):
        super().__init__()
        self.model = GraphNetwork(
            global_output_dim=context,
            node_features_dim=node_features_dim,
            node_features_hidden_dim=node_features_hidden_dim,
            edge_features_hidden_dim=edge_features_hidden_dim,
            message_passing_steps=message_passing_steps,
            use_residual=use_residual,
            aggr_type=aggregation_type,
            pooling_type=pooling_type,
            dropout_rate=dropout_rate,
        )
        self.batch_size = batch_size
        self.flow = zuko.flows.MAF(features=1, transforms=transforms, context=context, hidden_features=hidden_features)
        self.save_hyperparameters()
        self.validation_step_outputs = []
    
    def training_step(self, batch):
        y_pred = self.model(batch)
        loss = -self.flow(y_pred).log_prob(batch.y.unsqueeze(-1)).mean()
        self.log("train_loss", loss, prog_bar=True, batch_size=self.batch_size,)
        
        # logging learning rate
        optimizer = self.optimizers()
        lr = optimizer.param_groups[0]['lr']
        self.log('learning_rate', lr, on_step=True, on_epoch=True, batch_size=self.batch_size,)
        return loss
    
    def validation_step(self, batch):
        y_pred = self.model(batch) 
        y_flow = self.flow(y_pred)
        
        loss = -y_flow.log_prob(batch.y.unsqueeze(-1)).mean()
        self.log("val_loss", loss, prog_bar=True, batch_size=self.batch_size,)
        self.validation_step_outputs.append({"val_loss": loss, "batch": batch})
        
        # calculating mse loss
        mse_pred = y_flow.sample((100,)).squeeze()
        mse = mse_loss(mse_pred.mean(axis=0), batch.y)
        self.log("mse_loss", mse, batch_size=self.batch_size,)
        return loss
    
    def on_validation_epoch_end(self):
        ys = []
        pred = []
        
        for output in self.validation_step_outputs:
            batch = output["batch"]
            y_pred = self.model(batch)
            ys.append(batch.y)
            samp_flow = self.flow(y_pred).sample((100,)).squeeze()
            pred.append(samp_flow.T)
        
        ys = torch.cat(ys, dim=0).detach().cpu().numpy()
        pred = torch.cat(pred, dim=0)
        
        pred_mean = pred.mean(axis=1)
        pred_yerr = pred.std(axis=1)
        plt.errorbar(
            ys,
            pred_mean.detach().cpu().numpy(),
            yerr=pred_yerr.detach().cpu().numpy(),
            linestyle='',
            marker='o',
            markersize=1,
            alpha=0.4,
        )
        plt.plot(
            ys,
            ys,
            linestyle='--',
            color='lightgray',
        )
        
        # Log and clear outputs
        log_matplotlib_figure("validation_predictions")
        plt.close()
        self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=5,
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            "monitor": f"val_loss"
        }