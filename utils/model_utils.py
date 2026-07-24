# utils / model_utils.py

# -----------------
# Imports
# -----------------
import torch
from torch.utils.data import DataLoader
from lightning import Trainer
from typing import Tuple

from halo.models import FCNEncoder, FlowModel, GraphNetwork

from .config import Config, ModelType
from .data_handler import get_handler

# -----------------
# Functions
# -----------------

# ----- Model ops -----
def load_model(cfg: Config) -> FlowModel:
    """Reconstruct the configured encoder and load a single-flow checkpoint."""
    model_path = cfg.get_checkpoint_path()
    ckpt_file = next(model_path.glob("*.ckpt"))
    checkpoint = torch.load(ckpt_file, map_location="cpu")
    state_dict = checkpoint["state_dict"]
    hparams = checkpoint.get("hyper_parameters", {})
    context_dim = int(hparams.get("context_dim", 32))

    if cfg.model_type == ModelType.FCN:
        first_weight = state_dict["encoder.network.0.weight"]
        encoder = FCNEncoder(input_dim=first_weight.shape[1], context_dim=context_dim)
    else:
        first_edge_weight = state_dict[
            "encoder.graph_layers.0.edge_update.network.0.weight"
        ]
        node_features_dim = (first_edge_weight.shape[1] - 4) // 2
        layer_indices = {
            int(key.split(".")[2])
            for key in state_dict
            if key.startswith("encoder.graph_layers.")
        }
        encoder = GraphNetwork(
            node_features_dim=node_features_dim,
            context_dim=context_dim,
            message_passing_steps=len(layer_indices),
            aggregation_type="attention",
            pooling_type="central",
            boxsize=cfg.boxsize,
            radius=cfg.graph_radius,
        )

    model = FlowModel(
        encoder=encoder,
        context_dim=context_dim,
        target_dim=int(hparams.get("target_dim", 1)),
        transforms=int(hparams.get("transforms", 6)),
        flow_hidden_dims=tuple(hparams.get("flow_hidden_dims", (128, 128, 128))),
        learning_rate=float(hparams.get("learning_rate", 3e-4)),
        scheduler_patience=int(hparams.get("scheduler_patience", 10)),
        validation_samples=int(hparams.get("validation_samples", 100)),
        prediction_samples=int(hparams.get("prediction_samples", 200)),
    )
    model.load_state_dict(state_dict)
    return model

# -----------------
# Prediction + Evaluation
# -----------------
def get_predictions(
    cfg: Config, 
    test_loader: DataLoader, 
    model: FlowModel
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate predictions from a model.
    
    Returns:
        (truths, predictions, log_probs) - all denormalized
    """
    trainer = Trainer()
    predictions_dict = trainer.predict(model, test_loader)
    
    # extract samples + log probs
    predictions = torch.cat([p["samples"] for p in predictions_dict], dim=1).transpose(0, 1)
    log_probs = torch.cat([p["log_prob"] for p in predictions_dict], dim=0)

    # extract true values based on model type
    if cfg.model_type == ModelType.GNN:
        truths = torch.cat([batch.y for batch in test_loader], dim=0)
    elif cfg.model_type == ModelType.FCN:
        truths = torch.cat([batch[1] for batch in test_loader], dim=0)

    # denormalize
    handler = get_handler(cfg.model_type)
    stats = handler.load_stats(cfg)
    
    truths = handler.denormalize(truths, stats, cfg.label_field())
    predictions = handler.denormalize(predictions, stats, cfg.label_field())
    
    return truths.squeeze(), predictions.squeeze(), log_probs.squeeze()
