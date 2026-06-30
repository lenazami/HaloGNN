# utils / model_utils.py

# -----------------
# Imports
# -----------------
import torch
from torch.utils.data import DataLoader
from lightning import Trainer
from typing import Tuple, Union

from .config import Config, ModelType
from .data_handler import get_handler

# -----------------
# Functions
# -----------------

# ----- Model ops -----
def load_model(cfg: Config) -> Union["GraphModel", "FlowModel"]:
    """Load a trained model from checkpoint."""
    from models.gnn_model import GraphModel
    from models.fcn_model import FlowModel
    
    model_path = cfg.get_checkpoint_path()
    ckpt_file = next(model_path.glob("*.ckpt"))
    model_class = GraphModel if cfg.model_type == ModelType.GNN else FlowModel
    return model_class.load_from_checkpoint(ckpt_file)

# -----------------
# Prediction + Evaluation
# -----------------
def get_predictions(
    cfg: Config, 
    test_loader: DataLoader, 
    model: Union["GraphModel", "FlowModel"]
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