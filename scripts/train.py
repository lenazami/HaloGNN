# TODO: get rid of this
# ============================================
# scripts/train.py - Unified training
# ============================================
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
import wandb

from utils import Config, ModelType, load_data, get_logger
from models.fcn_model import FlowModel
from models.gnn_model import GraphModel

logger = get_logger(__name__)

def get_model(cfg: Config, input_dim: int) -> L.LightningModule:
    """Get appropriate model based on config."""
    if cfg.model_type == ModelType.FCN:
        return FlowModel(context=input_dim)
    else:
        return GraphModel(
            node_features_dim=input_dim,
            context=32,
            transforms=6,
            hidden_features=[128, 128, 128],
            node_features_hidden_dim=64,
            edge_features_hidden_dim=64,
            message_passing_steps=2,
            use_residual=True,
            aggregation_type="attention",
            pooling_type="central",
            dropout_rate=0.0,
        )

def train_model(cfg: Config, max_steps: int = 10000, patience: int = 50):
    """Train a model with given configuration."""
    logger.info(f"Training {cfg.model_type.value} for {cfg.sim.value} z={cfg.z}")
    
    # Load data
    train_loader, val_loader, _ = load_data(cfg, batch_size=64)
    
    # Determine input dimensions
    if cfg.model_type == ModelType.FCN:
        input_dim = next(iter(train_loader))[0].shape[1]
    else:
        input_dim = 15 if not cfg.observables_only else 8
    
    # Initialize model
    model = get_model(cfg, input_dim)
    
    # Setup logging
    run_name = f"{cfg.model_type.value}_{cfg.sim.value}_z{cfg.z}"
    if cfg.hm_present:
        run_name += "_hmz0"
    if cfg.observables_only:
        run_name += "_obs"
    
    wandb_logger = WandbLogger(project="DeepHalos", name=run_name, log_model=False)
    
    # Callbacks
    best_checkpoint = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best-{epoch:02d}-{val_loss:.4f}",
        dirpath=cfg.checkpoint_dir,
        verbose=True,
    )
    
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        mode="min",
        verbose=True,
    )
    
    # Training
    trainer = L.Trainer(
        max_steps=max_steps,
        logger=wandb_logger,
        val_check_interval=0.5 if cfg.sim == Simulation.TNG else 0.1,
        callbacks=[best_checkpoint, early_stopping],
        default_root_dir=cfg.checkpoint_dir,
        enable_progress_bar=True,
        gradient_clip_val=1.0,
    )
    
    trainer.fit(model, train_loader, val_loader)
    wandb.finish()
    
    logger.info(f"Best model saved to: {best_checkpoint.best_model_path}")
    return best_checkpoint.best_model_path

def train_all_models(cfg: Config):
    """Train models for all configurations."""
    # Train with and without observables
    for obs_only in [False, True]:
        cfg.observables_only = obs_only
        train_model(cfg)