# ------------------
# Imports
# ------------------
# general ------
import time
start_import = time.time()

import logging
from datetime import timedelta
from pathlib import Path
from typing import Tuple

# external -----
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    EarlyStopping,
)
import wandb

# local ------
from models.gnn_model import GraphModel
from DeepHalos.utils_old.__old_init__ import DataConfig, load_data, get_logger, device, _cfg_raw
from DeepHalos.utils_old.__old_init__ import SIMS, REDSHIFTS, OBSERVABLES

logger = logging.getLogger(__name__)

end_import = time.time()


# ------------------
# Functions
# ------------------
def train(
    cfg: DataConfig, train_loader, test_loader
) -> Tuple[GraphModel, str]:
    # Hyperparams
    context = 32
    flow_transforms = 6
    flow_hidden_features = [128, 128, 128]
    node_features_hidden_dim = 64
    edge_features_hidden_dim = 64
    message_passing_steps = 2
    use_residual = True
    aggr_type = "attention"
    pooling_type = "central"
    dropout_rate = 0.0

    suff1 = "_hm0" if cfg.hm_present else ""
    suff2 = "_obs" if cfg.observables_only else ""

    run_name = f"GNN_{cfg.sim}_z{cfg.z}{suff1}{suff2}"

    # instantiate model
    model = GraphModel(
        context=context,
        transforms=flow_transforms,
        hidden_features=flow_hidden_features,
        node_features_hidden_dim=node_features_hidden_dim,
        edge_features_hidden_dim=edge_features_hidden_dim,
        message_passing_steps=message_passing_steps,
        use_residual=use_residual,
        aggregation_type=aggr_type,
        pooling_type=pooling_type,
        dropout_rate=dropout_rate,
        node_features_dim=15 if not cfg.observables_only else 8,
    )
    model.to(device)

    # logger object to monitor progress
    wandb_logger = WandbLogger(
        log_model=False,
        project=run_name,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    # save the best model
    # TODO: c
    # ckpt_dir = cfg.root / "models" / run_name
    ckpt_dir = _cfg_raw['out_root'] / "model_outs" / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    best_check = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename=f"best_gnn_{cfg.sim}_z{cfg.z}_r{cfg.graph_radius:.1f}"
                + "-{step:02d}-{val_loss:.2f}-{mse_loss:.2f}",
        dirpath=ckpt_dir,
        verbose=True,
    )

    # stops early to save us headaches
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=50,  # Number of checks with no improvement before stopping
        mode="min",
        verbose=True,
    )

    # instantiate trainer
    trainer = L.Trainer(
        max_steps=_cfg_raw['max_steps'],
        logger=wandb_logger,
        log_every_n_steps=100,
        gradient_clip_val=1.0,
        callbacks=[lr_monitor, best_check, early_stop],
        val_check_interval=0.5 if cfg.sim == "TNG" else 0.1,
    )
    # training
    trainer.fit(model,train_loader,test_loader)
    wandb.finish()

    return best_check.best_model_path
    
def main():
    logger = get_logger("GNN", level=_cfg_raw['logging']['level'])
    logger.info(f"GNN is using device: {device}")
    logger.info(f"Imports took {str(timedelta(end_import-start_import))}")
    
    start_whole = time.time()
    
    for obs in OBSERVABLES:
        for sim in SIMS:
            for z in REDSHIFTS:
                cfg = DataConfig(
                    root=c,
                    model_type="GNN",
                    sim=sim,
                    z=z,
                    graph_radius=_cfg_raw['graph_radius'],
                    observables_only=obs,
                    hm_present=_cfg_raw['hm_present']
                )
                logger.info(f"\n\n{'='*50}")
                logger.info(f"Processing suite={cfg.sim}, z={cfg.z}, radius={cfg.graph_radius}, obs_only={cfg.observables_only}")
                logger.info(f"{'='*50}\n")
                
                t0 = time.time()
                train_loader, val_loader, _ = load_data(cfg, batch_size=_cfg_raw['batch_size'])
                logger.maininfo(f"Loading took {str(timedelta(seconds=(time.time()-t0)))}. Beginning training...")

                # TRAINING
                t1 = time.time()
                logger.info(f"Starting training.")
                bestpath = train(cfg, train_loader, val_loader, max_steps=_cfg_raw['max_steps'])
                logger.info(f"Best model saved at: {bestpath}")
                logger.maininfo(f"Training took {str(timedelta(seconds=(time.time()-t1)))}.")

    # Finishing ------
    logger.info(f"This program had a total runtime of {str(timedelta(seconds=(time.time()-start_whole)))}")
    
# ------------------
# Main Training
# ------------------

if __name__ == "__main__":
    main()