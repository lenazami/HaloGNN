# ------------------
# Imports
# ------------------
# general
import time
start_import = time.time()

import logging
from pathlib import Path
from datetime import timedelta
import pandas as pd

# torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
import wandb

# local
from models.fcn_model import FlowModel
from DeepHalos.utils_old.__old_init__ import load_data, get_logger, DataConfig, device, _cfg_raw
from DeepHalos.utils_old.__old_init__ import SIMS, REDSHIFTS, OBSERVABLES

logger = logging.getLogger(__name__)

end_import = time.time()


# ------------------
# Functions
# ------------------

def train(
    cfg: DataConfig, train_loader, test_loader
) -> str:
    """
    Trains a model on a given simulation suite and redshift, saving the best model to the checkpoint path.

    Args:
        config (DataConfig): All kwargs specified.

        train_loader (DataLoader): DataLoader for training data.
        test_loader (DataLoader): DataLoader for test data.
    """
    # configurations
    suff1 = "_hm0" if cfg.hm_present else ""
    suff2 = "_obs" if cfg.observables_only else ""

    # TODO: !!!! changing model_outs to models later !!!!! 
    # run_name = f"models/FCN_{cfg.sim}_z{cfg.z}{suff1}{suff2}"
    run_name = f"FCN_{cfg.sim}_z{cfg.z}{suff1}{suff2}"
    
    # instantiate model
    model = FlowModel(context=next(iter(test_loader))[0].shape[1])
    model.to(device)
    wandb_logger = WandbLogger(project=run_name, log_model=False)
    
    # callbacks
    # TODO: ckpt_dir = _cfg_raw['out_root'] / "model_outs" / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = _cfg_raw['out_root'] / "model_outs" / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    best_check = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best-{step:04d}-{val_loss:.2f}-{mse_loss:.2f}",
        dirpath=_cfg_raw['out_root'] / run_name,
        verbose=True,
    )
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=cfg.early_stop_patience,  # Number of checks with no improvement before stopping
        mode="min",
        verbose=True,
    )
    
    trainer = L.Trainer(
        max_steps=_cfg_raw['max_steps'],
        logger=wandb_logger,
        val_check_interval=0.5 if cfg.sim == "TNG" else 0.05,
        callbacks=[best_check, early_stop],
        default_root_dir=_cfg_raw['root'] / run_name,
        enable_progress_bar=False,
        gradient_clip_val=1.0,
    )
    
    # training
    trainer.fit(model, train_loader, test_loader)
    wandb.finish()
    
    return best_check.best_model_path

def main():
    logger = get_logger("FCN", level=_cfg_raw['logging']['level'])
    logger.info(f"FCN is using device: {device}")
    logger.info(f"Imports took {str(timedelta(end_import-start_import))}")
    
    start_whole = time.time()
    
    for obs in OBSERVABLES:
        for sim in SIMS:
            for z in REDSHIFTS:
                cfg = DataConfig(
                    root=_cfg_raw['root'],
                    model_type="FCN",
                    sim=sim,
                    z=z,
                    graph_radius=_cfg_raw['graph_radius'],
                    observables_only=obs,
                    hm_present=_cfg_raw['hm_present']
                )
                logger.maininfo(f"Loading FCN data for sim={sim}, z={z}, obs_only={cfg.observables_only}")
                
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
