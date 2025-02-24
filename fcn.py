# ------------------
# Imports
# ------------------
# general
import os
import numpy as np

import time

start_import = time.time()
from datetime import timedelta, datetime as dt
from pathlib import Path
import pandas as pd

# torch
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
import wandb

from fcn_model import FlowModel
from helpers import *

end_import = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------
# Dictionaries and Global Variables
# ------------------
"""
Stores redshifts, sim suites, and filepaths for data
Sorted by file size/galaxy count
"""
data_path = Path(
    "/n/holystore01/LABS/itc_lab/Lab/galaxyGNN/carol_processed_data/baseline"
)

# TODO changed for debugging purposes, will move dir to shared folder later
checkpoint_path = Path("/n/home03/hbrittain/model_outs/")

# ------------------
# Functions
# ------------------

def train(config: DataConfig,
          train_loader, 
          test_loader,
          max_steps: int = 200_000):
    """
    Trains a model on a given simulation suite and redshift, saving the best model to the checkpoint path.

    Args:
        config (DataConfig): All kwargs specified.
        
        train_loader (DataLoader): DataLoader for training data.
        test_loader (DataLoader): DataLoader for test data.
    """
    # run name
    run_name = (
        f"FCN_{sim}_z{z}_target_{config.target}"
        if not config.observables_only
        else f"FCN_{sim}_z{z}_target_{config.target}_observable_features_only"
    )
    wandb_logger = WandbLogger(
        log_model=False,
        project=run_name,
    )

    best_check = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename=f"best_fcn_{config.sim}z{config.z}"+"_model-{step:02d}-{val_loss:.2f}-{mse_loss:.2f}",
        dirpath=config.ckpt_dir / run_name,
        verbose=True,
    )
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=50,  # Number of checks with no improvement before stopping
        mode="min",
        verbose=True,
    )

    print("Checkpoint path = ", config.ckpt_dir / run_name)
    # training
    trainer = L.Trainer(
        max_steps=max_steps,
        logger=wandb_logger,
        gradient_clip_val=1.0,
        val_check_interval=0.5 if config.sim == "TNG" else 0.05,
        callbacks=[best_check, early_stop],
        default_root_dir=config.ckpt_dir / run_name,
        enable_progress_bar=False,
    )

    num_features = next(iter(test_loader))[0].shape[1]
    model = FlowModel(context=num_features)

    trainer.fit(
        model=model, train_dataloaders=train_loader, val_dataloaders=test_loader
    )

    print(f"Best  model saved at: {best_check.best_model_path}")
    wandb.finish()


# ------------------
# Main Training
# ------------------

if __name__ == "__main__":
    print(f"{os.path.basename(__file__)} is running")
    print(
    f"Imports took {str(timedelta(seconds=(end_import-start_import)))}. This program is using: {device}"
    )
    
    start_whole = time.time()
    datadir = Path("/n/holystore01/LABS/itc_lab/Lab/galaxyGNN/carol_processed_data/baseline")
    zs = [3,4,5,6]
    sims = ['TNG', 'ASTRID']
    
    target_hm_z0 = False
    
    for observable_features_only in [False, True]:
        for sim in sims:
            for z in zs:
                # LOADING DATA
                config = DataConfig(
                    data_dir=datadir.as_posix(),
                    ckpt_dir=checkpoint_path.as_posix(),
                    model_type='fcn',
                    sim=sim,
                    z=z,
                    observables_only=observable_features_only,
                    hm_present=False
                )
                start_load = time.time()
                print(f"\033[35mLoading data for {config.sim} at z={config.z}\033[37m")
                train_loader, val_loader, _ = load_data(config)
                
                end_load = time.time()

                # TRAINING
                print(
                    f"\033[35mLoading lasted {str(timedelta(seconds=(end_load-start_load)))}. Beginning training\033[37m"
                )
                start_train = time.time()
                train(config,
                    train_loader,
                    val_loader
                )
                end_train = time.time()
                print(f"Training lasted {str(timedelta(seconds=(end_train-start_train)))}")

    # Finishing ------
    end_whole = time.time()
    print(f"The fully connected program took {str(timedelta(seconds=(end_whole-start_whole)))}")
