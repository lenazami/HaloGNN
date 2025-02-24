# ------------------
# Imports
# ------------------
# general ------
import os
import time

start_import = time.time()
import numpy as np
from datetime import timedelta
from pathlib import Path

# torch -----
import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
import wandb

# model ------
from gnn_model import GraphModel
from helpers import *

end_import = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------
# Dictionaries and Global Variables
# ------------------
'''
Stores redshifts, sim suites, and filepaths for training data
Sorted by file size/galaxy count
'''
# TODO changed for debugging purposes
checkpoint_path = Path("/n/home03/hbrittain/model_outs/")
data_path = Path(f"/n/holystore01/LABS/itc_lab/Lab/galaxyGNN/carol_processed_data/")

# ------------------
# Functions
# ------------------

def train(config, train_loader, test_loader, max_steps=200_000) -> tuple[GraphModel, str]:
    context = 32
    flow_transforms = 6
    flow_hidden_features = [128,128,128]
    node_features_hidden_dim = 64
    edge_features_hidden_dim = 64
    message_passing_steps = 2
    use_residual = True
    aggr_type = 'attention' 
    pooling_type = 'central'
    dropout_rate = 0.

    suff1 = "_target_HaloMass_z0" if config.hm_present else "_target_HaloMass"
    suff2 = "_observable_features_only" if config.observables_only else ""
    
    run_name = f"GNN_{config.sim}_z{config.z}{suff1}{suff2}"
    
    # instantiate model
    model = GraphModel(
        context=context,
        transforms=flow_transforms,
        hidden_features = flow_hidden_features,
        node_features_hidden_dim=node_features_hidden_dim,
        edge_features_hidden_dim=edge_features_hidden_dim,
        message_passing_steps=message_passing_steps,
        use_residual=use_residual,
        aggregation_type=aggr_type,
        pooling_type=pooling_type,
        dropout_rate=dropout_rate,
        node_features_dim=15 if not config.observables_only else 8 
    )
    model.to(device)
    
    # logger object to monitor progress
    wandb_logger = WandbLogger(
        log_model=False, 
        project=run_name,
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # save the best model
    best_check = ModelCheckpoint(
            monitor="val_loss",         
            mode="min",                  
            save_top_k=1,                
            filename=f"best_gnn_{config.sim}_z{config.z}"+"_model-{step:02d}-{val_loss:.2f}-{mse_loss:.2f}", 
            dirpath=config.ckpt_dir / run_name,
            verbose=True                 
        )
    
    # stops early to save us headaches
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=50,  # Number of checks with no improvement before stopping
        mode="min",
        verbose=True
    )
    
    # instantiate trainer
    trainer = L.Trainer(
        max_steps=max_steps,
        logger=wandb_logger, 
        log_every_n_steps=100,
        gradient_clip_val=1.0,
        callbacks=[lr_monitor, best_check, early_stop],
        val_check_interval=0.5 if config.sim == "TNG" else 0.1,
    )
    # training
    trainer.fit(
        model=model, 
        train_dataloaders=train_loader, 
        val_dataloaders=test_loader,
    )
    
    print(f"Best model saved at: {best_check.best_model_path}")
    print(type(best_check.best_model_path))
    wandb.finish()
    
    return model, best_check.best_model_path
# ------------------
# Main Training
# ------------------

if __name__ == '__main__':
    print(f"{os.path.basename(__file__)} is running")
    print(
    f"Imports took {str(timedelta(seconds=(end_import-start_import)))}. This program is using: {device}"
    )
    
    start_whole = time.time()
    graph_radius = 2000.
    data_name = f'gnn_{graph_radius:.1f}/'
    
    sims = ['TNG', 'ASTRID']
    zs = [3,4,5,6]
    
    # observable_features_only = False 
    for observable_features_only in [False, True]:
        for sim in sims:
            for z in zs:
                # LOADING DATA
                config = DataConfig(
                    data_dir=data_path,
                    ckpt_dir=checkpoint_path,
                    model_type='gnn',
                    sim=sim,
                    z=z,
                    graph_radius=graph_radius,
                    observables_only=observable_features_only
                )
                batch_size = 64
                start_load = time.time()
                print(f"\033[35mLoading data for {config.sim} at z={config.z}\033[37m")
                train_loader, val_loader, _ = load_data(config,
                                                        batch_size=batch_size)
                end_load = time.time()
                
                # TRAINING
                print(f"\033[35mLoading lasted {str(timedelta(seconds=(end_load-start_load)))}. Beginning training\033[37m")
                start_train = time.time()
                _, _ = train(config, train_loader, val_loader)
                end_train=time.time()
                print(f"Training lasted {str(timedelta(seconds=(end_train-start_train)))}")

    # Finishing ------
    end_whole = time.time()
    print(f"This program took {str(timedelta(seconds=(end_whole-start_whole)))}")