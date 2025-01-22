# ------------------
# Imports
# ------------------
# general
import os
print(f"{os.path.basename(__file__)} is running")
import time
import numpy as np
from datetime import timedelta
from pathlib import Path
# torch, lighting, etc.
import torch
from torch_geometric.loader import DataLoader
# TODO get rid of batch import after restricted
from torch_geometric.data import Batch
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
import wandb
# model
from gnn_model import GraphModel
from data_utils import get_split_indices

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"This program is using: {device}")

# ------------------
# Dictionaries and Global Variables
# ------------------
'''
Stores redshifts, sim suites, and filepaths for training data
Sorted by file size/galaxy count
'''

checkpoint_path = Path("/n/holystore01/LABS/itc_lab/Lab/galaxyGNN/models/")

# ------------------
# Functions
# ------------------


def mask_features(data_list, features_to_mask):
    feature_names = data_list[0].feature_names
    keep_mask = [name not in features_to_mask for name in feature_names]
    keep_indices = [i for i, keep in enumerate(keep_mask) if keep]
    
    masked_data_list = []
    for data in data_list:
        new_data = data.clone()
        new_data.x = data.x[:, keep_indices]
        new_data.feature_names = [name for name in feature_names if name not in features_to_mask]
        masked_data_list.append(new_data)
    return masked_data_list

def standardize_features(data, train_stats):
    feature_names = data[0].feature_names
    # find feature indices in stats
    feature_indices = [train_stats['feature_names'].index(name) for name in feature_names if name in train_stats['feature_names']]
    means = train_stats['means'].astype(np.float32)
    stds = train_stats['stds'].astype(np.float32)
    halo_mass_idx = train_stats['feature_names'].index('HaloMass')
    for graph in data:
        graph.x[:,feature_indices] = (graph.x[:, feature_indices] - means[feature_indices]) / stds[feature_indices]
        graph.y = (graph.y - means[halo_mass_idx]) / stds[halo_mass_idx]

def convert_to_float32(data_list):
    """Convert all tensor attributes to float32."""
    for data in data_list:
        for key, value in data:
            if key not in ['central_mask', 'edge_index', 'feature_names']:
                data[key] = torch.tensor(value).float()  
    return data_list

def load_data(
        datadir, 
        sim,
        z,
        train_sim=None,
        observable_features_only=False,
        batch_size=64, 
    ):
    if train_sim is None:
        train_sim = sim

    data = torch.load(datadir / f'{sim}_z{z}/{sim}_z{z}_all_graphs.pt')

    train_stats = torch.load(datadir / f'{train_sim}_z{z}/{train_sim}_z{z}_feature_stats.pt')
    data = convert_to_float32(data)


    standardize_features(data, train_stats)
    if observable_features_only:
        non_observable_features = [
            "HaloMass",
            "GalaxyVel_1",
            "GalaxyVel_2",
            "GalaxyVel_3",
            "GalaxyRhalf",
            "GalaxyMass",
            "GalaxyVel",
            "SFR",
        ]
        data = mask_features(data, features_to_mask=non_observable_features)
    

    train_idx, val_idx, test_idx = get_split_indices(len(data), test_size=0.1 if sim=='TNG' else 0.01, val_size=0.05)
    
    train_data = [data[i] for i in train_idx]
    val_data = [data[i] for i in val_idx]
    test_data = [data[i] for i in test_idx]
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False,)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False,) 

    return train_loader, val_loader, test_loader



def train(sim, z, train_loader, test_loader, observable_features_only=False):
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

    run_name = (
        f"GNN_{sim}_z{z}"
        if not observable_features_only
        else f"GNN_{sim}_z{z}_observable_features_only"
    ) 
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
        node_features_dim=15 if not observable_features_only else 8 
    )
    
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
            filename=f"best_gnn_{sim}z{z}""_model-{step:02d}-{val_loss:.2f}-{mse_loss:.2f}", 
            dirpath=checkpoint_path / run_name,
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
        max_steps=200_000,
        logger=wandb_logger, 
        log_every_n_steps=100,
        gradient_clip_val=1.0,
        callbacks=[lr_monitor, best_check, early_stop],
        val_check_interval=0.5 if sim == "TNG" else 0.1,
    )
    # training
    trainer.fit(
        model=model, 
        train_dataloaders=train_loader, 
        val_dataloaders=test_loader,
    )
    
    print(f"Best model saved at: {best_check.best_model_path}")
    wandb.finish()
    
# ------------------
# Main Training
# ------------------

if __name__ == '__main__':
    start_whole = time.time()
    graph_radius = 2000.
    data_path = Path(f"/n/holystore01/LABS/itc_lab/Lab/galaxyGNN/carol_processed_data/gnn_{graph_radius:.1f}/")
    sims = ['TNG', 'ASTRID']
    zs = [3,]
    observable_features_only = False 
    for sim in sims:
        for z in zs:
            # LOADING DATA
            batch_size = 64
            start_load = time.time()
            print(f"\033[35mLoading data for {sim} at z={z}\033[37m")
            train_loader, val_loader, _ = load_data(datadir=data_path, sim=sim, z=z, batch_size=batch_size, observable_features_only=observable_features_only)
            end_load = time.time()
            
            # TRAINING
            print(f"\033[35mLoading lasted {str(timedelta(seconds=(end_load-start_load)))}. Beginning training\033[37m")
            start_train = time.time()
            train(sim, z, train_loader, val_loader, observable_features_only=observable_features_only)
            end_train=time.time()
            print(f"Training lasted {str(timedelta(seconds=(end_train-start_train)))}")

    # Finishing ------
    end_whole = time.time()
    print(f"This program took {str(timedelta(seconds=(end_whole-start_whole)))}")