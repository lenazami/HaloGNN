# ------------------
# Imports
# ------------------
# general
import os
print(f"{os.path.basename(__file__)} is running")
import time
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
from sklearn.model_selection import train_test_split
import wandb
# model
from git_folder.gnn_model import GraphModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"This program is using: {device}")

# ------------------
# Dictionaries and Global Variables
# ------------------
'''
Stores redshifts, sim suites, and filepaths for training data
Sorted by file size/galaxy count
'''
data_dir = Path('/n/home03/hbrittain/data_outs/gnn/')

filenames = {
    'TNG': {
        # 6: data_dir / 'TNG_z6/TNG_z6_all_graphs.pt',
        # 5: data_dir / 'TNG_z5/TNG_z5_all_graphs.pt',
        4: data_dir / 'TNG_z4/TNG_z4_all_graphs.pt'
    },
    'ASTRID': {
        # 6: data_dir / 'ASTRID_z6/ASTRID_z6_all_graphs.pt',
        # 5: data_dir / 'ASTRID_z5/ASTRID_z5_all_graphs.pt',
        4: data_dir / 'ASTRID_z4/ASTRID_z4_all_graphs.pt',
        # 3: data_dir / 'ASTRID_z3/ASTRID_z3_all_graphs.pt'
    } 
}

# ------------------
# Functions
# ------------------
def load_data(datadir, batch_size=64, limit_data=False):
    '''
    filenames[sim][z] -> gives simulation suite and redshift for this particular dataset
    '''
    data = torch.load(datadir)
    
    if limit_data == True:
        limit = 0.2
        data = data.sample(frac=limit, random_state=42)
        
        batched_data = Batch.from_data_list(data)
        firstpart = batched_data.x[:,:1] #masks out gmass and velocities; as well as the superfluous positions
        secpart = batched_data.x[:,-12:]
        combine = torch.cat((firstpart, secpart), dim=1)
        batched_data.x = combine
        data = batched_data.to_data_list()
    
    train, test = train_test_split(data, test_size=0.2, random_state=42)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def train(sim, z, train_loader, test_loader):
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
        node_features_dim=13
    )
    
    wandb_logger = WandbLogger(
        log_model=False, 
        project=f'HaloGraph_{sim}_{z}')
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # save the best model
    best_check = ModelCheckpoint(
            monitor="val_loss",         
            mode="min",                  
            save_top_k=1,                
            filename=f"best_gnn_{sim}z{z}""_model-{step:02d}-{val_loss:.2f}-{mse_loss:.2f}", 
            dirpath=f'/n/home03/hbrittain/networks/LIGHTNING_FILES/GNN_{sim}_z{z}',
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
        max_steps=100_000,
        logger=wandb_logger, 
        val_check_interval=0.5,
        log_every_n_steps=100,
        # check_val_every_n_epoch=None,
        gradient_clip_val=1.0,
        # limit_val_batches=0.,
        callbacks=[lr_monitor, best_check, early_stop],
        default_root_dir='/n/home03/hbrittain/networks/LIGHTNING_FILES/'
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
    for sim, data in filenames.items():
        for z, filepath in data.items():
            # LOADING DATA
            batch_size = 64
            start_load = time.time()
            print(f"\033[35mLoading data for {sim} at z={z}\033[37m")
            train_loader, test_loader = load_data(datadir=filenames[sim][z], batch_size=batch_size)
            end_load = time.time()
            
            # TRAINING
            print(f"\033[35mLoading lasted {str(timedelta(seconds=(end_load-start_load)))}. Beginning training\033[37m")
            start_train = time.time()
            train(sim, z, train_loader, test_loader)
            end_train=time.time()
            print(f"Training lasted {str(timedelta(seconds=(end_train-start_train)))}")

    # Finishing ------
    end_whole = time.time()
    print(f"This program took {str(timedelta(seconds=(end_whole-start_whole)))}")