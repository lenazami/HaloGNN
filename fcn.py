# ------------------
# Imports
# ------------------
# general
import os
print(f"{os.path.basename(__file__)} is running")
import time
start_import = time.time()
from datetime import timedelta, datetime as dt
from pathlib import Path
import pandas as pd
# torch
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
import wandb

from fcn_model import FlowModel
end_import = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Imports took {str(timedelta(seconds=(end_import-start_import)))}. This program is using: {device}")

# ------------------
# Dictionaries and Global Variables
# ------------------
'''
Stores redshifts, sim suites, and filepaths for data
Sorted by file size/galaxy count
'''
data_path = Path('/n/holystore01/LABS/itc_lab/Lab/galaxyGNN/carol_processed_data/baseline')
checkpoint_path = Path('/n/holystore01/LABS/itc_lab/Lab/galaxyGNN/models/')

filenames = {
    'TNG': {
        # 6: data_path / 'TNG_z6/TNG_z6',
        # 5: data_path / 'TNG_z5/TNG_z5',
        4: data_path / 'TNG_z4/TNG_z4'
    },
    'ASTRID': {
        # 6: data_path / 'ASTRID_z6/ASTRID_z6',
        #5: data_path / 'ASTRID_z5/ASTRID_z5',
        4: data_path / 'ASTRID_z4/ASTRID_z4',
        #3: data_path / 'ASTRID_z3/ASTRID_z3'
    } 
}

# ------------------
# Functions
# ------------------
def load_data(datadir, trainsim_datadir=None,observable_features_only=False, batch_size=64,):
    '''
    filenames[sim][z] -> gives simulation suite and redshift for this particular dataset
    '''
    if trainsim_datadir is None:
        trainsim_datadir = datadir
    data = pd.read_csv(f'{datadir}_raw.csv')
    train_stats = pd.read_csv(f'{trainsim_datadir}_stats.csv', index_col=0)
    
    # Get means and stds from training data
    train_means = train_stats['mean']
    train_stds = train_stats['std']
    
    non_observable_features = [
        'HaloMass',  
        'GalaxyMass_Sum', 
        'GalaxyMass_Max', 
        'GalaxyMass_Mean', 
        'SFR_Sum', 
        'SFR_Max', 
        'SFR_Mean', 
        'Velocity_Dispersion', 
        'Velocity_Max', 
        'Velocity_Mean'
    ]
    features_to_drop = non_observable_features if observable_features_only else ['HaloMass']
    def standardize_data(data):
        return (data - train_means) / train_stds

    def create_dataloader(data_split, shuffle):
        standardized_data = standardize_data(data_split)
        features = standardized_data.drop(columns=features_to_drop)
        targets = standardized_data['HaloMass']
        
        dataset = TensorDataset(
            torch.tensor(features.to_numpy(), dtype=torch.float32),
            torch.tensor(targets.to_numpy(), dtype=torch.float32).unsqueeze(-1)
        )
        
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    train_data, test_data = train_test_split(data, random_state=42, test_size=0.1)
    train_data, val_data = train_test_split(train_data, random_state=42, test_size=0.1)
    train_loader = create_dataloader(train_data, shuffle=True)
    val_loader = create_dataloader(val_data, shuffle=False)
    test_loader = create_dataloader(test_data, shuffle=False)
    
    return train_loader, val_loader, test_loader  


def train(sim, z, train_loader, test_loader, observable_features_only=False):
    run_name = f'FCN_{sim}_z{z}' if not observable_features_only else f'FCN_{sim}_z{z}_observable_features_only'
    wandb_logger = WandbLogger(
        log_model=False, 
        project=run_name,
        )
    
    best_check = ModelCheckpoint(
        monitor="val_loss",         
        mode="min",                  
        save_top_k=1,  
        filename=f"best_fcn_{sim}z{z}""_model-{step:02d}-{val_loss:.2f}-{mse_loss:.2f}", 
        dirpath = checkpoint_path / run_name,
        verbose=True                 
    )
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=50,  # Number of checks with no improvement before stopping
        mode="min",
        verbose=True
    )

    trainer = L.Trainer(
        max_steps=200_000,
        logger=wandb_logger, 
        gradient_clip_val=1.0,
        val_check_interval=0.5,
        callbacks=[best_check, early_stop],
        default_root_dir=checkpoint_path / run_name,
        enable_progress_bar=False
        )
    
    num_features = next(iter(test_loader))[0].shape[1]
    model = FlowModel(context=num_features)
    
    trainer.fit(model=model, 
                train_dataloaders=train_loader, 
                val_dataloaders=test_loader)

    print(f"Best  model saved at: {best_check.best_model_path}")
    wandb.finish()

# ------------------
# Main Training
# ------------------

if __name__ == '__main__':
    start_whole = time.time()
    for observable_features_only in [True, False]:
        for sim, data in filenames.items():
            for z, filepath in data.items():
                # LOADING DATA
                start_load = time.time()
                print(f"\033[35mLoading data for {sim} at z={z}\033[37m")
                train_loader, val_loader, _ = load_data(
                    datadir=filenames[sim][z], 
                    observable_features_only=observable_features_only,
                )
                end_load = time.time()
                
                # TRAINING
                print(f"\033[35mLoading lasted {str(timedelta(seconds=(end_load-start_load)))}. Beginning training\033[37m")
                start_train = time.time()
                train(sim, z, train_loader, val_loader)
                end_train=time.time()
                print(f"Training lasted {str(timedelta(seconds=(end_train-start_train)))}")

    # Finishing ------
    end_whole = time.time()
    print(f"The fully connected program took {str(timedelta(seconds=(end_whole-start_whole)))}")