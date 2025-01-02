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

from baseline_model import LinearModel
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
data_path = Path('/n/home03/hbrittain/data_outs/baseline/')

filenames = {
    # 'TNG': {
    #     # 6: data_path / 'TNG_z6/TNG_z6',
    #     # 5: data_path / 'TNG_z5/TNG_z5',
    #     4: data_path / 'TNG_z4/TNG_z4'
    # },
    'ASTRID': {
        # 6: data_path / 'ASTRID_z6/ASTRID_z6',
        5: data_path / 'ASTRID_z5/ASTRID_z5',
        4: data_path / 'ASTRID_z4/ASTRID_z4',
        3: data_path / 'ASTRID_z3/ASTRID_z3'
    } 
}

# ------------------
# Functions
# ------------------
def load_data(datadir):
    '''
    filenames[sim][z] -> gives simulation suite and redshift for this particular dataset
    '''
    data = pd.read_csv(f'{datadir}_normalized.csv')
    # for some reason redshift screws things up, but we want to preserve this for later
    data = data.drop(columns='Z') 
    
    # # uncomment to limit data
    # limit = 0.2
    # data = data.sample(frac=limit, random_state=42)

    train, test = train_test_split(data, random_state=42)

    # train
    feat_train = train.drop(columns=['HaloMass']) 
    targ_train = train['HaloMass']
    train_dataset = TensorDataset(
        torch.tensor(feat_train.to_numpy(), dtype=torch.float32), 
        torch.tensor(targ_train.to_numpy(), dtype=torch.float32).unsqueeze(-1))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # test
    feat_test = test.drop(columns=['HaloMass'])
    targ_test = test['HaloMass']
    test_dataset = TensorDataset(
        torch.tensor(feat_test.to_numpy(), dtype=torch.float32), 
        torch.tensor(targ_test.to_numpy(), dtype=torch.float32).unsqueeze(-1))
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, test_loader

def train(sim, z, train_loader, test_loader):
    wandb_logger = WandbLogger(
        log_model=False, 
        project=f'FCN_{sim}_z{z}'
        )
    
    best_check = ModelCheckpoint(
        monitor="val_loss",         
        mode="min",                  
        save_top_k=1,  
        filename=f"best_fcn_{sim}z{z}""_model-{step:02d}-{val_loss:.2f}-{mse_loss:.2f}", 
        dirpath=f'/n/home03/hbrittain/networks/LIGHTNING_FILES/FCN_{sim}_z{z}',  
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
        default_root_dir='/n/home03/hbrittain/networks/LIGHTNING_FILES/',
        enable_progress_bar=False
        )
    
    num_features = num_features = next(iter(test_loader))[0].shape[1]
    model = LinearModel(context=num_features)
    
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
    
    for sim, data in filenames.items():
        for z, filepath in data.items():
            # LOADING DATA
            start_load = time.time()
            print(f"\033[35mLoading data for {sim} at z={z}\033[37m")
            train_loader, test_loader = load_data(datadir=filenames[sim][z])
            end_load = time.time()
            
            # TRAINING
            print(f"\033[35mLoading lasted {str(timedelta(seconds=(end_load-start_load)))}. Beginning training\033[37m")
            start_train = time.time()
            train(sim, z, train_loader, test_loader)
            end_train=time.time()
            print(f"Training lasted {str(timedelta(seconds=(end_train-start_train)))}")

    # Finishing ------
    end_whole = time.time()
    print(f"The fully connected program took {str(timedelta(seconds=(end_whole-start_whole)))}")