# ------------------
# Imports
# ------------------
# general
import os
from data_utils import get_split_indices
import numpy as np

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
print(
    f"Imports took {str(timedelta(seconds=(end_import-start_import)))}. This program is using: {device}"
)

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
checkpoint_path = Path("/n/holystore01/LABS/itc_lab/Lab/galaxyGNN/models/")

# ------------------
# Functions
# ------------------


def load_data(
    datadir,
    z,
    sim='TNG',
    train_sim=None,
    observable_features_only=False,
    batch_size=64,
    target='HaloMass',
):
    """
    filenames[sim][z] -> gives simulation suite and redshift for this particular dataset
    """
    if train_sim is None:
        train_sim = sim
    data = pd.read_csv(datadir / f"{sim}_z{z}/{sim}_z{z}_raw.csv")
    train_stats = pd.read_csv(datadir / f"{sim}_z{z}/{sim}_z{z}_stats.csv", index_col=0)

    # Get means and stds from training data
    train_means = train_stats["mean"]
    train_stds = train_stats["std"]

    non_observable_features = [
        "HaloMass",
        "GalaxyMass_Sum",
        "GalaxyMass_Max",
        "GalaxyMass_Mean",
        "SFR_Sum",
        "SFR_Max",
        "SFR_Mean",
        "Velocity_Dispersion",
        "Velocity_Max",
        "Velocity_Mean",
    ]
    if sim == 'TNG':
        non_observable_features.append('HaloMass_z0')

    if observable_features_only:
        features_to_drop = non_observable_features
    else:
        features_to_drop = ['HaloMass']
        if sim == 'TNG':
            features_to_drop.append('HaloMass_z0')
    
    def standardize_data(data):
        return (data - train_means) / train_stds

    def create_dataloader(data_split, shuffle):
        standardized_data = standardize_data(data_split)
        features = standardized_data.drop(columns=features_to_drop)
        targets = standardized_data[target]

        dataset = TensorDataset(
            torch.tensor(features.to_numpy(), dtype=torch.float32),
            torch.tensor(targets.to_numpy(), dtype=torch.float32).unsqueeze(-1),
        )

        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    train_idx, val_idx, test_idx = get_split_indices(len(data), test_size=0.1 if sim=='TNG' else 0.01, val_size=0.05)

    
    train_data = data.iloc[train_idx]
    val_data = data.iloc[val_idx]
    test_data = data.iloc[test_idx]


    train_loader = create_dataloader(train_data, shuffle=True)
    val_loader = create_dataloader(val_data, shuffle=False)
    test_loader = create_dataloader(test_data, shuffle=False)

    return train_loader, val_loader, test_loader


def train(sim, z, train_loader, test_loader, observable_features_only=False, target='HaloMass'):
    run_name = (
        f"FCN_{sim}_z{z}_target_{target}"
        if not observable_features_only
        else f"FCN_{sim}_z{z}_target_{target}_observable_features_only"
    )
    wandb_logger = WandbLogger(
        log_model=False,
        project=run_name,
    )

    best_check = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename=f"best_fcn_{sim}z{z}"
        "_model-{step:02d}-{val_loss:.2f}-{mse_loss:.2f}",
        dirpath=checkpoint_path / run_name,
        verbose=True,
    )
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=50,  # Number of checks with no improvement before stopping
        mode="min",
        verbose=True,
    )

    print("Checkpoint path = ", checkpoint_path / run_name)
    trainer = L.Trainer(
        max_steps=200_000,
        logger=wandb_logger,
        gradient_clip_val=1.0,
        val_check_interval=0.5 if sim == "TNG" else 0.05,
        callbacks=[best_check, early_stop],
        default_root_dir=checkpoint_path / run_name,
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
    start_whole = time.time()
    datadir = Path("/n/holystore01/LABS/itc_lab/Lab/galaxyGNN/carol_processed_data/baseline")
    zs = [3,]
    target = 'HaloMass_z0'
    if target == 'HaloMass':
        sims = ['TNG', 'ASTRID']
    elif target == 'HaloMass_z0':
        sims = ['TNG']
    for observable_features_only in [False, True]:
        for sim in sims:
            for z in zs:
                # LOADING DATA
                start_load = time.time()
                print(f"\033[35mLoading data for {sim} at z={z}\033[37m")
                train_loader, val_loader, _ = load_data(
                    datadir=datadir,
                    sim=sim,
                    z=z,
                    observable_features_only=observable_features_only,
                    target=target,
                )
                end_load = time.time()

                # TRAINING
                print(
                    f"\033[35mLoading lasted {str(timedelta(seconds=(end_load-start_load)))}. Beginning training\033[37m"
                )
                start_train = time.time()
                train(
                    sim,
                    z,
                    train_loader,
                    val_loader,
                    observable_features_only=observable_features_only,
                    target=target,
                )
                end_train = time.time()
                print(
                    f"Training lasted {str(timedelta(seconds=(end_train-start_train)))}"
                )

    # Finishing ------
    end_whole = time.time()
    print(
        f"The fully connected program took {str(timedelta(seconds=(end_whole-start_whole)))}"
    )
