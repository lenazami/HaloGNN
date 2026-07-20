'''
it was easier to just copypaste everything versus importing all the functions to match 
'''
# ------------------
# Imports
# ------------------
# general
import time
from pathlib import Path
import hydra
from omegaconf import DictConfig
from datetime import timedelta
import logging
from typing import List, Tuple, Dict, Optional, Any

# external
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    EarlyStopping,
)
import wandb

# local imports - assuming these exist in your codebase
from models.gnn_model import GraphModel
from DeepHalos.utils_old.__old_init__ import DataConfig, read_cat_data, _cfg_raw
from data.generate_gnn import find_neighbors, apply_periodic_boundary, process_features
from .gnn import train

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)

# ------------------
# GalaxyDataset Class
# ------------------
class GalaxyDataset(Dataset):
    def __init__(self, 
                 cfg: DataConfig, 
                 max_galaxies: int = 100,
                 transform=None, 
                 pre_transform=None, 
                 pre_filter=None):
        super().__init__(None, transform, pre_transform, pre_filter)
        self.cfg = cfg
        self.max_galaxies = max_galaxies
        
        # Load catalog data
        logger.info(f"Loading catalog data for {cfg.sim} at z={cfg.z}")
        self.cat_data = read_cat_data(cfg)
        
        # Find central galaxies and their neighbors
        logger.info("Finding central galaxies and their neighbors")
        self.central_idx, self.neighbor_idx = find_neighbors(self.cat_data, cfg)
        
        # Process features for all galaxies
        logger.info("Processing galaxy features")
        self.halo_masses, self.halo_masses_z0, self.positions, self.features, self.feature_names = process_features(self.cat_data, cfg.box_size)
        
        if self.cfg.observables_only:
            obs_indices = [i for i, name in enumerate(self.feature_names) 
                          if name in ["GalaxyMass", "GalaxyRhalf", "SFR", 
                                     "jwst_f090w", "jwst_f150w", "jwst_f277w", "jwst_f444w"]]
            self.features = self.features[:, obs_indices]
            self.feature_names = [self.feature_names[i] for i in obs_indices]
            
        # Compute overall dataset statistics
        self._compute_statistics()
        
        logger.info(f"Dataset initialized with {len(self)} galaxy groups")

    def _compute_statistics(self):
        """Compute dataset statistics for normalization."""
        # Compute feature statistics
        self.features_means = self.features.mean(axis=0)
        self.features_stds = self.features.std(axis=0)
        
        # Compute halo mass statistics for central galaxies
        central_halo_masses = self.halo_masses[self.central_idx]
        self.halo_masses_mean = central_halo_masses.mean()
        self.halo_masses_std = central_halo_masses.std()
        
        # Store stats dictionary
        self.stats = {
            "feature_names": self.feature_names + ["HaloMass"],
            "means": np.append(self.features_means, self.halo_masses_mean),
            "stds": np.append(self.features_stds, self.halo_masses_std),
        }
        
        # Add z0 stats if available
        if self.halo_masses_z0 is not None:
            central_halo_masses_z0 = self.halo_masses_z0[self.central_idx]
            self.halo_masses_z0_mean = central_halo_masses_z0.mean()
            self.halo_masses_z0_std = central_halo_masses_z0.std()
            
            self.stats["feature_names"] = self.feature_names + ["HaloMass", "HaloMass_z0"]
            self.stats["means"] = np.append(
                self.stats["means"],
                self.halo_masses_z0_mean
            )
            self.stats["stds"] = np.append(
                self.stats["stds"],
                self.halo_masses_z0_std
            )

    def len(self):
        """Return the number of graphs in the dataset."""
        return len(self.central_idx)

    def get(self, idx: int) -> Data:
        """Get a single graph by index."""
        c_idx = self.central_idx[idx]
        nbrs = self.neighbor_idx[idx]
        graph_idx = [c_idx] + nbrs
        n = len(graph_idx)
        
        # Create central galaxy mask
        central_mask = torch.zeros(n, dtype=torch.bool)
        central_mask[0] = True
        
        # Build fully-connected edge_index just once per graph
        row, col = np.triu_indices(n, k=1)
        edge_index = torch.tensor(
            np.vstack([np.concatenate([row, col]),
                       np.concatenate([col, row])]),
            dtype=torch.long
        )
        
        # Calculate edge features (distances between pairs)
        idx_arr = np.array(graph_idx)
        diffs = apply_periodic_boundary(
            self.positions[idx_arr[row]] - self.positions[idx_arr[col]],
            self.cfg.box_size
        )
        
        # Create edge attributes (normalized distances)
        distances = np.linalg.norm(diffs, axis=1)
        norm_diffs = diffs / (2.0 * self.cfg.graph_radius)
        
        # Create full edge attributes: [delta_x, delta_y, delta_z, distance]
        edge_attr = torch.tensor(
            np.hstack([norm_diffs, distances.reshape(-1, 1) / (2.0 * self.cfg.graph_radius)]),
            dtype=torch.float32
        )
        
        # Duplicate edge attributes for symmetrical edges
        edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
        
        # Use the target specified in the config
        target = self.halo_masses_z0[c_idx] if self.cfg.hm_present else self.halo_masses[c_idx]
        
        return Data(
            x=torch.tensor(self.features[idx_arr], dtype=torch.float32),
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor(target, dtype=torch.float32),
            central_mask=central_mask,
            feature_names=self.feature_names,
            global_attr=torch.tensor([n / self.max_galaxies], dtype=torch.float32),
        )

    def get_stats(self) -> Dict[str, Any]:
        """Return dataset statistics."""
        return self.stats

# ------------------
# Main Function
# ------------------
def main():
    print(f"Running on device: {device}")
    start_time = time.time()
    
    # sets simulation and redshift 
    sim = "ASTRID"
    z = 4
    
    # List of radius cuts to try
    radius_cuts = [1000.0, 1500.0, 2000.0, 5000.0]
    
    for radius in radius_cuts:
        logger.info(f"\n\n{'='*50}")
        logger.info(f"Training with radius = {radius:.1f}")
        logger.info(f"{'='*50}\n")
        
        # data config
        dat_cfg = DataConfig(
            root=_cfg_raw["root"],
            model_type="GNN",
            sim=sim,
            z=z,
            graph_radius=radius,
            observables_only=_cfg_raw['observables_only'],
            hm_present=_cfg_raw['hm_present']
        )
        
        # make dataset
        dataset = GalaxyDataset(dat_cfg)
        
        # split
        dataset_size = len(dataset)
        train_size = int(0.7 * dataset_size)
        val_size = int(0.15 * dataset_size)
        test_size = dataset_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        # cant load the .pt bc 5_000,10_000 are too big to generate
        train_loader = DataLoader(
            train_dataset, 
            batch_size=_cfg_raw['batch_size'], 
            shuffle=True, 
            # num_workers=cfg.num_workers
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=_cfg_raw['batch_size'], 
            shuffle=False, 
            # num_workers=cfg.num_workers
        )
        
        _ = DataLoader(
            test_dataset, 
            batch_size=_cfg_raw['batch_size'], 
            shuffle=False, 
            # num_workers=cfg.num_workers
        )
        
        logger.info(f"Dataset splits: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
        
        # Train the model
        train_start = time.time()
        _, _ = train(
            dat_cfg, 
            train_loader, 
            val_loader
        )
        train_duration = time.time() - train_start
        
        logger.info(f"Training for radius {radius:.1f} completed in {timedelta(seconds=train_duration)}")
        
        # Save dataset statistics
        stats_path = dat_cfg.out_root / f"{sim}_z{z}_r{radius:.1f}_feature_stats.pt"
        torch.save(dataset.get_stats(), stats_path)
        logger.info(f"Stats saved to {stats_path}")
        
    total_duration = time.time() - start_time
    logger.info(f"\nTotal runtime: {timedelta(seconds=total_duration)}")

if __name__ == "__main__":
    main()