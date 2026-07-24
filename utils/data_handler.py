# utils / data_handlers.py
# from pathlib import Path
import numpy as np
from abc import ABC, abstractmethod
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch_geometric.loader import DataLoader as GeomDataLoader
from typing import Any, List, Optional
from operator import itemgetter

from .config import Config, ModelType
from .io import get_file, convert_to_float32

device = "cuda" if torch.cuda.is_available() else "cpu"

class FCNHandler(BaseDataHandler):
    """Handler for Fully Connected Network data."""
    
    OBSERVABLE_FEATURES = [
        "HaloMass", "GalaxyMass_Sum", "GalaxyMass_Max", "GalaxyMass_Mean",
        "SFR_Sum", "SFR_Max", "SFR_Mean", "Velocity_Dispersion",
        "Velocity_Max", "Velocity_Mean", "HaloMass_z0"
    ]
    
    def read_data(self, cfg: Config) -> pd.DataFrame:
        filepath = get_file(cfg, "data")
        return pd.read_csv(filepath)

    def load_stats(self, cfg: Config) -> pd.DataFrame:
        filepath = get_file(cfg, "stats")
        return pd.read_csv(filepath, index_col=0)
    
    def normalize(self, data: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
        cols = stats.index.tolist()
        cols = [c for c in cols if c in data.columns and c != "HaloMass_z0"]
        data[cols] = (data[cols] - stats.loc[cols, "mean"]) / stats.loc[cols, "std"]
        return data
    
    def denormalize(self, data: torch.Tensor, stats: pd.DataFrame, field: str) -> torch.Tensor:
        mean = stats.loc[field, "mean"]
        std = stats.loc[field, "std"]
        return data * std + mean
    
    def filter_observables(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.drop(columns=self.OBSERVABLE_FEATURES, errors="ignore")
    
    def create_dataloader(self, data: pd.DataFrame, cfg: Config,
                         batch_size: int, shuffle: bool,
                         indices: Optional[List[int]] = None) -> DataLoader:
        if indices is not None:
            data = data.iloc[indices]
        
        feature_cols = [c for c in data.columns if c not in [cfg.label_field(), cfg.feature_field()]]
        features = torch.tensor(data[feature_cols].values, dtype=torch.float32, device=device)
        targets = torch.tensor(data[cfg.label_field()].values, dtype=torch.float32, device=device).unsqueeze(-1)
        
        dataset = TensorDataset(features, targets)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

class GNNHandler(BaseDataHandler):
    """Handler for Graph Neural Network data."""
    
    OBSERVABLE_FEATURES = [
        "HaloMass", "GalaxyVel_1", "GalaxyVel_2", "GalaxyVel_3",
        "GalaxyRhalf", "GalaxyMass", "GalaxyVel", "SFR"
    ]
    
    def read_data(self, cfg: Config) -> List:
        filepath = get_file(cfg, "data")
        data = torch.load(filepath)
        return convert_to_float32(data)

    def load_stats(self, cfg: Config) -> dict:
        filepath = get_file(cfg, "stats")
        return torch.load(filepath)

    def normalize(self, data: List, stats: dict) -> List:
        feature_names = data[0].feature_names
        feature_indices = [
            stats["feature_names"].index(name)
            for name in feature_names
            if name in stats["feature_names"]
        ]

        means = torch.tensor(stats["means"], dtype=torch.float32, device=device)
        stds = torch.tensor(stats["stds"], dtype=torch.float32, device=device)
        halo_idx = stats["feature_names"].index("HaloMass")

        for graph in data:
            graph = graph.to(device)
            graph.x[:, feature_indices] = (graph.x[:, feature_indices] - means[feature_indices]) / stds[feature_indices]
            graph.y = (graph.y - means[halo_idx]) / stds[halo_idx]
            
        return data

    def denormalize(self, data: torch.Tensor, stats: dict, field: str) -> torch.Tensor:
        means = stats["means"].astype(np.float32)
        stds = stats["stds"].astype(np.float32)
        idx = stats["feature_names"].index(field)
        return data * stds[idx] + means[idx]
    
    # def denormalize(self, preds, cfg):
    #     halo_mass_idx = stats["feature_names"].index(cfg.label_field())

    def filter_observables(self, data: List) -> List:
        masked_data = []
        kept_indices = [
            i for i, name in enumerate(data[0].feature_names)
            if name not in self.OBSERVABLE_FEATURES
        ]
        
        for graph in data:
            temp = graph.clone()
            temp.x = graph.x[:, kept_indices]
            temp.feature_names = [graph.feature_names[i] for i in kept_indices]
            masked_data.append(temp)
        
        return masked_data
    
    def create_dataloader(self, data: List, cfg: Config,
                         batch_size: int, shuffle: bool,
                         indices: Optional[List[int]] = None) -> GeomDataLoader:
        if indices is not None:
            data = list(itemgetter(*indices)(data))
        return GeomDataLoader(data, batch_size=batch_size, shuffle=shuffle)

def get_handler(model_type: ModelType) -> BaseDataHandler:
    """Factory function to get appropriate handler."""
    if model_type == ModelType.FCN:
        return FCNHandler()
    elif model_type == ModelType.GNN:
        return GNNHandler()
    else:
        raise ValueError(f"Unknown model type: {model_type}")