# utils / data_utils.py

# -----------------
# Imports
# -----------------
# general
import numpy as np
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GeomDataLoader
from sklearn.model_selection import train_test_split
from typing import Tuple, Union

from .config import Config, Simulation
from .data_handler import get_handler


# -----------------
# Functions
# -----------------
# ----- feature processing -----
def get_split_indices(
    length: int, 
    test_size: float = 0.1, 
    val_size: float = 0.05,
    random_state: int = 42
) -> Tuple:
    """Generate train/val/test split indices."""
    indices = np.arange(length)
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=random_state)
    train_idx, val_idx = train_test_split(train_idx, test_size=val_size, random_state=random_state)
    return train_idx, val_idx, test_idx

def load_data(
    cfg: Config,
    batch_size: int = 64,
    only_test: bool = False,
    all_data: bool = False,
) -> Union[DataLoader, Tuple[DataLoader, DataLoader, DataLoader]]:
    """
    Unified data loading function for both FCN and GNN.
    
    Returns:
        Single DataLoader if all_data=True or only_test=True,
        otherwise returns (train_loader, val_loader, test_loader)
    """
    # get appropriate handler
    handler = get_handler(cfg.model_type)
    
    # load and normalize data
    data = handler.read_data(cfg)
    stats = handler.load_stats(cfg)
    data = handler.normalize(data, stats)
    
    # filter observables
    if cfg.observables_only:
        data = handler.filter_observables(data)

    # use all data--minimizes overhead of splitting
    if all_data:
        return handler.create_dataloader(data, cfg, batch_size, shuffle=False)
    
    # split data
    test_size = 0.1 if cfg.sim == Simulation.TNG else 0.01
    train_idx, val_idx, test_idx = get_split_indices(len(data), test_size=test_size, val_size=0.05)
    
    # create test loader
    test_loader = handler.create_dataloader(data, cfg, batch_size, shuffle=False, indices=test_idx)
    
    # if testing, only return the test loader; minimizes overhead
    if only_test:
        return test_loader
    
    # else, also return train and val loaders
    train_loader = handler.create_dataloader(data, cfg, batch_size, shuffle=True, indices=train_idx)
    val_loader = handler.create_dataloader(data, cfg, batch_size, shuffle=False, indices=val_idx)

    return train_loader, val_loader, test_loader