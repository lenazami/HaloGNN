# utils / io.py

# -----------------
# Imports
# -----------------
import numpy as np
import torch
from pathlib import Path
from typing import List, Literal, Optional

from .config import Config, ModelType

device = "cuda" if torch.cuda.is_available() else "cpu"

FILE_SUFFIXES = {
    ModelType.GNN: {"data": "graphs.pt", "stats": "stats.pt"},
    ModelType.FCN: {"data": "raw.csv", "stats": "stats.csv", "halos": "halos.pkl"},
}

# ------------------
# Functions
# ------------------
def get_file(cfg: Config, file_type: Literal["data", "stats", "halos"], create_dir: bool = False) -> Path:
    """Get file path based on config and data type."""
    if file_type=="data":
        search_dir = cfg.get_data_path()
    else:
        search_dir = cfg.get_stats_path()
        
    # create dir if it doesnt exist
    search_dir.mkdir(parents=True, exist_ok=True)
    
    # look for file
    pattern = FILE_SUFFIXES[cfg.model_type][file_type]
    candidates = list(search_dir.glob(f"*{pattern}"))
    if not candidates:
        raise FileNotFoundError(
            f"No {file_type} file found in {search_dir}\n"
            f"Looking for pattern: {pattern}\n"
            f"Run data generation script first!"
        )
    return candidates[0]

def save_file(data, cfg: Config, file_type: Literal["data", "stats"], filename: Optional[str] = None):
    """Save data to appropriate location."""
    if file_type == "data":
        save_dir = cfg.get_data_path()
    else:  # stats
        save_dir = cfg.get_stats_path()
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # default filenames
    if filename is None:
        if cfg.model_type == ModelType.FCN:
            filename = "raw.csv" if file_type == "data" else "stats.csv"
        else:  # GNN
            filename = "graphs.pt" if file_type == "data" else "stats.pt"
    
    filepath = save_dir / filename
    
    # Save based on type
    if filepath.suffix == ".csv":
        data.to_csv(filepath, index=(file_type == "stats"))
    elif filepath.suffix == ".pt":
        torch.save(data, filepath)
    else:
        raise ValueError(f"Unknown file type: {filepath.suffix}")
    
    print(f"Saved {file_type} to: {filepath}")
    return filepath

def read_cat_data(cfg: Config) -> np.ndarray:
        """Load and filter galaxy catalog by galmass."""
        cat_path = cfg.get_catalogue_path()
        print(f"Loading catalogue from: {cat_path}")
        
        arr = np.load(cat_path, allow_pickle=True)
        mask = arr["GalaxyMass"] >= cfg.min_mass
        return arr[mask]
    
def convert_to_float32(data_list: List[dict]) -> List[dict]:
    """Casts all tensor-like values in each dict to float32 on 'device',
    skips central_mask, edge_index, feature_names."""
    exclude = {"central_mask", "edge_index", "feature_names"}
    for d in data_list:
        for key, val in list(d.items()):
            if key in exclude:
                continue
            if isinstance(val, torch.Tensor):
                if val.dtype != torch.float32:
                    d[key] = val.to(device).float()
            else:
                d[key] = torch.tensor(val, dtype=torch.float32, device=device)
    return data_list