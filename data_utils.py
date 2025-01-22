import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

def read_data(data_path: Path, min_mass: float = 1e7) -> np.ndarray:
    """Load and filter galaxy catalog."""
    data = np.load(data_path)
    mask = (data['GalaxyMass'] >= min_mass) 
    return data[mask]

def get_split_indices(length, random_state=42, test_size=0.1, val_size=0.05):
    indices = np.arange(length)
    train_idx, test_idx = train_test_split(
        indices, random_state=random_state, test_size=test_size,
    )
    train_idx, val_idx = train_test_split(
        train_idx, random_state=random_state, test_size=val_size,
    )
    return train_idx, val_idx, test_idx