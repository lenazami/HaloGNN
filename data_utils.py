import numpy as np
from pathlib import Path

def read_data(data_path: Path, min_mass: float = 1e7) -> np.ndarray:
    """Load and filter galaxy catalog."""
    data = np.load(data_path)
    mask = (data['GalaxyMass'] >= min_mass) 
    return data[mask]

