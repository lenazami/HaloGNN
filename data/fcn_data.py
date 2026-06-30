# data / fcn_data.py

# ------------------
# Imports
# ------------------
import time
start_import = time.time()

from typing import List
from datetime import timedelta
import logging


# external
import torch
import numpy as np

# internal
from utils_old.logger import get_logger
from utils_old.config import Config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)
end_import = time.time()

# ------------------
# Functions
# ------------------

def make_halo_array(data: np.ndarray) -> List[np.ndarray]:
    """
    Takes in a data array and splits array when FOFID changes
    (catalogues are sorted by FOFID)
    """
    # On the off chance it is not sorted, this shouldn't take more than 5 seconds
    sorted_data = np.sort(data, order="FOFID")
    boundaries = np.flatnonzero(np.diff(sorted_data["FOFID"])) + 1
    return np.split(sorted_data, boundaries)

def sum_stats(halo, add_z0=False):
    """
    Computes summary statistics for a group of galaxies within a halo.
    Returns a dictionary of features, making it easy to add/remove features
    while maintaining consistency in the data pipeline.

    Parameters:
        halo: Structured numpy array containing galaxy properties within a halo

    Returns:
        dict: Dictionary of feature names and their values
    """
    MIN_SFR = 1e-5
    halo["SFR"] = np.where(halo["SFR"] == 0.0, MIN_SFR, halo["SFR"])
    
    # Mass features
    features = {
        "HaloMass": np.log10(halo["HaloMass"].max()),
        "GalaxyMass_Sum": np.log10(halo["GalaxyMass"].sum()),
        "GalaxyMass_Max": np.log10(halo["GalaxyMass"].max()),
        "GalaxyMass_Mean": np.log10(halo["GalaxyMass"].mean()),
    }
    if add_z0:
        features.update({"HaloMass_z0": np.log10(halo["HaloMass_z0"].max())})

    # Star formation features
    features.update({
        "SFR_Sum": np.log10(halo["SFR"].sum()),
        "SFR_Max": np.log10(halo["SFR"].max()),
        "SFR_Mean": np.log10(halo["SFR"].mean()),
    })
    
    # Velocity features
    features.update({
        "Velocity_Dispersion": halo["GalaxyVel"].std(),
        "Velocity_Max": halo["GalaxyVel"].max(),
        "Velocity_Mean": halo["GalaxyVel"].mean(),
    })
    
    # JWST photometry features
    for band, band_name in [
        ("jwst_f090w", "F090W"),
        ("jwst_f150w", "F150W"),
        ("jwst_f277w", "F277W"),
        ("jwst_f444w", "F444W"),
    ]:
        vals = halo[band]
        features.update({
            f"{band_name}_Sum": vals.sum(),
            f"{band_name}_Max": vals.max(),
            f"{band_name}_Mean": vals.mean(),
        })
    
    # Structural features
    features.update({
        "N_Galaxies": halo["GalaxyMass"].shape[0],
    })
    return features

def main():
    cfg = Config()
    
    logger = get_logger("FCN_Data")
    logger.info(f"FCN data generator is using device: {device}")
    logger.info(f"Imports took {timedelta(end_import-start_import)}")
    
    
    
    