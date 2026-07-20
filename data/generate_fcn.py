# ------------------
# Imports
# ------------------
# general
import time
start_import = time.time()

import pickle
from pathlib import Path
from datetime import timedelta
from typing import List, Dict
import logging

# external
import numpy as np
import pandas as pd
import torch
from itertools import product

# internal imports
from DeepHalos.utils_old.__old_init__ import read_cat_data, Config, get_logger, Simulation
from DeepHalos.utils_old.__old_init__ import SIMS, REDSHIFTS

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
    if (cfg.hm_present and cfg.sim==Simulation.TNG):
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

def main() -> None:
    cfg = Config()
    
    logger = get_logger("FCN_Data")
    logger.info(f"FCN data generator is using device: {device}")
    logger.info(f"Imports took {timedelta(end_import-start_import)}")
    
    hm_z0 = [True, False]
    
    for sim in SIMS:
        for z in REDSHIFTS:
            try:
                cfg.sim = sim
                cfg.z = z
                cfg.hm_present = hm_present
            
            
            logger.info(f"\n\n{'='*50}")
            logger.info(f"Processing suite={cfg.sim}, z={cfg.z}")
            logger.info(f"{'='*50}\n")
            t1 = time.time()
            
            # ----- generates halos -----
            cat = read_cat_data(cfg)
            halos = make_halo_array(cat)
            # Ensures that if data is generated again, it will overwrite the previous file
            with open(f"{cfg.output_path}/{sim}_z{z}_halos.pkl", "wb") as file:
                pickle.dump(halos, file)

            # ----- generates summary statistics -----
            # Processes ~50,000 halos/sec
            add_z0 = (cfg.hm_present and sim==Simulation.TNG)
            summary_stats = [sum_stats(h, add_z0=add_z0) for h in halos]
        
            df = pd.DataFrame(summary_stats)
            df.to_csv(f"{cfg.output_path}/{sim}_z{z}_summaries.csv", index=False)
            means = df.mean()
            stds = df.std()
            stats_df = pd.DataFrame({"mean": means, "std": stds})
            logger.debug(df["HaloMass"].iloc[:10])
            stats_df.to_csv(f"{cfg.output_path}/{sim}_z{z}_stats.csv")
            
            logger.info(f"Halo processing took {timedelta(time.time()-t1)}")

            
    logger.info(f"FCN data generation completed in {timedelta(time.time()-start_import)}")
# ------------------
# Main Processing
# ------------------
if __name__ == "__main__":
    main()