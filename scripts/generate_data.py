# scripts / generate_data.py
'''data generation!'''

import time

start_import = time.time()

from datetime import timedelta
from typing import List
import pickle

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from torch_geometric.data import Data

from utils import Config, ModelType, Simulation
from utils.io import read_cat_data, save_file

end_import = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------
# FCN
# ------------
def make_halo_array(data: np.ndarray) -> List[np.ndarray]:
    """splits array when FOFID changes"""
    # On the off chance it is not sorted, this shouldn't take more than 5 seconds
    sorted_data = np.sort(data, order="FOFID")
    boundaries = np.flatnonzero(np.diff(sorted_data["FOFID"])) + 1
    return np.split(sorted_data, boundaries)

def sum_stats(halo, include_z0=False):
    """
    Compute summary statistics for a halo.
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
    if include_z0:
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

def generate_fcn_data(cfg: Config):
    '''Generate FCN data for a given configuration.'''
    print(f"Generating FCN data for {cfg.sim.value} z={cfg.z}")
    t1 = time.time()
    # ----- generates halos -----
    cat = read_cat_data(cfg)
    halos = make_halo_array(cat)
    
    # Ensures that if data is generated again, it will overwrite the previous file
    halo_file = cfg.get_data_path() / f"{cfg.sim.value}_z{cfg.z}_halos.pkl"
    halo_file.parent.mkdir(parents=True, exist_ok=True)
    with open(halo_file, "wb") as f:
        pickle.dump(halos, f)
    print(f"Saved {len(halos)} halos")

    # ----- generates summary statistics -----
    # Processes ~50,000 halos/sec
    include_z0 = (cfg.hm_present and cfg.sim==Simulation.TNG)
    summary_stats = [sum_stats(h, include_z0=include_z0) for h in halos]

    # ----- save data and stats -----
    df = pd.DataFrame(summary_stats)
    save_file(df, cfg, "data", f"{cfg.sim.value}_z{cfg.z}_raw.csv")
    
    stats_df = pd.DataFrame({"mean": df.mean(), "std": df.std()})
    save_file(stats_df, cfg, "stats", f"{cfg.sim.value}_z{cfg.z}_stats.csv")
    
    print(f"Halo processing took {timedelta(time.time()-t1)}")
    
# ------------
# GNN
# ------------     

def find_neighbors(cat_data: np.ndarray, cfg: Config) -> tuple:
    """Find central galaxies and their neighbors."""
    # Find central galaxies (highest mass in each group)
    df = pd.DataFrame(
        {
            "GalaxyMass": cat_data["GalaxyMass"],
            "HaloMass": cat_data["HaloMass"],
            "FOFID": cat_data["FOFID"],
        }
    )
    # Tricky, use GalaxyMass instead? but want to match dataset fcn
    central_idx = df.groupby("FOFID")["HaloMass"].idxmax().values
    # Find neighbors using periodic boundary conditions
    positions = cat_data["GalaxyPos"] % cfg.box_size
    kdtree = cKDTree(positions, boxsize=cfg.box_size)
    neighbor_idx = kdtree.query_ball_point(positions[central_idx], cfg.graph_radius)
    
    return central_idx, neighbor_idx

def process_galaxy_features(cat_data: np.ndarray, cfg: Config) -> tuple:
    """Extract and process galaxy features."""
    feats = [
        ("HaloMass", np.log10(cat_data["HaloMass"])),
        ("GalaxyMass", np.log10(cat_data["GalaxyMass"])),
        ("GalaxyPos_1", cat_data["GalaxyPos"][:, 0] % cfg.box_size),
        ("GalaxyPos_2", cat_data["GalaxyPos"][:, 1] % cfg.box_size),
        ("GalaxyPos_3", cat_data["GalaxyPos"][:, 2] % cfg.box_size),
        ("GalaxyVel_1", cat_data["GalaxyVel"][:, 0]),
        ("GalaxyVel_2", cat_data["GalaxyVel"][:, 1]),
        ("GalaxyVel_3", cat_data["GalaxyVel"][:, 2]),
        ("GalaxyVel", np.linalg.norm(cat_data["GalaxyVel"], axis=1)),
        ("GalaxyRhalf", np.log10(cat_data["GalaxyRhalf"])),
        ("SFR", np.log10(np.where(cat_data["SFR"] == 0, 1e-5, cat_data["SFR"]))),
    ]
    
    # Add JWST bands if present
    for band in ["jwst_f090w", "jwst_f150w", "jwst_f277w", "jwst_f444w"]:
        if band in cat_data.dtype.names:
            feats.append((band, cat_data[band]))
    
    if "HaloMass_z0" in cat_data.dtype.names:
        feats.append(("HaloMass_z0", np.log10(cat_data["HaloMass_z0"])))
    
    feature_names, arrays = zip(*feats)
    feature_array = np.stack(arrays, axis=1)
    
    # Extract components
    halo_idx = feature_names.index("HaloMass")
    pos_idx = [feature_names.index(f"GalaxyPos_{i}") for i in [1, 2, 3]]
    
    halo_masses = feature_array[:, halo_idx]
    positions = feature_array[:, pos_idx]
    
    # Get z0 masses if present
    halo_masses_z0 = None
    if "HaloMass_z0" in feature_names:
        z0_idx = feature_names.index("HaloMass_z0")
        halo_masses_z0 = feature_array[:, z0_idx]
        exclude_idx = pos_idx + [halo_idx, z0_idx]
    else:
        exclude_idx = pos_idx + [halo_idx]
    
    # TODO: compare to generate_gnn ?
    galaxy_features = feature_array[:, [i for i in range(len(feature_names)) 
                                        if i not in exclude_idx]]
    galaxy_names = [feature_names[i] for i in range(len(feature_names)) 
                   if i not in exclude_idx]
    
    return halo_masses, halo_masses_z0, positions, galaxy_features, galaxy_names

def apply_periodic_boundary(delta: np.ndarray, boxsize: float) -> np.ndarray:
    """Apply periodic boundary conditions."""
    mask = np.abs(delta) > 0.5 * boxsize
    delta[mask] = np.where(
        delta[mask] > 0, 
        delta[mask] - boxsize, 
        delta[mask] + boxsize
    )
    return delta

def build_graph(idx: list, halo_masses: np.ndarray, halo_masses_z0: np.ndarray,
                positions: np.ndarray, features: np.ndarray, 
                feature_names: list, cfg: Config) -> Data:
    """Build graph for a group of galaxies."""
    node_features = features[idx]
    central_mask = np.zeros(len(idx), dtype=bool)
    central_mask[0] = True
    
    # compute distances to central galaxy
    delta_central = positions[idx] - positions[idx[0]]
    delta_central = apply_periodic_boundary(delta_central, cfg.box_size)
    delta_central = delta_central / cfg.graph_radius
    dist_central = np.linalg.norm(delta_central, axis=1)
    
    # Add spatial features
    node_features = np.hstack([
        node_features, 
        delta_central, 
        dist_central.reshape(-1, 1)
    ])
    
    graph_names = feature_names + [
        "delta_x_central", "delta_y_central", "delta_z_central", "distance_central"
    ]
    
    # create edges
    n_nodes = len(idx)
    edge_idx = np.meshgrid(np.arange(n_nodes), np.arange(n_nodes))
    edge_idx = np.array(edge_idx, dtype=np.int64).reshape(2, -1)
    
    # edge features
    idx_array = np.array(idx)
    edge_positions = apply_periodic_boundary(
        positions[idx_array[edge_idx[0]]] - positions[idx_array[edge_idx[1]]], 
        cfg.box_size
    )
    edge_features = torch.tensor(
        edge_positions / (2.0 * cfg.graph_radius), 
        dtype=torch.float32
    )
    edge_features = torch.cat([
        edge_features, 
        torch.norm(edge_features, dim=1).reshape(-1, 1)
    ], dim=1)
    
    return Data(
        x=torch.tensor(node_features, dtype=torch.float32),
        edge_index=torch.tensor(edge_idx),
        edge_attr=edge_features,
        y=torch.tensor(halo_masses[idx[0]], dtype=torch.float32),
        y_z0=torch.tensor(halo_masses_z0[idx[0]], dtype=torch.float32) if halo_masses_z0 is not None else None,
        central_mask=torch.tensor(central_mask),
        feature_names=graph_names,
        global_attr=torch.tensor([n_nodes / 100], dtype=torch.float32),
    )

def generate_gnn_data(cfg: Config):
    """Generate GNN data for given configuration."""
    print(f"Generating GNN data for {cfg.sim.value} z={cfg.z}")
    # Load and process data
    cat_data = read_cat_data(cfg)
    
    # Find central galaxies and neighbors
    central_idx, neighbor_idx = find_neighbors(cat_data, cfg)

    # Process features
    halo_masses, halo_masses_z0, positions, features, feature_names = process_features(
        cat_data, cfg.box_size
    )

    # Compute statistics
    stats = {
        "feature_names": feature_names + ["HaloMass"],
        "means": np.append(features.mean(axis=0), halo_masses[central_idx].mean(axis=0)),
        "stds": np.append(features.std(axis=0), halo_masses[central_idx].std(axis=0)),
    }
    
    if halo_masses_z0 is not None:
        stats["feature_names"].append("HaloMass_z0")
        stats["means"] = np.append(stats["means"], halo_masses_z0[central_idx].mean(axis=0))
        stats["stds"] = np.append(stats["stds"], halo_masses_z0[central_idx].std(axis=0))
    
    # Create graphs
    graphs = []
    for center, neighbors in zip(central_idx, neighbor_idx):
        indices = [center] + neighbors
        graph = build_graph(
            indices, halo_masses, halo_masses_z0, 
            positions, features, feature_names, cfg
        )
        graphs.append(graph)
    
    # Save
    save_file(graphs, cfg, "data", f"{cfg.sim.value}_z{cfg.z}_graphs.pt")
    save_file(stats, cfg, "stats", f"{cfg.sim.value}_z{cfg.z}_stats.pt")
    
    
def generate_all_data(cfg: Config):
    """Generate data based on model type."""
    print(f"Data generator is using device: {device}")
    print(f"Imports took {timedelta(end_import-start_import)}")
    
    if cfg.model_type == ModelType.FCN:
        generate_fcn_data(cfg)
    else:
        generate_gnn_data(cfg) 