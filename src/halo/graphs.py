# ------------------
# Imports
# ------------------
# general
import time
start_import = time.time()

# from pathlib import Path
# from datetime import timedelta
# import logging

# external
import numpy as np
# import pandas as pd
from scipy.spatial import cKDTree
import torch
from torch_geometric.data import Data
# from itertools import product

# internal
# from DeepHalos.utils_old.__old_init__ import read_cat_data, DataConfig, _cfg_raw
# from DeepHalos.utils_old.__old_init__ import SIMS, REDSHIFTS, OBSERVABLES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# logger = logging.getLogger("GNN_data")

# ------------------
# Functions
# ------------------
def find_neighbors(cat_data: np.ndarray, boxsize, graph_radius) -> tuple:
    """Find central galaxies and their neighbors."""
    # added subgroup id to catalogs to be able to match halos between gnn and fcn?
    central_idx = np.flatnonzero(cat_data["sgpID"] == 1)
    # Find neighbors using periodic boundary conditions
    positions = cat_data["GalaxyPos"] % boxsize
    tree = cKDTree(positions, boxsize=boxsize)
    nbr_idx = tree.query_ball_point(positions[central_idx], graph_radius, return_sorted=False)
    return central_idx, nbr_idx

def process_features(cat_data: np.ndarray, boxsize: float) -> tuple:
    """Extract and process galaxy features."""
    # Process features with appropriate transformations
    names = cat_data.dtype.names
    # n_galaxies = len(cat_data)
    
    positions = cat_data["GalaxyPos"] % boxsize
    vel = cat_data["GalaxyVel"]
    bands = [n for n in names if n.startswith("jwst_")]
    
    feat_names = [
        "GalaxyMass",
        *(f"GalaxyVel_{i + 1}" for i in range(3)),
        "GalaxyVel",
        "GalaxyRhalf",
        "SFR",
        *bands,
    ]
    feats = np.column_stack([
        np.log10(cat_data["GalaxyMass"]),
        vel,
        np.linalg.norm(vel, axis=1),
        np.log10(cat_data["GalaxyRhalf"]),
        np.log10(np.maximum(cat_data["SFR"], 1e-5)),
        *(cat_data[name] for name in bands),
    ]).astype(np.float32)
    
    halo_masses = np.log10(cat_data["HaloMass"]).astype(np.float32)
    # to predict hm in present day
    halo_masses_z0 = (
        np.log10(cat_data["HaloMass_z0"]).astype(np.float32)
        if "HaloMass_z0" in names
        else None
    )
    
    return halo_masses, halo_masses_z0, positions, feats, feat_names


def apply_periodic_boundary(delta: np.ndarray, boxsize: float) -> np.ndarray:
    """Apply periodic boundary conditions to position differences."""
    return (delta + 0.5 * boxsize) % boxsize - 0.5 * boxsize

def build_graph(
    center: int,
    idx: list,
    halo_masses: np.ndarray,
    halo_masses_z0: np.ndarray,
    positions: np.ndarray,
    features: np.ndarray,
    feature_names: list,
    boxsize: float,
    radius: float,
    max_galaxies: int = 100,
) -> Data:
    """Build a graph for a group of galaxies using a PyTorch Geometric Data object."""
    idx = np.array(idx)
    # Get features for this group
    node_features = features[idx]

    # Create central galaxy mask
    central_mask = idx == center

    # Calculate distances to central galaxy
    delta_central = positions[idx] - positions[center]
    delta_central = apply_periodic_boundary(delta_central, boxsize) / radius
    dist_central = np.linalg.norm(delta_central, axis=1)

    # Add central distances to features
    node_features = np.hstack(
        [node_features, delta_central, dist_central[:, None]]
    )
    graph_names = feature_names + [
        "delta_x_central",
        "delta_y_central",
        "delta_z_central",
        "distance_central",
    ]

    # Create edges between all pairs
    n_nodes = len(idx)

    edge_idx = np.meshgrid(np.arange(n_nodes), np.arange(n_nodes))
    edge_idx = np.array(edge_idx, dtype=np.int64).reshape(2, -1)
    # Calculate edge features (distances between pairs)
    
    edge_positions = apply_periodic_boundary(
        positions[idx[edge_idx[0]]] - positions[idx[edge_idx[1]]], boxsize
    )
    edge_features = torch.tensor(edge_positions / (2.0 * radius), dtype=torch.float32)
    edge_features = torch.cat(
        [edge_features, torch.norm(edge_features, dim=1, keepdim=True)], dim=1
    )

    return Data(
        x=torch.tensor(node_features),
        edge_index=torch.tensor(edge_idx),
        edge_attr=edge_features,
        y=torch.tensor(halo_masses[center]),
        y_z0=torch.tensor(halo_masses_z0[center]) if halo_masses_z0 is not None else None,
        central_mask=torch.from_numpy(central_mask),
        feature_names=graph_names,
        global_attr=torch.tensor([n_nodes / max_galaxies], dtype=torch.float32),
    )


def create_galaxy_graphs(cfg: DataConfig) -> list:
    """Loads catalog, finds neighbors, builds graphs, returns (graphs, stats)."""
    # Load and process data
    cat_data = read_cat_data(cfg)
    
    # Find central galaxies and neighbors
    central_idx, idx = find_neighbors(cat_data, cfg)

    # Process features
    halo_masses, halo_masses_z0, positions, features, feature_names = process_features(
        cat_data, cfg.box_size
    )

    features_means, features_stds = features.mean(axis=0), features.std(axis=0)

    central_halo_masses = halo_masses[central_idx]
    halo_masses_means, halo_masses_stds = central_halo_masses.mean(
        axis=0
    ), central_halo_masses.std(axis=0)

    stats = {
        "feature_names": feature_names + ["HaloMass"],
        "means": np.append(
            features_means,
            halo_masses_means,
        ),
        "stds": np.append(
            features_stds,
            halo_masses_stds,
        ),
    }
    if halo_masses_z0 is not None:
        central_halo_masses_z0 = halo_masses_z0[central_idx]
        halo_masses_z0_means, halo_masses_z0_stds = central_halo_masses_z0.mean(
            axis=0
        ), central_halo_masses_z0.std(axis=0)
        stats["feature_names"] = feature_names + ["HaloMass", "HaloMass_z0"]
        stats["means"] = np.append(
            np.append(
                features_means,
                halo_masses_means,
            ),
            halo_masses_z0_means,
        )
        stats["stds"] = np.append(
            np.append(
                features_stds,
                halo_masses_stds,
            ),
            halo_masses_z0_stds,
        )

    # Create graphs
    graphs = []
    for center, nodes in zip(central_idx, idx):
        graphs.append(
            build_graph(
                center,
                nodes,
                halo_masses,
                halo_masses_z0,
                positions,
                features,
                feature_names,
                cfg.box_size,
                cfg.graph_radius,
            )
        )
    
    return graphs, stats


# ------------------
# Main
# ------------------
def main():
    print(f"GNN data generator is using device: {device}")
    start = time.time()
    cfg = DataConfig()
    
    
    for sim,z in product(SIMS, REDSHIFTS):
        if sim == "TNG" and z == 3:
            continue
        start = time.time()
        cfg.sim = sim
        cfg.z = z
        output_path = cfg.root / f"processed_data/gnn_{cfg.graph_radius:.1f}/{cfg.sim}_z{cfg.z}"
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Processing suite {sim} at redshift {z}")
        
        logger.debug(f"Output path: {output_path}")
        
        graphs, stats = create_galaxy_graphs(cfg)
        torch.save(graphs, output_path / f"all_graphs.pt")
        torch.save(stats, output_path / f"feature_stats.pt")
        
        logger.info(f"Wrote {len(graphs)} graphs to {output_path}")
        logger.debug([g.y for g in graphs[:10]])
        logger.info(f"{cfg.sim} z{cfg.z} halo processing took {timedelta(seconds=(time.time()-start))}.\n")
            
            
    logger.info(f"This program took {timedelta(seconds=(time.time()-start_import))}.\n")

if __name__ == "__main__":
    main()