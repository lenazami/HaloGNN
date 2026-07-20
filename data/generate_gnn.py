# ------------------
# Imports
# ------------------
# general
import time
start_import = time.time()

from pathlib import Path
from datetime import timedelta
import logging

# external
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import torch
from torch_geometric.data import Data
from itertools import product

# internal
from DeepHalos.utils_old.__old_init__ import read_cat_data, DataConfig, _cfg_raw
from DeepHalos.utils_old.__old_init__ import SIMS, REDSHIFTS, OBSERVABLES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger("GNN_data")

# ------------------
# Functions
# ------------------
def find_neighbors(
    cat_data: np.ndarray, cfg: DataConfig
) -> tuple:
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


def process_features(cat_data: np.ndarray, boxsize: float) -> tuple:
    """Extract and process galaxy features."""
    # Process features with appropriate transformations
    
    feats = [
        ("HaloMass",   np.log10(cat_data["HaloMass"])),
        ("GalaxyMass", np.log10(cat_data["GalaxyMass"])),
        ("GalaxyPos_1", cat_data["GalaxyPos"][:, 0] % boxsize),
        ("GalaxyPos_2", cat_data["GalaxyPos"][:, 1] % boxsize),
        ("GalaxyPos_3", cat_data["GalaxyPos"][:, 2] % boxsize),
        ("GalaxyVel_1", cat_data["GalaxyVel"][:, 0]),
        ("GalaxyVel_2", cat_data["GalaxyVel"][:, 1]),
        ("GalaxyVel_3", cat_data["GalaxyVel"][:, 2]),
        ("GalaxyVel",   np.linalg.norm(cat_data["GalaxyVel"], axis=1)),
        ("GalaxyRhalf", np.log10(cat_data["GalaxyRhalf"])),
        ("SFR",         np.log10(np.where(cat_data["SFR"] == 0, 1e-5, cat_data["SFR"]))),
        ("jwst_f090w",  cat_data["jwst_f090w"]),
        ("jwst_f150w",  cat_data["jwst_f150w"]),
        ("jwst_f277w",  cat_data["jwst_f277w"]),
        ("jwst_f444w",  cat_data["jwst_f444w"]),
    ]
    if "HaloMass_z0" in cat_data.dtype.names:
        feats.append(("HaloMass_z0", np.log10(cat_data["HaloMass_z0"])))
        
    feature_names, arrays = zip(*feats)
    feature_names = list(feature_names)
    feature_array = np.stack(arrays, axis=1)
    
    halo_idx = feature_names.index("HaloMass")
    pos_idx = [feature_names.index(f) for f in ["GalaxyPos_1", "GalaxyPos_2", "GalaxyPos_3"]]

    if "HaloMass_z0" in feature_names:
        z0_idx = feature_names.index("HaloMass_z0")
        non_halo_pos = [i for i in range(len(feature_names))
            if i not in pos_idx + [halo_idx, z0_idx]]
        halo_masses_z0 = feature_array[:, z0_idx]
    else:
        non_halo_pos = [i for i in range(len(feature_names)) 
                        if i not in pos_idx + [halo_idx]]
        halo_masses_z0 = None

    halo_masses = feature_array[:, halo_idx]
    positions = feature_array[:, pos_idx]
    galaxy_features = feature_array[:, non_halo_pos]
    galaxy_names = [feature_names[i] for i in non_halo_pos]

    return halo_masses, halo_masses_z0, positions, galaxy_features, galaxy_names


def apply_periodic_boundary(delta: np.ndarray, boxsize: float) -> np.ndarray:
    """Apply periodic boundary conditions to position differences."""
    
    mask = np.abs(delta) > 0.5 * boxsize
    delta[mask] = np.where(
        delta[mask] > 0, delta[mask] - boxsize, delta[mask] + boxsize
    )
    return delta


def build_graph(
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
    # Get features for this group
    node_features = features[idx]

    # Create central galaxy mask
    central_mask = np.zeros(len(idx), dtype=bool)
    central_mask[0] = True

    # Calculate distances to central galaxy
    delta_central = positions[idx] - positions[idx[0]]
    delta_central = apply_periodic_boundary(delta_central, boxsize) / radius
    dist_central = np.linalg.norm(delta_central, axis=1)

    # Add central distances to features
    node_features = np.hstack(
        [node_features, delta_central, dist_central.reshape(-1, 1)]
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
    idx_array = np.array(idx)
    edge_positions = apply_periodic_boundary(
        positions[idx_array[edge_idx[0]]] - positions[idx_array[edge_idx[1]]], boxsize
    )
    edge_features = torch.tensor(edge_positions / (2.0 * radius), dtype=torch.float32)
    edge_features = torch.cat(
        [edge_features, torch.norm(edge_features, dim=1).reshape(-1, 1)], dim=1
    )

    return Data(
        x=torch.tensor(node_features),
        edge_index=torch.tensor(edge_idx),
        edge_attr=edge_features,
        y=halo_masses[idx[0]],
        y_z0=halo_masses_z0[idx[0]] if halo_masses_z0 is not None else None,
        central_mask=torch.tensor(central_mask),
        feature_names=graph_names,
        global_attr=torch.tensor([n_nodes / max_galaxies], dtype=torch.float32),
    )


def create_galaxy_graphs(cfg: DataConfig) -> list:
    """Loads catalog, finds neighbors, builds graphs, returns (graphs, stats)."""
    # Load and process data
    cat_data = read_cat_data(cfg)
    
    # Find central galaxies and neighbors
    central_idx, neighbor_idx = find_neighbors(cat_data, cfg)

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
    for center, neighbors in zip(central_idx, neighbor_idx):
        indices = [center] + neighbors
        graph = build_graph(
            indices,
            halo_masses,
            halo_masses_z0,
            positions,
            features,
            feature_names,
            cfg.box_size,
            cfg.graph_radius,
        )
        graphs.append(graph)
    
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