# ------------------
# Imports
# ------------------
# general
import time
start_import = time.time()

from pathlib import Path
import hydra
from omegaconf import DictConfig
from datetime import timedelta
import logging

# external
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import torch
from torch_geometric.data import Data

# internal
from DeepHalos.utils_old.__old_init__ import read_cat_data, DataConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger("GNN_data")

# Add to imports
import gc
import psutil
import os
from tqdm import tqdm  # For progress bars



# ------------------
# og Functions
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

# ------------------
# new Functions
# ------------------
# Add memory monitoring function
def log_memory_usage(label):
    """Log current memory usage with a descriptive label."""
    process = psutil.Process()
    ram_gb = process.memory_info().rss / (1024 * 1024 * 1024)
    logger.info(f"Memory usage ({label}): {ram_gb:.2f} GB")
    return ram_gb


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
    n_nodes = len(idx)
    
    # dealing with a potentially problematic large group
    if n_nodes > 2500:
        logger.warning(f"Processing large graph with {n_nodes} nodes and {n_nodes*n_nodes} edges")

    
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


def build_large_graph(
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
    """fully-connected graph w/ memory management."""
    n_nodes = len(idx)
    
    # Extract features for group
    idx_array = np.array(idx)
    node_features = features[idx]
    node_positions = positions[idx_array]
    
    # Create central galaxy mask
    central_mask = np.zeros(n_nodes, dtype=bool)
    central_mask[0] = True
    
    # Compute distances to central galaxy
    central_pos = node_positions[0]
    delta_central = node_positions - central_pos
    delta_central = apply_periodic_boundary(delta_central, boxsize) / radius
    dist_central = np.linalg.norm(delta_central, axis=1)
    
    # Add central distances to node features
    node_features = np.hstack(
        [node_features, delta_central, dist_central.reshape(-1, 1)]
    )
    graph_names = feature_names + [
        "delta_x_central",
        "delta_y_central",
        "delta_z_central",
        "distance_central",
    ]
    
    # Converting features to tensors to free numpy memory
    x_tensor = torch.tensor(node_features, dtype=torch.float32)
    central_mask_tensor = torch.tensor(central_mask)
    y_tensor = torch.tensor(halo_masses[idx[0]], dtype=torch.float32)
    y_z0_tensor = torch.tensor(halo_masses_z0[idx[0]], dtype=torch.float32) if halo_masses_z0 is not None else None
    global_attr_tensor = torch.tensor([n_nodes / max_galaxies], dtype=torch.float32)
    
    # Free memory
    del node_features, delta_central, dist_central
    gc.collect()
    
    # For edge construction, process in chunks if needed
    chunk_size = 5000  # Adjust based on your system's memory
    edge_indices = []
    edge_features_list = []
    
    # for very large graphs, we pre-allocate tensors for edge indices
    num_edges = n_nodes * n_nodes
    
    # For extremely large graphs, process in chunks
    logger.warning(f"Very large graph: {n_nodes} nodes with {num_edges} edges - processing in chunks")
    edge_idx_rows = []
    edge_idx_cols = []
    
    for i in range(0, n_nodes, chunk_size):
        chunk_end = min(i + chunk_size, n_nodes)
        
        # Create edge indices for this chunk
        for j in range(n_nodes):
            for k in range(i, chunk_end):
                edge_idx_rows.append(j)
                edge_idx_cols.append(k)
        
        # Process this chunk's edge features
        edge_features_chunk = []
        for j, k in zip(edge_idx_rows[-len(edge_idx_rows)//2:], edge_idx_cols[-len(edge_idx_cols)//2:]):
            delta = node_positions[j] - node_positions[k]
            delta = apply_periodic_boundary(delta.copy(), boxsize) / (2.0 * radius)
            edge_features_chunk.append(np.append(delta, np.linalg.norm(delta)))
        
        edge_features_list.extend(edge_features_chunk)
        
        logger.debug(f"Processed chunk {i//chunk_size + 1}, current memory: {log_memory_usage('chunk'):.2f} GB")
    
    edge_idx = torch.tensor([edge_idx_rows, edge_idx_cols], dtype=torch.long)
    edge_features = torch.tensor(edge_features_list, dtype=torch.float32)
    
    # Clean up
    del edge_idx_rows, edge_idx_cols, edge_features_chunk, edge_features_list
    gc.collect()
    
    # Create the graph object
    graph = Data(
        x=x_tensor,
        edge_index=edge_idx,
        edge_attr=edge_features,
        y=y_tensor,
        y_z0=y_z0_tensor,
        central_mask=central_mask_tensor,
        feature_names=graph_names,
        global_attr=global_attr_tensor,
    )
    
    # Final cleanup
    del edge_idx, edge_features, node_positions
    gc.collect()
    
    return graph