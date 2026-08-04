# src/halo/data.py
# dataset generation for fcn and gnn

# ------------------
# Imports
# ------------------
from typing import Any, Tuple, Union, List
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Subset, Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeomDataLoader
from sklearn.model_selection import train_test_split
from scipy.spatial import KDTree


# internal
from .config import Model, Simulation, PATHS

# ------------------
# Constants
# ------------------
# features we drop when training on observables only (differs by model type)
OBSERVABLE_FEATURES = {
    "Full": {
        "HaloMass", "GalaxyMass_Sum", "GalaxyMass_Max", "GalaxyMass_Mean",
        "SFR_Sum", "SFR_Max", "SFR_Mean", "Velocity_Dispersion",
        "Velocity_Max", "Velocity_Mean", "HaloMass_z0",
    },
    "Graph": {
        "HaloMass", "GalaxyVel_1", "GalaxyVel_2", "GalaxyVel_3",
        "GalaxyRhalf", "GalaxyMass", "GalaxyVel", "SFR",
    },
}

MIN_SFR = 1e-5

# ------------------
# Functions
# ------------------
def save_file(data: Any, filepath: Path, quiet: bool = True) -> Path:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, filepath)
    if not quiet:
        print(f"Saved to: {filepath}")
    return filepath


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
                    d[key] = val.float()
            else:
                d[key] = torch.tensor(val, dtype=torch.float32)
    return data_list


def load_cat(sim: Simulation) -> np.ndarray:
    """Load the galaxy catalog for a given simulation."""
    cat_path = sim.catalogue_path()
    print(f"Loading catalogue from: {cat_path}")
    return np.load(cat_path, allow_pickle=True)


def normalize(h: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Standardize a tensor, or invert it back with inverse=True."""
    return h * std + mean if inverse else (h - mean) / std


def filter_observables(data: list, drop: set) -> list:
    """Drop the observable-feature columns from every graph's x."""
    keep = [n not in drop for n in data[0].feature_names]
    mask = torch.tensor(keep)
    for g in data:
        g.x = g.x[:, mask]
        g.feature_names = [n for n, k in zip(g.feature_names, keep) if k]
    return data


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


class _NormalizedGraphs(torch.utils.data.Dataset):
    """Lazily standardize (and optionally column-filter) one graph at a time.

    The underlying graph list is memory-mapped and read-only, so we clone each
    accessed graph before writing into it. Only the graphs in the current batch
    are materialized, keeping host RAM ~ batch_size instead of the full dataset.
    """

    def __init__(self, data, cols, stat_cols, mean, std, target, keep_mask=None):
        self.data = data
        self.cols = cols
        self.stat_cols = stat_cols
        self.mean = mean
        self.std = std
        self.target = target
        self.keep_mask = keep_mask

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int) -> Data:
        g = self.data[i].clone()  # copy this graph out of the mmap (writable)
        g.x[:, self.cols] = normalize(g.x[:, self.cols], self.mean[self.stat_cols], self.std[self.stat_cols])
        g.y = normalize(g.y, self.mean[self.target], self.std[self.target])
        if self.keep_mask is not None:  # observables_only: drop columns after normalizing
            g.x = g.x[:, self.keep_mask]
            g.feature_names = [n for n, k in zip(g.feature_names, self.keep_mask.tolist()) if k]
        return g

def load_data(cfg: Mapping[str, Any], batch_size: int = 64, only_test: bool = False,
              all_data: bool = False, num_workers: int = 4):
    """
    Returns:
        Single DataLoader if all_data=True or only_test=True,
        otherwise returns (train_loader, val_loader, test_loader)
    """
    model = Model.from_cfg(dict(cfg))
    sim = Simulation.from_cfg(dict(cfg))

    stats = torch.load(PATHS.graph_stats(model, sim), map_location="cpu", weights_only=False)
    names = list(stats["feature_names"])
    mean = torch.as_tensor(stats["mean"], dtype=torch.float32)
    std = torch.as_tensor(stats["std"], dtype=torch.float32)

    if model.name == "Full":                               # single-node summaries: unchanged, tiny
        data = torch.load(PATHS.graphs(model, sim), map_location="cpu", mmap=True, weights_only=False)
        feat = list(data[0].feature_names)
    else:
        catalog = torch.load(PATHS.catalog(sim), map_location="cpu", weights_only=False)
        membership = torch.load(PATHS.membership(model, sim), map_location="cpu", weights_only=False)
        feat = list(catalog["feature_names"]) + [
            "delta_x_central", "delta_y_central", "delta_z_central", "distance_central",
        ]

    cols = [i for i, n in enumerate(feat) if n in names]
    stat_cols = [names.index(n) for n in feat if n in names]
    target = names.index(model.label_field)
    keep_mask = None
    if model.observables_only:
        drop = OBSERVABLE_FEATURES[model.name]
        keep_mask = torch.tensor([n not in drop for n in feat])

    if model.name == "Full":
        dataset = _NormalizedGraphs(data, cols, stat_cols, mean, std, target, keep_mask)
    else:
        targets = catalog["central_halo_mass_z0" if model.hm_present else "central_halo_mass"]
        dataset = Graphs(catalog, membership, targets, cols, stat_cols, mean, std, target, keep_mask)

    def loader(subset, shuffle):
        return GeomDataLoader(subset, batch_size=batch_size, shuffle=shuffle,
                              num_workers=num_workers, persistent_workers=num_workers > 0)
    # dont have to compute tts; minimizes overhead
    if all_data:
        return loader(dataset, shuffle=False)
    # split data
    test_size = 0.1 if sim.name == "TNG" else 0.01
    train_idx, val_idx, test_idx = get_split_indices(len(dataset), test_size=test_size, val_size=0.05)
    # create test loader
    test_loader = loader(Subset(dataset, test_idx), shuffle=False)
    # if testing, only return the test loader; minimizes overhead
    if only_test:
        return test_loader
    # else, also return train and val loaders
    train_loader = loader(Subset(dataset, train_idx), shuffle=True)
    val_loader = loader(Subset(dataset, val_idx), shuffle=False)
    return train_loader, val_loader, test_loader


# ----------
# fcn
# ----------
def make_halo_array(data: np.ndarray) -> List[np.ndarray]:
    """
    Takes in a data array and splits array when FOFID changes
    (catalogues are sorted by FOFID)
    """
    # On the off chance it is not sorted, this shouldn't take more than 5 seconds
    sorted_data = np.sort(data, order="FOFID")
    boundaries = np.flatnonzero(np.diff(sorted_data["FOFID"])) + 1
    return np.split(sorted_data, boundaries)


def sum_stats(halo: np.ndarray, add_z0: bool = False) -> Tuple:
    """
    Computes summary statistics for a group of galaxies within a halo.
    Returns a dictionary of features, making it easy to add/remove features
    while maintaining consistency in the data pipeline.

    Parameters:
        halo: Structured numpy array containing galaxy properties within a halo

    Returns:
        dict: Dictionary of feature names and their values
    """
    # target halo mass
    # and Structural features
    features = {"HaloMass": np.log10(halo["HaloMass"].max()),
                "N_Galaxies": len(halo)}
    if add_z0:
        features["HaloMass_z0"] = np.log10(halo["HaloMass_z0"].max())

    # JWST photometry features
    bands = [n for n in halo.dtype.names if n.startswith("jwst_")]

    for field in ("GalaxyMass", "SFR", *bands):
        values = np.maximum(halo[field], MIN_SFR) if field=="SFR" else halo[field]
        stats = np.array([
            values.sum(),
            values.max(),
            values.mean(),
        ])
        if field in ("GalaxyMass", "SFR"):
            stats = np.log10(stats)
        features.update({
            f"{field}_{name}": value
            for name, value in zip(("Sum", "Max", "Mean"), stats)
        })
 
    speed = np.linalg.norm(halo["GalaxyVel"], axis=1)

    features.update({
        "Velocity_Dispersion": speed.std(),
        "Velocity_Max": speed.max(),
        "Velocity_Mean": speed.mean(),
    })

    feats = np.fromiter(features.values(), dtype=np.float32)
    return feats, list(features)


def build_full_dataset(cat: np.ndarray, sim: Simulation, model: Model) -> Tuple[list, dict]:
    """full dataset as single-node graph."""
    halos = make_halo_array(cat)
    summaries = [sum_stats(h, add_z0=model.hm_present) for h in halos]

    summary_vals = np.stack([values for values, _ in summaries])
    summary_names = summaries[0][1]
    
    target_names = {"HaloMass", "HaloMass_z0"}
    
    feature_indices = [
        i for i, name in enumerate(summary_names)
        if name not in target_names
    ]
    target_index = summary_names.index(model.label_field)

    feature_names = [summary_names[i] for i in feature_indices]
   
    subset_indices = [*feature_indices, target_index]
    subset_names = [*feature_names, model.label_field]
    matrix = summary_vals[:, subset_indices]
    
    feats = torch.from_numpy(matrix[:, :-1])
    ys = torch.from_numpy(matrix[:, -1])
    # generating data as graphs so that down-the-line inference and comparison is easier
    graphs = [
        Data(
            x=feats[i : i+1],
            y=ys[i],
            feature_names=feature_names,
            edge_index=torch.empty((2, 0), dtype=torch.long),
        )
        for i in range(len(halos))
    ]
    stats = {
        "feature_names": subset_names,
        "mean": torch.tensor(matrix.mean(axis=0)),
        "std": torch.tensor(matrix.std(axis=0)),
    }
    return graphs, stats

# ----------
# gnn
# ----------

class Graphs(Dataset):
    """On-the-fly node features from catalog tables + CSR membership. Emits x/pos/y only;
    the model builds the n^2 pair geometry on-device. No edges are constructed here."""

    def __init__(self, catalog, membership, targets, cols, stat_cols, mean, std, target, keep_mask=None):
        self.features = catalog["features"]
        self.positions = catalog["positions"]
        self.central_idx = catalog["central_idx"]
        self.boxsize = catalog["boxsize"]
        self.feature_names = catalog["feature_names"]      # kept on the dataset, NOT on every sample
        self.flat = membership["neighbors_flat"]
        self.offsets = membership["offsets"]
        self.radius = membership["radius"]
        self.targets = targets                             # central targets, aligned with central_idx
        self.cols, self.stat_cols = cols, stat_cols
        self.mean, self.std, self.target = mean, std, target
        self.keep_mask = keep_mask

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def __getitem__(self, i: int) -> Data:
        idx = self.flat[self.offsets[i] : self.offsets[i + 1]]
        center = int(self.central_idx[i])
        node_pos = self.positions[idx]

        delta_c = apply_periodic_boundary(node_pos - self.positions[center], self.boxsize) / self.radius
        dist_c = np.linalg.norm(delta_c, axis=1, keepdims=True)
        x = torch.from_numpy(np.hstack([self.features[idx], delta_c, dist_c]).astype(np.float32))

        x[:, self.cols] = normalize(x[:, self.cols], self.mean[self.stat_cols], self.std[self.stat_cols])
        if self.keep_mask is not None:
            x = x[:, self.keep_mask]
        y = normalize(torch.as_tensor(self.targets[i], dtype=torch.float32),
                      self.mean[self.target], self.std[self.target])

        return Data(
            x=x,
            pos=torch.as_tensor(node_pos, dtype=torch.float32),
            y=y,
            central_mask=torch.from_numpy(idx == center),
            global_attr=torch.tensor([len(idx) / 100.0], dtype=torch.float32),
            radius=torch.tensor([float(self.radius)], dtype=torch.float32),
            boxsize=torch.tensor([float(self.boxsize)], dtype=torch.float32),
        )


# class Graphs(Dataset):
    # ...

def find_neighbors(cat_data: np.ndarray, boxsize, graph_radius) -> tuple:
    """Find central galaxies and their neighbors."""
    # added subgroup id to catalogs to be able to match halos between gnn and fcn?
    central_idx = np.flatnonzero(cat_data["sgpID"] == 1)
    # Find neighbors using periodic boundary conditions
    positions = cat_data["GalaxyPos"] % boxsize
    tree = KDTree(positions, boxsize=boxsize)
    nbr_idx = tree.query_ball_point(positions[central_idx], graph_radius)
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
        *(f"GalaxyVel_{i+1}" for i in range(3)),
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
        np.log10(np.maximum(cat_data["SFR"], MIN_SFR)),
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
    idx: np.ndarray,
    target_masses: np.ndarray,
    positions: np.ndarray,
    features: np.ndarray,
    feature_names: list,
    boxsize: float,
    radius: float,
    max_galaxies: int = 100,
) -> Data:
    """Build graph for group of galaxies using PyG Data object."""
    n_nodes = len(idx)
    # dealing with a potentially problematic large group
    if n_nodes > 2500:
        print(f"Processing large graph with {n_nodes} nodes and {n_nodes*n_nodes} edges")

    # Get features for this group
    node_features = features[idx]

    # Create central galaxy mask
    central_mask = idx == center

    # Calculate distances to central galaxy
    node_pos = positions[idx]
    delta_central = apply_periodic_boundary(node_pos - positions[center], boxsize) / radius
    dist_central = np.linalg.norm(delta_central, axis=1)

    # Add central distances to features
    # TODO: check if we can use columnstack and drop the :,None on dist
    node_features = np.hstack(
        [features[idx], delta_central, dist_central[:, None]]
    ).astype(np.float32)

    # Create edges between all pairs
    # n_nodes = len(idx)
    nodes = np.arange(n_nodes, dtype=np.int64)

    edge_idx = np.stack([
        np.repeat(nodes, n_nodes),
        np.tile(nodes, n_nodes),
    ])
    src, dst = edge_idx

    edge_delta = apply_periodic_boundary(
        node_pos[src] - node_pos[dst],
        boxsize,
    ) / (2.0 * radius)

    edge_distance = np.linalg.norm(
        edge_delta,
        axis=1,
        keepdims=True,
    )
    edge_attr = np.concatenate(
        [edge_delta, edge_distance],
        axis=1,
    ).astype(np.float32)

    edge_feat_names = [
        "delta_x_central",
        "delta_y_central",
        "delta_z_central",
        "distance_central",
    ]

    graph_feat_names = feature_names + edge_feat_names

    return Data(
        x=torch.from_numpy(node_features),
        edge_index=torch.from_numpy(edge_idx),
        edge_attr=torch.from_numpy(edge_attr),
        pos=torch.as_tensor(node_pos, dtype=torch.float32),
        y=torch.as_tensor(target_masses[center]),
        central_mask=torch.from_numpy(central_mask),
        feature_names=graph_feat_names,
        global_attr=torch.tensor([n_nodes / max_galaxies], dtype=torch.float32),
    )

def dense_pair_geometry(pos: Tensor, batch: Tensor, radius: float, boxsize: float):
    """dense graph edges, built on `pos.device`."""
    counts = torch.bincount(batch)
    starts = torch.cumsum(counts, 0) - counts
    edges_per = counts * counts
    graph_of_edge = torch.repeat_interleave(torch.arange(counts.numel(), device=pos.device), edges_per)
    k = torch.arange(int(edges_per.sum()), device=pos.device) - (torch.cumsum(edges_per, 0) - edges_per)[graph_of_edge]
    n = counts[graph_of_edge]
    src = starts[graph_of_edge] + k // n
    dst = starts[graph_of_edge] + k % n

    delta = pos[src] - pos[dst]
    delta = (delta + 0.5 * boxsize) % boxsize - 0.5 * boxsize
    delta = delta / (2.0 * radius)
    dist = torch.linalg.norm(delta, dim=1, keepdim=True)
    return torch.stack([src, dst]), torch.cat([delta, dist], dim=1)
    

def build_graph_dataset(cat: np.ndarray, sim: Simulation, model: Model) -> Tuple[dict, dict, dict]:
    """Build the GNN graph dataset from a catalog. 
    Returns (catalog, membership, stats)."""
    central_idx, neighbor_idx = find_neighbors(cat, sim.boxsize, model.graph_radius)
    halo_masses, halo_masses_z0, positions, features, feature_names = process_features(cat, sim.boxsize)

    lengths = np.fromiter((len(n) for n in neighbor_idx), dtype=np.int64, count=len(neighbor_idx))
    offsets = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
    neighbors_flat = np.concatenate([np.asarray(n, dtype=np.int32) for n in neighbor_idx])

    catalog = {
        "features": features.astype(np.float32),
        "positions": np.asarray(positions, dtype=np.float32),
        "central_idx": np.asarray(central_idx, dtype=np.int64),
        "central_halo_mass": halo_masses[central_idx].astype(np.float32),
        "central_halo_mass_z0": (
            halo_masses_z0[central_idx].astype(np.float32) if halo_masses_z0 is not None else None
        ),
        "feature_names": feature_names,
        "boxsize": float(sim.boxsize),
    }
    membership = {"neighbors_flat": neighbors_flat, "offsets": offsets, "radius": float(model.graph_radius)}

    # target is the label mass (z0 present-day mass when hm_present)
    central_target = (halo_masses_z0 if model.hm_present else halo_masses)[central_idx]
    stats = {
        "feature_names": feature_names + [model.label_field],
        "mean": torch.tensor(np.append(features.mean(axis=0), central_target.mean()), dtype=torch.float32),
        "std": torch.tensor(np.append(features.std(axis=0), central_target.std()), dtype=torch.float32),
    }
    return catalog, membership, stats

# -----------------------------
# graph construction variants
# -----------------------------