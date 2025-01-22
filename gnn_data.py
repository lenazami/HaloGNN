# ------------------
# Imports
# ------------------
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import torch
from torch_geometric.data import Data
import time
from data_utils import read_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("The GNN data script used: ", device)

# ------------------
# Dictionaries and Global Variables
# ------------------
DATA_PATH = Path(f"/n/holystore01/LABS/itc_lab/Lab/galaxyGNN/")

filenames = {
    "TNG": {
        6: DATA_PATH / "high-z-jwst-TNG/TNG100_galaxy_halo_catalog_z6.npy",
        5: DATA_PATH / "high-z-jwst-TNG/TNG100_galaxy_halo_catalog_z5.npy",
        4: DATA_PATH / "high-z-jwst-TNG/TNG100_galaxy_halo_catalog_z4.npy",
        3: DATA_PATH / "high-z-jwst-TNG/TNG100_galaxy_halo_catalog_z3.npy",
    },
    "ASTRID": {
        6: DATA_PATH / "high-z-jwst/ASTRID_galaxy_halo_catalog_047.npy",
        5: DATA_PATH / "high-z-jwst/ASTRID_galaxy_halo_catalog_107.npy",
        4: DATA_PATH / "high-z-jwst/ASTRID_galaxy_halo_catalog_147.npy",
        3: DATA_PATH / "high-z-jwst/ASTRID_galaxy_halo_catalog_214.npy",
    },
}
BOXSIZE = {
    "ASTRID": 250_000,
    "TNG": 75_000,
}


base_path = Path(f"/n/holystore01/LABS/itc_lab/Lab/galaxyGNN/carol_processed_data/")


def find_neighbors(
    cat_data: np.ndarray, boxsize: float, radius: float = 2000.0
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
    positions = cat_data["GalaxyPos"] % boxsize
    kdtree = cKDTree(positions, boxsize=boxsize)
    neighbor_idx = kdtree.query_ball_point(positions[central_idx], radius)
    return central_idx, neighbor_idx


def process_features(cat_data: np.ndarray, boxsize: float) -> tuple:
    """Extract and process galaxy features."""
    # Process features with appropriate transformations
    features = {
        "HaloMass": np.log10(cat_data["HaloMass"]),
        "GalaxyMass": np.log10(cat_data["GalaxyMass"]),
        "GalaxyPos_1": cat_data["GalaxyPos"][:, 0] % boxsize,
        "GalaxyPos_2": cat_data["GalaxyPos"][:, 1] % boxsize,
        "GalaxyPos_3": cat_data["GalaxyPos"][:, 2] % boxsize,
        "GalaxyVel_1": cat_data["GalaxyVel"][:, 0],
        "GalaxyVel_2": cat_data["GalaxyVel"][:, 1],
        "GalaxyVel_3": cat_data["GalaxyVel"][:, 2],
        "GalaxyVel": np.linalg.norm(cat_data["GalaxyVel"], axis=1),
        "GalaxyRhalf": np.log10(cat_data["GalaxyRhalf"]),
        "SFR": np.log10(np.where(cat_data["SFR"] == 0, 1e-5, cat_data["SFR"])),
        "jwst_f090w": cat_data["jwst_f090w"],
        "jwst_f150w": cat_data["jwst_f150w"],
        "jwst_f277w": cat_data["jwst_f277w"],
        "jwst_f444w": cat_data["jwst_f444w"],
    }
    if "HaloMass_z0" in cat_data.dtype.names:
        features["HaloMass_z0"] = np.log10(cat_data["HaloMass_z0"])

    df = pd.DataFrame(features)
    feature_array = df.values
    feature_names = df.columns.tolist()
    halo_idx = feature_names.index("HaloMass")
    pos_idx = [
        feature_names.index(f) for f in ["GalaxyPos_1", "GalaxyPos_2", "GalaxyPos_3"]
    ]

    if "HaloMass_z0" in feature_names:
        halo_z0_idx = feature_names.index("HaloMass_z0")
        non_halo_and_pos_idx = [
            i
            for i in range(len(feature_names))
            if i not in pos_idx and i != halo_idx and i != halo_z0_idx
        ]
        halo_masses_z0 = feature_array[:, halo_z0_idx]
    else:
        non_halo_and_pos_idx = [
            i for i in range(len(feature_names)) if i not in pos_idx and i != halo_idx
        ]
        halo_masses_z0 = None

    halo_masses = feature_array[:, halo_idx]
    positions = feature_array[:, pos_idx]
    galaxy_features = feature_array[:, non_halo_and_pos_idx]
    galaxy_names = [
        name for i, name in enumerate(feature_names) if i in non_halo_and_pos_idx
    ]

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
    """Build a graph for a group of galaxies."""
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


def create_galaxy_graphs(
    data_path: Path, suite: str = "TNG", radius: float = 2000.0, min_mass: float = 1e7
) -> list:
    """Main function to create galaxy graphs from catalog."""
    # Load and process data
    boxsize = BOXSIZE[suite]
    cat_data = read_data(data_path, min_mass=1e7)
    print(f"This catalogue contains {cat_data.shape} galaxies")
    # Find central galaxies and neighbors
    central_idx, neighbor_idx = find_neighbors(cat_data, boxsize, radius)

    # Process features
    halo_masses, halo_masses_z0, positions, features, feature_names = process_features(
        cat_data, boxsize
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
    for i, c_idx in enumerate(central_idx):
        graph_idx = [
            c_idx,
        ] + neighbor_idx[i]
        graph = build_graph(
            graph_idx,
            halo_masses,
            halo_masses_z0,
            positions,
            features,
            feature_names,
            boxsize,
            radius,
        )
        graphs.append(graph)
    print("Final len graphs = ", len(graphs))
    print([g.y for g in graphs[:10]])
    return graphs, stats


def get_time(elapsed):
    """
    Gets time elapsed in text format
    """
    hours, remainder = divmod(elapsed, 3600)  # 3600 seconds in an hour
    minutes, seconds = divmod(remainder, 60)  # 60 seconds in a minute
    return f"{int(hours)} hours, {int(minutes)} minutes, {seconds:.2f} seconds"


def process(sim, z, filepath, graph_radius, min_mass):
    start = time.time()
    output_path = base_path / f"gnn_{graph_radius:.1f}/{sim}_z{z}"
    output_path.mkdir(parents=True, exist_ok=True)
    # Housewarming
    print(f"\033[35mCurrently processing redshift z={z} for {sim} Suite\033[37m")

    graphs, stats = create_galaxy_graphs(
        filepath, suite=sim, radius=graph_radius, min_mass=min_mass
    )
    torch.save(graphs, output_path / f"{sim}_z{z}_all_graphs.pt")
    # save feature stastitics
    torch.save(stats, output_path / f"{sim}_z{z}_feature_stats.pt")
    end = time.time()
    print(f"{sim}z{z} halo processing took {get_time(end-start)}.")


if __name__ == "__main__":
    # ------------------
    # Main Processing
    # ------------------
    graph_radius = 2000.0
    min_mass = 1e7

    start_whole = time.time()
    for sim, data in filenames.items():
        for z, filepath in data.items():
            process(sim, z, filepath, graph_radius, min_mass)

    # ------------------
    # Finishing
    # ------------------
    end_whole = time.time()
    print(f"This program took {get_time(end_whole-start_whole)}")
