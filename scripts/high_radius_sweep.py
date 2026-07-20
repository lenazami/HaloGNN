# ------------------
# Imports
# ------------------
# general -----
from __future__ import annotations

import gc
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# external -----
import numpy as np
import pandas as pd
import psutil
import torch
from scipy.spatial import cKDTree
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


# ------------------
# Parameters
# ------------------
class Simulation(Enum):
    TNG = "TNG"
    ASTRID = "ASTRID"


SIM = Simulation.ASTRID
Z = 4
MIN_MASS = 1e7
CATALOGUE_ROOT = Path("/n/holystore01/LABS/itc_lab/Lab/galaxyGNN")
CONTEXT_RADII = [2500.0, 5000.0, 10000.0]
LOCAL_EDGE_RADIUS = 2000.0
BATCH_SIZE = 1
NUM_WORKERS = 0
MAX_STEPS = 1000
PATIENCE = 20
RANDOM_STATE = 42
MAX_EDGE_GUARD = 1_500_000
MAX_NODE_GUARD = 20_000
MAX_CONTEXT_NODES: Optional[int] = None
MAX_GALAXIES_SCALE = 100.0
PROFILE_SAMPLE_LIMIT: Optional[int] = None

RUN_MODE = os.environ.get("HALOGNN_HIGH_RADIUS_MODE", "profile")
SCRATCH_ROOT = Path(os.environ.get("SCRATCH", "/n/netscratch")) / "halognn" / "high_radius"
PROJECT_RESULTS_ROOT = Path("~/halognn/results/high_radius").expanduser()
PROJECT_CHECKPOINT_ROOT = Path("~/halognn/checkpoints/high_radius").expanduser()


# ------------------
# Constants
# ------------------
BOX_SIZES = {Simulation.TNG: 75_000.0, Simulation.ASTRID: 250_000.0}
ASTRID_CAT_IDS = {3: "214", 4: "147", 5: "107", 6: "047"}
SPATIAL_FEATURE_NAMES = [
    "delta_x_central",
    "delta_y_central",
    "delta_z_central",
    "distance_central",
]
LOGGER = logging.getLogger("high_radius_sweep")


# ------------------
# Data Classes
# ------------------
@dataclass(frozen=True)
class RadiusProfile:
    radius: float
    halo_index: int
    n_nodes: int
    n_edges: int
    max_distance: float
    status: str


# ------------------
# Helpers
# ------------------
def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def run_name(context_radius: float) -> str:
    label = f"GNN_{SIM.value}_z{Z}_hybrid_r{context_radius:.0f}_local{LOCAL_EDGE_RADIUS:.0f}"
    if MAX_CONTEXT_NODES is not None:
        label += f"_cap{MAX_CONTEXT_NODES}"
    return label


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def memory_gb() -> float:
    return psutil.Process().memory_info().rss / 1024**3


def minimum_image_delta(delta: np.ndarray, box_size: float) -> np.ndarray:
    delta = np.array(delta, copy=True)
    mask = np.abs(delta) > 0.5 * box_size
    delta[mask] = np.where(delta[mask] > 0, delta[mask] - box_size, delta[mask] + box_size)
    return delta


def central_indices(cat_data: np.ndarray) -> np.ndarray:
    df = pd.DataFrame(
        {
            "GalaxyMass": cat_data["GalaxyMass"],
            "HaloMass": cat_data["HaloMass"],
            "FOFID": cat_data["FOFID"],
        }
    )
    return df.groupby("FOFID")["HaloMass"].idxmax().values.astype(np.int64)


def catalogue_path(sim: Simulation, z: int) -> Path:
    if sim == Simulation.ASTRID:
        filename = f"ASTRID_galaxy_halo_catalog_{ASTRID_CAT_IDS[z]}.npy"
        return CATALOGUE_ROOT / "catalogs/high-z-jwst-ASTRID" / filename
    filename = f"TNG100_galaxy_halo_catalog_z{z}.npy"
    return CATALOGUE_ROOT / "high-z-jwst-TNG" / filename


def read_catalogue(sim: Simulation, z: int) -> np.ndarray:
    path = catalogue_path(sim, z)
    if not path.exists():
        raise FileNotFoundError(f"Catalogue not found: {path}")
    arr = np.load(path, allow_pickle=True)
    return arr[arr["GalaxyMass"] >= MIN_MASS]


def process_galaxy_features(cat_data: np.ndarray, box_size: float) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray, List[str]]:
    feats = [
        ("HaloMass", np.log10(cat_data["HaloMass"])),
        ("GalaxyMass", np.log10(cat_data["GalaxyMass"])),
        ("GalaxyPos_1", cat_data["GalaxyPos"][:, 0] % box_size),
        ("GalaxyPos_2", cat_data["GalaxyPos"][:, 1] % box_size),
        ("GalaxyPos_3", cat_data["GalaxyPos"][:, 2] % box_size),
        ("GalaxyVel_1", cat_data["GalaxyVel"][:, 0]),
        ("GalaxyVel_2", cat_data["GalaxyVel"][:, 1]),
        ("GalaxyVel_3", cat_data["GalaxyVel"][:, 2]),
        ("GalaxyVel", np.linalg.norm(cat_data["GalaxyVel"], axis=1)),
        ("GalaxyRhalf", np.log10(cat_data["GalaxyRhalf"])),
        ("SFR", np.log10(np.where(cat_data["SFR"] == 0, 1e-5, cat_data["SFR"]))),
    ]

    for band in ["jwst_f090w", "jwst_f150w", "jwst_f277w", "jwst_f444w"]:
        if band in cat_data.dtype.names:
            feats.append((band, cat_data[band]))

    if "HaloMass_z0" in cat_data.dtype.names:
        feats.append(("HaloMass_z0", np.log10(cat_data["HaloMass_z0"])))

    feature_names, arrays = zip(*feats)
    feature_names = list(feature_names)
    feature_array = np.stack(arrays, axis=1).astype(np.float32)

    halo_idx = feature_names.index("HaloMass")
    pos_idx = [feature_names.index(f"GalaxyPos_{i}") for i in [1, 2, 3]]
    halo_masses = feature_array[:, halo_idx]
    positions = feature_array[:, pos_idx]

    halo_masses_z0 = None
    excluded = pos_idx + [halo_idx]
    if "HaloMass_z0" in feature_names:
        z0_idx = feature_names.index("HaloMass_z0")
        halo_masses_z0 = feature_array[:, z0_idx]
        excluded.append(z0_idx)

    kept = [i for i in range(len(feature_names)) if i not in excluded]
    galaxy_features = feature_array[:, kept]
    galaxy_names = [feature_names[i] for i in kept]
    return halo_masses, halo_masses_z0, positions, galaxy_features, galaxy_names


def safe_std(values: np.ndarray) -> np.ndarray:
    std = values.std(axis=0).astype(np.float32)
    return np.where(std == 0, 1.0, std)


def build_hybrid_edges(
    node_positions: np.ndarray,
    center_index: int,
    context_radius: float,
    local_edge_radius: float,
    box_size: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    edges: List[Tuple[int, int]] = []
    n_nodes = len(node_positions)

    for idx in range(n_nodes):
        if idx != center_index:
            edges.append((center_index, idx))
            edges.append((idx, center_index))

    noncentral = np.array([idx for idx in range(n_nodes) if idx != center_index], dtype=np.int64)
    if len(noncentral) > 1:
        tree = cKDTree(node_positions[noncentral] % box_size, boxsize=box_size)
        pairs = tree.query_pairs(local_edge_radius, output_type="ndarray")
        for src_local, dst_local in pairs:
            src = int(noncentral[src_local])
            dst = int(noncentral[dst_local])
            edges.append((src, dst))
            edges.append((dst, src))

    if not edges:
        return (
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0, 4), dtype=torch.float32),
        )

    unique_edges = list(dict.fromkeys(edges))
    src = np.array([edge[0] for edge in unique_edges], dtype=np.int64)
    dst = np.array([edge[1] for edge in unique_edges], dtype=np.int64)
    rel = minimum_image_delta(node_positions[src] - node_positions[dst], box_size)
    norm_rel = rel / (2.0 * context_radius)
    dist = np.linalg.norm(rel, axis=1, keepdims=True) / (2.0 * context_radius)
    edge_attr = np.hstack([norm_rel, dist]).astype(np.float32)
    edge_index = np.vstack([src, dst])
    return torch.tensor(edge_index, dtype=torch.long), torch.tensor(edge_attr, dtype=torch.float32)


def estimate_hybrid_edges(
    node_positions: np.ndarray,
    center_index: int,
    local_edge_radius: float,
    box_size: float,
) -> int:
    n_nodes = len(node_positions)
    central_edges = 2 * max(n_nodes - 1, 0)
    noncentral = np.array([idx for idx in range(n_nodes) if idx != center_index], dtype=np.int64)
    if len(noncentral) <= 1:
        return central_edges
    tree = cKDTree(node_positions[noncentral] % box_size, boxsize=box_size)
    return central_edges + 2 * len(tree.query_pairs(local_edge_radius))


def status_for_graph(n_nodes: int, n_edges: int) -> str:
    if n_nodes > MAX_NODE_GUARD:
        return "too_many_nodes"
    if n_edges > MAX_EDGE_GUARD:
        return "too_many_edges"
    return "ok"


def save_json(data: Dict[str, object], path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def copy_small_outputs(scratch_run_dir: Path, project_run_dir: Path) -> None:
    ensure_dir(project_run_dir)
    for path in scratch_run_dir.glob("*"):
        if path.is_file() and path.suffix in {".csv", ".json", ".pt", ".npz"}:
            shutil.copy2(path, project_run_dir / path.name)


# ------------------
# Dataset
# ------------------
class StreamingHybridRadiusDataset(Dataset):
    """Builds high-radius hybrid graphs on demand."""

    def __init__(
        self,
        cat_data: np.ndarray,
        sim: Simulation,
        z: int,
        context_radius: float,
        local_edge_radius: float,
        hm_present: bool = False,
        max_context_nodes: Optional[int] = None,
    ):
        self.cat_data = cat_data
        self.sim = sim
        self.z = z
        self.box_size = BOX_SIZES[sim]
        self.context_radius = context_radius
        self.local_edge_radius = local_edge_radius
        self.hm_present = hm_present
        self.max_context_nodes = max_context_nodes

        self.central_idx = central_indices(cat_data)
        self.halo_masses, self.halo_masses_z0, self.positions, self.features, self.feature_names = process_galaxy_features(
            cat_data,
            self.box_size,
        )
        self.graph_feature_names = self.feature_names + SPATIAL_FEATURE_NAMES
        self.kdtree = cKDTree(self.positions % self.box_size, boxsize=self.box_size)
        self.stats = self._compute_stats()

    @classmethod
    def from_catalogue(
        cls,
        context_radius: float,
        local_edge_radius: float,
        hm_present: bool = False,
        max_context_nodes: Optional[int] = None,
    ) -> "StreamingHybridRadiusDataset":
        return cls(
            read_catalogue(SIM, Z),
            SIM,
            Z,
            context_radius,
            local_edge_radius,
            hm_present=hm_present,
            max_context_nodes=max_context_nodes,
        )

    @property
    def node_features_dim(self) -> int:
        return len(self.graph_feature_names)

    def _compute_stats(self) -> Dict[str, object]:
        central_halo_masses = self.halo_masses[self.central_idx]
        target_std = float(central_halo_masses.std())
        if target_std == 0:
            target_std = 1.0

        stats: Dict[str, object] = {
            "feature_names": self.feature_names + ["HaloMass"],
            "means": np.append(self.features.mean(axis=0), central_halo_masses.mean()).astype(np.float32),
            "stds": np.append(safe_std(self.features), target_std).astype(np.float32),
        }
        if self.halo_masses_z0 is not None:
            central_z0 = self.halo_masses_z0[self.central_idx]
            z0_std = float(central_z0.std()) or 1.0
            stats["feature_names"] = self.feature_names + ["HaloMass", "HaloMass_z0"]
            stats["means"] = np.append(stats["means"], central_z0.mean()).astype(np.float32)
            stats["stds"] = np.append(stats["stds"], z0_std).astype(np.float32)
        return stats

    def __len__(self) -> int:
        return len(self.central_idx)

    def _context_indices(self, halo_index: int) -> np.ndarray:
        center = int(self.central_idx[halo_index])
        neighbors = np.array(
            self.kdtree.query_ball_point(self.positions[center], self.context_radius),
            dtype=np.int64,
        )
        neighbors = neighbors[neighbors != center]

        if self.max_context_nodes is not None and len(neighbors) > self.max_context_nodes - 1:
            delta = minimum_image_delta(self.positions[neighbors] - self.positions[center], self.box_size)
            nearest = np.argsort(np.linalg.norm(delta, axis=1))[: self.max_context_nodes - 1]
            neighbors = neighbors[nearest]

        return np.concatenate([[center], neighbors])

    def profile_one(self, halo_index: int) -> RadiusProfile:
        idx = self._context_indices(halo_index)
        node_positions = self.positions[idx]
        deltas = minimum_image_delta(node_positions - node_positions[0], self.box_size)
        distances = np.linalg.norm(deltas, axis=1)
        n_edges = estimate_hybrid_edges(node_positions, 0, self.local_edge_radius, self.box_size)
        status = status_for_graph(len(idx), n_edges)
        return RadiusProfile(
            radius=self.context_radius,
            halo_index=halo_index,
            n_nodes=len(idx),
            n_edges=n_edges,
            max_distance=float(distances.max(initial=0.0)),
            status=status,
        )

    def __getitem__(self, halo_index: int) -> Data:
        idx = self._context_indices(halo_index)
        node_positions = self.positions[idx]
        base_x = self.features[idx]
        means = np.asarray(self.stats["means"][: len(self.feature_names)], dtype=np.float32)
        stds = np.asarray(self.stats["stds"][: len(self.feature_names)], dtype=np.float32)
        base_x = (base_x - means) / stds

        delta_central = minimum_image_delta(node_positions - node_positions[0], self.box_size) / self.context_radius
        dist_central = np.linalg.norm(delta_central, axis=1, keepdims=True)
        x = np.hstack([base_x, delta_central, dist_central]).astype(np.float32)

        edge_index, edge_attr = build_hybrid_edges(
            node_positions,
            center_index=0,
            context_radius=self.context_radius,
            local_edge_radius=self.local_edge_radius,
            box_size=self.box_size,
        )

        target_field = "HaloMass_z0" if self.hm_present else "HaloMass"
        target_values = self.halo_masses_z0 if self.hm_present else self.halo_masses
        if target_values is None:
            raise ValueError("hm_present=True requested, but HaloMass_z0 is absent.")
        target_idx = list(self.stats["feature_names"]).index(target_field)
        y = (target_values[idx[0]] - self.stats["means"][target_idx]) / self.stats["stds"][target_idx]

        central_mask = torch.zeros(len(idx), dtype=torch.bool)
        central_mask[0] = True

        return Data(
            x=torch.tensor(x, dtype=torch.float32),
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor(y, dtype=torch.float32),
            y_z0=torch.tensor(self.halo_masses_z0[idx[0]], dtype=torch.float32) if self.halo_masses_z0 is not None else None,
            central_mask=central_mask,
            feature_names=self.graph_feature_names,
            global_attr=torch.tensor([len(idx) / MAX_GALAXIES_SCALE], dtype=torch.float32),
        )


# ------------------
# Profiling
# ------------------
def profile_dataset(dataset: StreamingHybridRadiusDataset, output_path: Path) -> pd.DataFrame:
    limit = PROFILE_SAMPLE_LIMIT or len(dataset)
    rows = [dataset.profile_one(halo_index).__dict__ for halo_index in range(limit)]
    df = pd.DataFrame(rows)
    ensure_dir(output_path.parent)
    df.to_csv(output_path, index=False)
    LOGGER.info(
        "Profiled radius %.1f: max nodes=%s max edges=%s statuses=%s",
        dataset.context_radius,
        df["n_nodes"].max(),
        df["n_edges"].max(),
        dict(df["status"].value_counts()),
    )
    return df


def save_split_indices(dataset_size: int, output_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(dataset_size)
    train_idx, test_idx = train_test_split(indices, test_size=0.1, random_state=RANDOM_STATE)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.05, random_state=RANDOM_STATE)
    ensure_dir(output_path.parent)
    np.savez(output_path, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
    return train_idx, val_idx, test_idx


def write_run_metadata(radius: float, scratch_run_dir: Path) -> None:
    save_json(
        {
            "sim": SIM.value,
            "z": Z,
            "context_radius": radius,
            "local_edge_radius": LOCAL_EDGE_RADIUS,
            "batch_size": BATCH_SIZE,
            "num_workers": NUM_WORKERS,
            "max_steps": MAX_STEPS,
            "max_edge_guard": MAX_EDGE_GUARD,
            "max_node_guard": MAX_NODE_GUARD,
            "max_context_nodes": MAX_CONTEXT_NODES,
            "run_mode": RUN_MODE,
            "scratch_run_dir": str(scratch_run_dir),
            "project_results_root": str(PROJECT_RESULTS_ROOT),
            "project_checkpoint_root": str(PROJECT_CHECKPOINT_ROOT),
        },
        scratch_run_dir / "config_snapshot.json",
    )


# ------------------
# Training
# ------------------
class MemoryLogger:
    def on_train_epoch_end(self, trainer, pl_module) -> None:
        LOGGER.info("Memory after train epoch %s: %.2f GB", trainer.current_epoch, memory_gb())
        gc.collect()

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        LOGGER.info("Memory after validation epoch %s: %.2f GB", trainer.current_epoch, memory_gb())
        gc.collect()


def train_dataset(
    dataset: StreamingHybridRadiusDataset,
    profile_df: pd.DataFrame,
    scratch_run_dir: Path,
) -> Optional[Path]:
    if (profile_df["status"] != "ok").any():
        LOGGER.warning("Skipping radius %.1f because profile guardrails failed.", dataset.context_radius)
        return None

    import lightning as L
    import wandb
    from lightning.pytorch.callbacks import Callback, EarlyStopping, LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import WandbLogger
    from models.gnn_model import GraphModel

    class LightningMemoryLogger(MemoryLogger, Callback):
        pass

    train_idx, val_idx, _ = save_split_indices(len(dataset), scratch_run_dir / "split_indices.npz")
    train_loader = DataLoader(
        Subset(dataset, train_idx.tolist()),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx.tolist()),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    model = GraphModel(
        batch_size=BATCH_SIZE,
        context=32,
        transforms=6,
        hidden_features=[128, 128, 128],
        node_features_dim=dataset.node_features_dim,
        node_features_hidden_dim=64,
        edge_features_hidden_dim=64,
        message_passing_steps=2,
        use_residual=True,
        aggregation_type="attention",
        pooling_type="central",
        dropout_rate=0.0,
    )

    ckpt_dir = ensure_dir(scratch_run_dir / "checkpoints")
    best_check = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename=f"best_{run_name(dataset.context_radius)}" + "-{step:02d}-{val_loss:.2f}-{mse_loss:.2f}",
        dirpath=ckpt_dir,
        verbose=True,
    )
    trainer = L.Trainer(
        max_steps=MAX_STEPS,
        logger=WandbLogger(project="DeepHalos", name=run_name(dataset.context_radius), log_model=False),
        log_every_n_steps=100,
        gradient_clip_val=1.0,
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            best_check,
            EarlyStopping(monitor="val_loss", patience=PATIENCE, mode="min", verbose=True),
            LightningMemoryLogger(),
        ],
        val_check_interval=0.1,
        default_root_dir=scratch_run_dir,
    )
    trainer.fit(model, train_loader, val_loader)
    wandb.finish()

    if best_check.best_model_path:
        project_ckpt_dir = ensure_dir(PROJECT_CHECKPOINT_ROOT / run_name(dataset.context_radius))
        best_path = Path(best_check.best_model_path)
        shutil.copy2(best_path, project_ckpt_dir / best_path.name)
        return project_ckpt_dir / best_path.name
    return None


# ------------------
# Main
# ------------------
def run_radius(radius: float) -> None:
    label = run_name(radius)
    scratch_run_dir = ensure_dir(SCRATCH_ROOT / label)
    project_run_dir = ensure_dir(PROJECT_RESULTS_ROOT / label)
    write_run_metadata(radius, scratch_run_dir)

    LOGGER.info("Starting %s for %s", RUN_MODE, label)
    LOGGER.info("Initial memory: %.2f GB", memory_gb())
    dataset = StreamingHybridRadiusDataset.from_catalogue(
        context_radius=radius,
        local_edge_radius=LOCAL_EDGE_RADIUS,
        hm_present=False,
        max_context_nodes=MAX_CONTEXT_NODES,
    )
    LOGGER.info("Loaded dataset with %s halos; memory %.2f GB", len(dataset), memory_gb())

    profile_df = profile_dataset(dataset, scratch_run_dir / "radius_profile.csv")
    torch.save(dataset.stats, scratch_run_dir / "feature_stats.pt")

    if RUN_MODE in {"train", "both"}:
        best_path = train_dataset(dataset, profile_df, scratch_run_dir)
        save_json({"best_checkpoint": str(best_path) if best_path else None}, scratch_run_dir / "train_summary.json")

    copy_small_outputs(scratch_run_dir, project_run_dir)
    LOGGER.info("Finished %s in %s", label, project_run_dir)


def main() -> None:
    setup_logging()
    start = time.time()
    LOGGER.info("High-radius sweep mode=%s", RUN_MODE)
    if RUN_MODE not in {"profile", "train", "both"}:
        raise ValueError("RUN_MODE must be one of: profile, train, both")
    for radius in CONTEXT_RADII:
        run_radius(radius)
    LOGGER.info("High-radius sweep finished in %s", timedelta(seconds=time.time() - start))


if __name__ == "__main__":
    main()
