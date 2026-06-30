import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from models.gnn_model import GraphModel
from scripts.high_radius_sweep import (
    SPATIAL_FEATURE_NAMES,
    Simulation,
    StreamingHybridRadiusDataset,
    build_hybrid_edges,
    minimum_image_delta,
    profile_dataset,
)


def synthetic_catalog() -> np.ndarray:
    dtype = [
        ("GalaxyMass", float),
        ("HaloMass", float),
        ("FOFID", int),
        ("GalaxyPos", float, 3),
        ("GalaxyVel", float, 3),
        ("GalaxyRhalf", float),
        ("SFR", float),
        ("jwst_f090w", float),
        ("jwst_f150w", float),
        ("jwst_f277w", float),
        ("jwst_f444w", float),
    ]
    arr = np.zeros(6, dtype=dtype)
    arr["FOFID"] = [1, 1, 1, 2, 2, 2]
    arr["HaloMass"] = [10, 12, 11, 20, 21, 22]
    arr["GalaxyMass"] = [3, 5, 4, 4, 5, 6]
    arr["GalaxyPos"] = [
        [0, 0, 0],
        [10, 0, 0],
        [18, 0, 0],
        [100, 0, 0],
        [110, 0, 0],
        [130, 0, 0],
    ]
    arr["GalaxyVel"] = np.arange(18).reshape(6, 3)
    arr["GalaxyRhalf"] = np.linspace(1, 2, 6)
    arr["SFR"] = np.linspace(1, 6, 6)
    for band in ["jwst_f090w", "jwst_f150w", "jwst_f277w", "jwst_f444w"]:
        arr[band] = np.linspace(0.1, 0.6, 6)
    return arr


def edge_set(edge_index: torch.Tensor) -> set[tuple[int, int]]:
    return set(map(tuple, edge_index.T.tolist()))


def test_hybrid_edges_connect_center_and_local_noncentral_pairs():
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [18.0, 0.0, 0.0],
            [80.0, 0.0, 0.0],
        ]
    )
    edge_index, edge_attr = build_hybrid_edges(
        positions,
        center_index=0,
        context_radius=100.0,
        local_edge_radius=10.0,
        box_size=1000.0,
    )

    edges = edge_set(edge_index)
    assert {(0, 1), (1, 0), (0, 2), (2, 0), (0, 3), (3, 0)} <= edges
    assert (1, 2) in edges
    assert (2, 1) in edges
    assert (1, 3) not in edges
    assert all(src != dst for src, dst in edges)
    assert len(edges) == edge_index.shape[1]
    assert edge_attr.shape == (edge_index.shape[1], 4)


def test_hybrid_edges_use_periodic_boundary_for_edge_attr():
    positions = np.array(
        [
            [1.0, 0.0, 0.0],
            [99.0, 0.0, 0.0],
        ]
    )
    edge_index, edge_attr = build_hybrid_edges(
        positions,
        center_index=0,
        context_radius=10.0,
        local_edge_radius=2.5,
        box_size=100.0,
    )

    edges = edge_index.T.tolist()
    center_to_neighbor = edges.index([0, 1])
    neighbor_to_center = edges.index([1, 0])
    assert torch.isclose(edge_attr[center_to_neighbor, 0], torch.tensor(2.0 / 20.0))
    assert torch.isclose(edge_attr[neighbor_to_center, 0], torch.tensor(-2.0 / 20.0))
    assert np.allclose(minimum_image_delta(positions[0] - positions[1], 100.0), [2.0, 0.0, 0.0])


def test_streaming_dataset_returns_valid_graph_without_saving_list():
    dataset = StreamingHybridRadiusDataset(
        synthetic_catalog(),
        sim=Simulation.ASTRID,
        z=4,
        context_radius=25.0,
        local_edge_radius=12.0,
    )

    graph = dataset[0]
    assert graph.x.shape[1] == len(dataset.feature_names) + len(SPATIAL_FEATURE_NAMES)
    assert graph.edge_index.shape[0] == 2
    assert graph.edge_attr.shape[1] == 4
    assert graph.central_mask.sum().item() == 1
    assert graph.feature_names == dataset.graph_feature_names
    assert graph.y.ndim == 0


def test_graph_model_forward_on_synthetic_high_radius_graph():
    dataset = StreamingHybridRadiusDataset(
        synthetic_catalog(),
        sim=Simulation.ASTRID,
        z=4,
        context_radius=25.0,
        local_edge_radius=12.0,
    )
    batch = next(iter(DataLoader([dataset[0]], batch_size=1)))
    model = GraphModel(
        node_features_dim=dataset.node_features_dim,
        context=32,
        transforms=2,
        hidden_features=[16, 16],
        node_features_hidden_dim=16,
        edge_features_hidden_dim=16,
        message_passing_steps=1,
        aggregation_type="attention",
        pooling_type="central",
    )
    model.eval()

    with torch.no_grad():
        summary = model(batch)

    assert summary.shape == (1, 32)


def test_profile_dataset_writes_expected_columns(tmp_path):
    dataset = StreamingHybridRadiusDataset(
        synthetic_catalog(),
        sim=Simulation.ASTRID,
        z=4,
        context_radius=25.0,
        local_edge_radius=12.0,
    )
    output_path = tmp_path / "radius_profile.csv"
    df = profile_dataset(dataset, output_path)
    loaded = pd.read_csv(output_path)

    expected = {"radius", "halo_index", "n_nodes", "n_edges", "max_distance", "status"}
    assert expected <= set(df.columns)
    assert expected <= set(loaded.columns)
    assert set(loaded["status"]) == {"ok"}
