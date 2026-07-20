import numpy as np
import torch
from data.generate_gnn import build_graph

def test_build_graph_simple():
    # Two nodes: central at index 0, neighbor at index 1
    idx = [0, 1]
    halo_masses = np.array([1.0, 2.0])
    halo_masses_z0 = None
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    features = np.array([[10.0, 20.0], [30.0, 40.0]])
    feature_names = ["f1", "f2"]
    boxsize = 10.0
    radius = 1.0

    data = build_graph(
        idx, halo_masses, halo_masses_z0,
        positions, features, feature_names,
        boxsize, radius, max_galaxies=2
    )

    # Validate node features tensor shape
    assert isinstance(data.x, torch.Tensor)
    assert data.x.shape[0] == 2
    # Validate edge_index for complete graph of 2 nodes (2x4)
    assert data.edge_index.shape == (2, 4)
    # Validate edge_attr shape (4 edges, 4 features: 3 coords + distance)
    assert data.edge_attr.shape[1] == 4
    # Validate target
    assert data.y == halo_masses[0]
    # Validate global attributes
    assert torch.allclose(data.global_attr, torch.tensor([2/2], dtype=torch.float32))
