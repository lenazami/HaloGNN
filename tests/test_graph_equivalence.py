# tests/test_graph_equivalence.py
import numpy as np
import torch
from types import SimpleNamespace
import pytest

from data.generate_gnn import build_graph, apply_periodic_boundary
from torch_cluster import radius_graph

def build_graph_radius_variant(idx, halo_masses, halo_masses_z0, positions, features,
                               feature_names, boxsize, radius, max_galaxies=100):
    # central distances & node features (NumPy → Torch)
    pos_np = positions[np.array(idx)]
    delta_central = apply_periodic_boundary(pos_np - pos_np[0], boxsize) / radius
    dist_central = np.linalg.norm(delta_central, axis=1).reshape(-1,1)
    x = np.hstack([features[idx], delta_central, dist_central])
    x = torch.tensor(x, dtype=torch.float32)

    # use radius_graph to build edges (C++ accelerated)
    pos = torch.tensor(pos_np / radius, dtype=torch.float32)
    edge_index = radius_graph(pos, r=1.0, loop=True)
    src, dst = edge_index

    # compute exactly the same edge_attr as original: 
    #   (pos_diff / (2*radius), norm) — note we need to reverse the scaling
    # first reconstruct raw diffs in original units:
    rel = (pos[src] - pos[dst]) * radius
    raw_rel = rel.numpy()
    edge_attr = torch.cat([
        torch.from_numpy(raw_rel / (2.0 * radius)),
        torch.norm(rel, dim=1).unsqueeze(1)
    ], dim=1)

    return x, edge_index.numpy(), edge_attr.numpy()

@pytest.fixture
def synthetic():
    # small group of 4 galaxies in a box of size 10
    N = 4
    boxsize, radius = 10.0, 5.0
    rng = np.random.RandomState(42)
    # structured array with required fields
    dtype = [
        ("GalaxyMass", float), ("HaloMass", float), ("FOFID", int),
        ("GalaxyPos", float, 3), ("GalaxyVel", float, 3),
        ("GalaxyRhalf", float), ("SFR", float),
        ("jwst_f090w", float), ("jwst_f150w", float),
        ("jwst_f277w", float), ("jwst_f444w", float)
    ]
    arr = np.zeros(N, dtype=dtype)
    arr["GalaxyMass"] = rng.rand(N) * 1e8 + 1e7
    arr["HaloMass"]   = rng.rand(N) * 1e9 + 1e8
    arr["FOFID"]      = np.arange(N)
    arr["GalaxyPos"]  = rng.rand(N,3) * boxsize
    arr["GalaxyVel"]  = rng.rand(N,3) * 100
    arr["GalaxyRhalf"]= rng.rand(N) * 10 + 1
    arr["SFR"]        = rng.rand(N) * 10
    for band in ("090w","150w","277w","444w"):
        arr[f"jwst_f{band}"] = rng.rand(N)
    # dummy features from your process_features (just identity arrays)
    # for simplicity we'll use one feature per galaxy
    features = np.vstack([arr["GalaxyMass"],]).T
    feature_names = ["GalaxyMass"]
    # all indices
    idx = list(range(N))
    return SimpleNamespace(arr=arr, boxsize=boxsize, radius=radius,
                           features=features, feature_names=feature_names,
                           halo_masses=arr["HaloMass"], halo_masses_z0=None,
                           idx=idx)

def test_build_graph_equivalence(synthetic):
    s = synthetic
    # old graph
    old = build_graph(
        s.idx, s.halo_masses, s.halo_masses_z0,
        s.arr["GalaxyPos"], s.features, s.feature_names,
        s.boxsize, s.radius
    )
    x_old = old.x.numpy()
    ei_old = old.edge_index.numpy()
    ea_old = old.edge_attr.numpy()

    # new graph
    x_new, ei_new, ea_new = build_graph_radius_variant(
        s.idx, s.halo_masses, s.halo_masses_z0,
        s.arr["GalaxyPos"], s.features, s.feature_names,
        s.boxsize, s.radius
    )

    # 1) Node‐feature shapes match
    assert x_old.shape == x_new.shape

    # 2) Central‐distance columns match (the last column)
    assert np.allclose(x_old[:,-1], x_new[:,-1], atol=1e-6)

    # 3) Shared edges agree on edge attributes
    #    find the set of edges that appear in both
    set_old = set(map(tuple, ei_old.T.tolist()))
    set_new = set(map(tuple, ei_new.T.tolist()))
    shared = list(set_old & set_new)
    assert shared, "No shared edges to compare!"
    # pick a handful to compare
    for u,v in shared[:10]:
        mask_old = np.all(ei_old.T == [u,v], axis=1)
        mask_new = np.all(ei_new.T == [u,v], axis=1)
        assert np.allclose(ea_old[mask_old], ea_new[mask_new], atol=1e-6)

    # 4) New graph has fewer or equal edges than old (radius filtering)
    assert ei_new.shape[1] <= ei_old.shape[1]