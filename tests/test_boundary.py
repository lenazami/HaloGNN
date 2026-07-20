import numpy as np
from data.generate_gnn import apply_periodic_boundary

def test_no_wrap():
    arr = np.array([0.2, -0.2, 0.4])
    out = apply_periodic_boundary(arr.copy(), boxsize=1.0)
    assert np.allclose(out, arr)

def test_wrap():
    arr = np.array([0.6, 0.9, -0.9])
    out = apply_periodic_boundary(arr.copy(), boxsize=1.0)
    # values >0.5 subtract boxsize, values < -0.5 add boxsize
    expected = np.array([-0.4, -0.1, 0.1])
    assert np.allclose(out, expected)
