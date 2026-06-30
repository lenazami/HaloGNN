import pytest
from utils_old.config import DataConfig
from pathlib import Path

def test_dataconfig_valid():
    cfg = DataConfig(
        root=Path("."),
        model_type="GNN",
        sim="TNG",
        z=4,
    )
    assert cfg.box_size == 75000
    assert cfg.sim == "TNG"
    assert cfg.model_type == "GNN"

@pytest.mark.parametrize("sim", ["FOO", "ILLUSTRISTNG"])
def test_dataconfig_invalid_sim(sim):
    with pytest.raises(ValueError):
        DataConfig(
            root=Path("."),
            model_type="GNN",
            sim=sim,
            z=4,
        )

@pytest.mark.parametrize("z", [2, 7, 0])
def test_dataconfig_invalid_z(z):
    with pytest.raises(ValueError):
        DataConfig(
            root=Path("."),
            model_type="GNN",
            sim="TNG",
            z=z,
        )
