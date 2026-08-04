# src/halo/config.py
# helpers for pathfinding, simulation and model validations
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

MODELS = ("Full", "Graph")

# -----------
# config
# -----------
def load_config():
    """read config yaml"""
    with open(PATHS.config) as file:
        return yaml.safe_load(file)

def validate_config(cfg: dict[str, Any]) -> None:
    """check config combinations we do not support."""
    sim = cfg["simulation"]["sim"]
    z = int(cfg["simulation"]["z"])
    model = cfg["model"]

    # bad combos
    if z == 3 and sim == "TNG":
        raise ValueError("Redshift 3 incompatible with TNG")
    if model["hm_present"] and sim != "TNG":
        raise ValueError("hm_present=True is only supported for TNG.")
    
# -----------
# config classes
# -----------
@dataclass(frozen=True)
class Simulation:
    name: str
    z: int
    boxsize: int
    astrid_id: str
    catalogue_dir: Path
    catalogue_paths: str

    @classmethod
    def from_cfg(cls, cfg: dict[str, Any] | None = None, **overrides: Any) -> "Simulation":
        cfg = load_config() if cfg is None else cfg
        values = dict(cfg["simulation"])
        values.update({key: value for key, value in overrides.items() if value is not None})

        name = str(values["sim"])
        z = int(values["z"])
        boxsizes = values["boxsizes"]
        catalogue_ids = values["astrid_catalogue_ids"]
        templates = values["catalogue_paths"]

        catalogue_dir = Path(cfg["paths"]["catalogue_dir"]).expanduser()
        return cls(
            name=name,
            z=z,
            boxsize=int(boxsizes[name]),
            astrid_id=str(catalogue_ids[z]),
            catalogue_dir=catalogue_dir,
            catalogue_paths=str(templates[name]),
        )

    @property
    def tag(self) -> str:
        return f"{self.name}_z{self.z}"

    def catalogue_path(self, *, validate: bool = True) -> Path:
        filename = self.catalogue_paths.format(
            sim=self.name, z=self.z, astrid_id=self.astrid_id
        )
        path = self.catalogue_dir / filename
        if validate and not path.exists():
            raise FileNotFoundError(f"Catalogue not found: {path}")
        return path


@dataclass(frozen=True)
class Model:
    name: str
    graph_radius: float
    observables_only: bool
    hm_present: bool
    train_sim: str | None = None
    # hyperparameters (overridable via model.hyperparameters in configs.yaml)
    max_steps: int = 100_000
    batch_size: int = 64
    patience: int = 50
    context_dim: int = 32
    hidden_dims: tuple[int, ...] = (128, 128, 128)
    flow_transforms: int = 6
    learning_rate: float = 5e-4
    scheduler_patience: int = 5

    @classmethod
    def from_cfg(cls, cfg: dict[str, Any] | None = None, **overrides: Any) -> "Model":
        cfg = load_config() if cfg is None else cfg
        values = dict(cfg["model"])
        values.update({key: value for key, value in overrides.items() if value is not None})
        hp = dict(values.get("hyperparameters") or {})
        if "hidden_dims" in hp:
            hp["hidden_dims"] = tuple(hp["hidden_dims"])   # yaml gives a list
        return cls(
            name=str(values["model_type"]),
            graph_radius=float(values["graph_radius"]),
            observables_only=bool(values["observables_only"]),
            hm_present=bool(values["hm_present"]),
            train_sim=values.get("train_sim"),
            **hp,
        )

    @property
    def label_field(self) -> str:
        tag = "_z0" if self.hm_present else ""
        return f"HaloMass{tag}"

    def tag(self, **labels) -> str:
        tag = "".join(f"_{value}" for value in labels.values() if value not in (False,None))
        return f"{self.name.lower()}{tag}"

    # def tag(self, **labels) -> str:
    #     parts = [self.name.lower()]

    #     for key, value in labels.items():
    #         if value is True:
    #             parts.append(key)
    #         elif value not in (False, None):
    #             parts.append(f"{key}-{value}")

    #     return "_".join(parts)
# -----------
# pathing
# -----------
@dataclass(frozen=True)
class Paths:
    """repo layout pathfinding"""

    root: Path
    store: Path

    @classmethod
    def from_file(cls, file: str | Path) -> "Paths":
        root = Path(file).resolve().parents[2]
        store = root
        cfg_path = root / "configs.yaml"
        if cfg_path.exists():
            with open(cfg_path) as fh:
                cfg = yaml.safe_load(fh) or {}
            store_dir = (cfg.get("paths") or {}).get("store_dir")
            if store_dir:
                store = Path(store_dir).expanduser()
        return cls(root=root, store=store)

    @property
    def config(self) -> Path:
        return self.root / "configs.yaml"

    @property
    def data(self) -> Path:
        return self.store / "data"

    @property
    def outputs(self) -> Path:
        return self.root / "outputs"

    @property
    def results(self) -> Path:
        # analysis tables + figures
        return self.root / "results"

    def file_stem(self, model: Model, sim: Simulation, **labels):
        labels = {
            "r": model.graph_radius if model.name == "Graph" else None,
            "hmz0": model.hm_present,
            "obs": model.observables_only,
            "train": (
                model.train_sim
                if model.train_sim is not None and model.train_sim != sim.name
                else None
            ),
            **labels,
        }
        model_tag = model.tag(**labels)
        return f"{model_tag}/{sim.tag}", f"{model_tag}_{sim.tag}"
    
    def graphs(self, model: Model, sim: Simulation, **labels) -> Path:
        dir, file_tag = self.file_stem(model, sim, **labels)
        return self.data / dir / f"{file_tag}_graphs.pt"

    def graph_stats(self, model: Model, sim: Simulation, **labels) -> Path:
        dir, file_tag = self.file_stem(model, sim, **labels)
        return self.data / dir / f"{file_tag}_stats.pt"
    
    def output_dir(self, kind: str, model: Model, sim: Simulation, **labels) -> Path:
        dir, _ = self.file_stem(model, sim, **labels)
        return self.outputs / kind / dir

    def checkpoint(self, model: Model, sim: Simulation, **labels) -> Path:
        return self.output_dir("checkpoints", model, sim, **labels)

    def catalog(self, sim: Simulation) -> Path:
        # written once per sim/redshift
        return self.data / "catalog" / f"catalog_{sim.tag}.pt"

    def membership(self, model: Model, sim: Simulation, **labels) -> Path:
        dir, file_tag = self.file_stem(model, sim, **labels)
        return self.data / dir / f"{file_tag}_membership.pt"

PATHS = Paths.from_file(__file__)
