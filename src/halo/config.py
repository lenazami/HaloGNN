"""Small, YAML-driven configuration helpers kept separate from ``utils.config``.

The public objects are intentionally similar to ``utils_rsd``::

    cfg = load_config()
    sim = Simulation.from_cfg(cfg)
    model = Model.from_cfg(cfg)
    paths = Paths.from_file(__file__)
    data = paths.data_path(sim, model)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# -----------
# config
# -----------
def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read ``configs.yaml`` and return its plain dictionary."""
    path = Path(path) if path is not None else Path(__file__).resolve().parents[2] / "configs.yaml"
    with path.open() as file:
        config = yaml.safe_load(file)
    validate_config(config)
    return config

def validate_config(cfg: dict[str, Any]) -> None:
    """check config combinations we do not support."""
    simulation = cfg["simulation"]
    model = cfg["model"]
    sim = simulation["sim"]
    z = int(simulation["z"])

    if sim not in simulation["boxsizes"]:
        raise ValueError(f"Unknown simulation: {sim}")
    if z not in {int(value) for value in simulation["astrid_catalogue_ids"]}:
        raise ValueError(f"Invalid redshift: {z}")
    if z == 3 and sim == "TNG":
        raise ValueError("Redshift 3 incompatible with TNG")
    if float(model["graph_radius"]) <= 0:
        raise ValueError("graph_radius must be positive")
    if float(model["min_mass"]) <= 0:
        raise ValueError("min_mass must be positive")
    if model["hm_present"] and sim != "TNG":
        raise ValueError("hm_present=True is only supported for TNG.")

# -----------
# simulation
# -----------
@dataclass(frozen=True)
class Simulation:
    name: str
    z: int
    boxsize: int
    astrid_id: str
    catalogue_dir: Path
    catalogue_template: str

    @classmethod
    def from_cfg(cls, cfg: dict[str, Any] | None = None, **overrides: Any) -> "Simulation":
        cfg = load_config() if cfg is None else cfg
        values = dict(cfg["simulation"])
        values.update({key: value for key, value in overrides.items() if value is not None})

        name = str(values["sim"])
        z = int(values["z"])
        boxsizes = values["boxsizes"]
        catalogue_ids = values["astrid_catalogue_ids"]
        templates = values["catalogue_templates"]
        if name not in boxsizes:
            raise ValueError(f"Unknown simulation: {name}")
        if z not in {int(value) for value in catalogue_ids}:
            raise ValueError(f"Invalid redshift: {z}")

        catalogue_dir = Path(cfg["paths"]["catalogue_dir"]).expanduser()
        if not catalogue_dir.is_absolute():
            catalogue_dir = Paths.from_file(__file__).root / catalogue_dir
        return cls(
            name=name,
            z=z,
            boxsize=int(boxsizes[name]),
            astrid_id=str(catalogue_ids[z]),
            catalogue_dir=catalogue_dir,
            catalogue_template=str(templates[name]),
        )

    @property
    def tag(self) -> str:
        return f"{self.name}_z{self.z}"

    @property
    def sim(self) -> str:
        return self.name

    def catalogue_path(self, *, validate: bool = True) -> Path:
        filename = self.catalogue_template.format(
            sim=self.name, z=self.z, astrid_id=self.astrid_id
        )
        path = self.catalogue_dir / filename
        if validate and not path.exists():
            raise FileNotFoundError(f"Catalogue not found: {path}")
        return path

# -----------
# model
# -----------
@dataclass(frozen=True)
class Model:
    name: str
    graph_radius: float
    observables_only: bool
    hm_present: bool
    min_mass: float
    train_sim: str | None = None

    @classmethod
    def from_cfg(cls, cfg: dict[str, Any] | None = None, **overrides: Any) -> "Model":
        cfg = load_config() if cfg is None else cfg
        values = dict(cfg["model"])
        values.update({key: value for key, value in overrides.items() if value is not None})
        return cls(
            name=str(values["model_type"]),
            graph_radius=float(values["graph_radius"]),
            observables_only=bool(values["observables_only"]),
            hm_present=bool(values["hm_present"]),
            min_mass=float(values["min_mass"]),
            train_sim=values.get("train_sim"),
        )

    @property
    def directory_name(self) -> str:
        if self.name == "GNN":
            return f"gnn_{self.graph_radius:.1f}"
        return self.name.lower()

    @property
    def model_type(self) -> str:
        return self.name

    @property
    def label_field(self) -> str:
        return "HaloMass_z0" if self.hm_present else "HaloMass"

    @property
    def feature_field(self) -> str:
        return "HaloMass" if self.hm_present else "HaloMass_z0"

    def checkpoint_name(self, simulation: Simulation) -> str:
        name = f"{self.name}_{simulation.name}_z{simulation.z}"
        if self.hm_present:
            name += "_hmz0"
        if self.observables_only:
            name += "_obs"
        return name


# -----------
# pathing
# -----------
@dataclass(frozen=True)
class Paths:
    """repo layout pathfinding"""

    root: Path

    @classmethod
    def from_file(cls, file: str | Path) -> "Paths":
        # config_refactor.py lives at <repo>/src/halo/config_refactor.py.
        return cls(Path(file).resolve().parents[2])

    @property
    def config(self) -> Path:
        return self.root / "configs.yaml"

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def outputs(self) -> Path:
        return self.root / "outputs"
    
    # @property
    # def processed(self) -> Path:
    #     return self.data / "processed"

    @property
    def stats(self) -> Path:
        return self.data / "stats"

    @property
    def checkpoints(self) -> Path:
        return self.outputs / "checkpoints"

    def data_directory(self, model: Model) -> Path:
        return self.data / model.directory_name

    def stats_directory(self, model: Model) -> Path:
        return self.stats / model.directory_name

    def checkpoint_directory(self, model: Model) -> Path:
        return self.checkpoints / model.directory_name

    def data_path(self, simulation: Simulation, model: Model) -> Path:
        return self.data_directory(model) / simulation.tag

    def stats_path(self, simulation: Simulation, model: Model) -> Path:
        train_sim = model.train_sim or simulation.name
        return self.stats_directory(model) / f"{train_sim}_z{simulation.z}"

    def checkpoint_path(
        self, simulation: Simulation, model: Model, *, create: bool = False
    ) -> Path:
        path = self.checkpoint_directory(model) / model.checkpoint_name(simulation)
        if create:
            path.mkdir(parents=True, exist_ok=True)
        elif not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    def create_output_directories(self, model: Model) -> None:
        directories = (
            self.data_directory(model),
            self.stats_directory(model),
            self.checkpoint_directory(model),
            *(self.outputs / name for name in ("figures", "metrics", "reports")),
        )
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


PATHS = Paths.from_file(__file__)


# # -----------
# # compatibility helpers
# # -----------
# def boxsize(cfg: dict[str, Any]) -> int:
#     return Simulation.from_cfg(cfg).boxsize


# def catalogue_path(cfg: dict[str, Any], *, validate: bool = True) -> Path:
#     return Simulation.from_cfg(cfg).catalogue_path(validate=validate)


# def model_directory(cfg: dict[str, Any]) -> str:
#     return Model.from_cfg(cfg).directory_name


# def label_field(cfg: dict[str, Any]) -> str:
#     return Model.from_cfg(cfg).label_field


# def feature_field(cfg: dict[str, Any]) -> str:
#     return Model.from_cfg(cfg).feature_field


# def data_directory(cfg: dict[str, Any]) -> Path:
#     return PATHS.data_directory(Model.from_cfg(cfg))


# def stats_directory(cfg: dict[str, Any]) -> Path:
#     return PATHS.stats_directory(Model.from_cfg(cfg))


# def checkpoint_directory(cfg: dict[str, Any]) -> Path:
#     return PATHS.checkpoint_directory(Model.from_cfg(cfg))


# def results_directory(cfg: dict[str, Any]) -> Path:
#     return PATHS.results


# def data_path(cfg: dict[str, Any]) -> Path:
#     return PATHS.data_path(Simulation.from_cfg(cfg), Model.from_cfg(cfg))


# def stats_path(cfg: dict[str, Any]) -> Path:
#     return PATHS.stats_path(Simulation.from_cfg(cfg), Model.from_cfg(cfg))


# def checkpoint_path(cfg: dict[str, Any], *, create: bool = False) -> Path:
#     return PATHS.checkpoint_path(
#         Simulation.from_cfg(cfg), Model.from_cfg(cfg), create=create
#     )


# def create_output_directories(cfg: dict[str, Any]) -> None:
#     PATHS.create_output_directories(Model.from_cfg(cfg))
