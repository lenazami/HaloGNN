# utils / config.py
'''Configures data loading and model training presets. Defaults imported from configs.yaml'''

# -----------------
# Imports
# -----------------
from pathlib import Path
import yaml
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Literal
# import os

# load defaults
_cfg_path =  Path(__file__).parent.parent / "configs.yaml"
_cfg = yaml.safe_load(_cfg_path.read_text())

class Simulation(Enum):
    TNG = "TNG"
    ASTRID = "ASTRID"

class ModelType(Enum):
    GNN = "GNN"
    FCN = "FCN"

# constants
BOX_SIZES = {Simulation.TNG: 75_000,Simulation.ASTRID: 250_000}
# REDSHIFTS = [3, 4, 5, 6]
ASTRID_CAT_IDS = {3: "214", 4: "147", 5: "107", 6: "047"}

@dataclass
class Config:
    # user-specified paths
    catalogue_dir: Path = field(default_factory=lambda: Path(_cfg.get("catalogue_dir", ".")))
    project_dir: Path = field(default_factory=lambda: Path(_cfg.get("project_dir", ".")))
    # heavy artifacts live OUTSIDE the code repo (see REFACTOR_PLAN.md); fall back to project_dir if unset
    data_root: Path = field(default_factory=lambda: Path(_cfg.get("data_root", _cfg.get("project_dir", "."))))
    out_root: Path = field(default_factory=lambda: Path(_cfg.get("out_root", _cfg.get("project_dir", "."))))
    
    # required fields, can be set with defaults    
    model_type: ModelType = ModelType(_cfg["model_type"])
    sim: Simulation = Simulation(_cfg["sim"])
    z: Literal[3, 4, 5, 6] = _cfg["z"]

    # optional parameters
    graph_radius: float = _cfg.get("graph_radius", 2000.0)
    observables_only: bool = _cfg.get("observables_only", False)
    hm_present: bool = _cfg.get("hm_present", False)
    min_mass: float = _cfg.get("min_mass", 1e7)
    train_sim: Optional[str] = None
    
    # computed paths - relative to project_dir
    data_dir: Path = field(init=False)
    stats_dir: Path = field(init=False)
    checkpoint_dir: Path = field(init=False)
    results_dir: Path = field(init=False)
    
    # other computed fields
    box_size: int = field(init=False)
    astrid_id: str = field(init=False)

    def __post_init__(self):
        # validation
        if self.z == 3 and self.sim == Simulation.TNG:
            raise ValueError(f"Redshift 3 incompatible with TNG")
        if self.z not in ASTRID_CAT_IDS:
            raise ValueError(f"Invalid redshift: {self.z}")
        if self.hm_present and self.sim != Simulation.TNG:
            raise ValueError(f"hm_present=True is only supported for TNG.")
        
        # ensure paths exist
        self.project_dir = self.project_dir.expanduser().resolve()
        self.catalogue_dir = self.catalogue_dir.expanduser().resolve()
        self.data_root = self.data_root.expanduser().resolve()
        self.out_root = self.out_root.expanduser().resolve()
        
        if not self.catalogue_dir.exists():
            raise FileNotFoundError(
                f"Catalogue directory not found: {self.catalogue_dir}\n"
                f"Please download catalogues and update 'catalogue_dir' in configs.yaml"
            )
            
        # set up directory structure
        self._setup_directories()
        
        # derived
        self.box_size = BOX_SIZES[self.sim]
        self.astrid_id = ASTRID_CAT_IDS[self.z]
        self.train_sim  = self.train_sim or self.sim.value
    
    def _setup_directories(self):
        '''creates directory structure'''
        # model specific directory name
        model_dir = (
            f"gnn_{self.graph_radius:.1f}" 
            if self.model_type==ModelType.GNN 
            else "fcn"
        )
        
        # Roots point OUTSIDE the repo so heavy artifacts are decoupled from the code.
        # data + stats under data_root; checkpoints + results under out_root
        # TODO(refactor): the leaf names below (processed/stats/<model_dir>, and the checkpoint
        # leaf names from get_checkpoint_path) do NOT yet match the on-disk artifacts.
        # See REFACTOR_PLAN.md -> "Naming reconciliation" before relying on these to resolve.
        self.data_dir = self.data_root / "data" / "processed" / model_dir
        self.stats_dir = self.data_root / "data" / "stats" / model_dir
        self.checkpoint_dir = self.out_root / "checkpoints" / model_dir
        self.results_dir = self.out_root / "results"
        
        # create directories
        for dir_path in [self.data_dir, self.stats_dir, self.checkpoint_dir, 
                         self.results_dir / "figures", 
                         self.results_dir / "metrics",
                         self.results_dir / "reports"]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def get_catalogue_path(self) -> Path:
        """Get path to raw catalogue file."""
        # TODO: once the galaxyGNN perms gets sorted this will change to 
        # root/catalogs/(astrid or tng)
        # for now i dont have perms so im just doing this separately 
        # but ive commented out what it should be in the future for ease
        # (literally everything else before that line can be deleted)
       
        if self.sim == Simulation.ASTRID:
            filename = f"ASTRID_galaxy_halo_catalog_{self.astrid_id}.npy"
            path = self.catalogue_dir / "catalogs/high-z-jwst-ASTRID" / filename
        else:  # TNG
            filename = f"TNG100_galaxy_halo_catalog_z{self.z}.npy"
            path = self.catalogue_dir / "high-z-jwst-TNG" / filename
        
        if not path.exists():
            raise FileNotFoundError(
                f"Catalogue not found: {path}\n"
                f"Expected structure: {self.catalogue_dir}/{self.sim.value}/{filename}"
            )
        return path

    def get_data_path(self) -> Path:
        """Get path for processed data."""
        return self.data_dir / f"{self.sim.value}_z{self.z}"
    
    def get_stats_path(self) -> Path:
        """Get path for statistics."""
        sim = self.train_sim
        return self.stats_dir / f"{sim}_z{self.z}"
    
    def get_checkpoint_path(self, create: bool = False) -> Path:
        """Get checkpoint directory path."""
        ckpt_name = f"{self.model_type.value}_{self.sim.value}_z{self.z}"
        if self.hm_present:
            ckpt_name += "_hmz0"
        if self.observables_only:
            ckpt_name += "_obs"
        
        path = self.checkpoint_dir / ckpt_name
        if create:
            path.mkdir(parents=True, exist_ok=True)
        elif not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path
    
    def label_field(self) -> str:
        return "HaloMass_z0" if self.hm_present else "HaloMass"
    
    # TODO: check if feautre_field is necessary
    def feature_field(self) -> str:
        return "HaloMass" if self.hm_present else "HaloMass_z0"