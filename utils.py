# -----------------
# Imports
# -----------------
import numpy as np
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from dataclasses import dataclass, field

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# -----------------
# Better handling for args
# -----------------
@dataclass
class DataConfig:
    data_dir: str
    ckpt_dir: str
    model_type: str
    sim: str
    z: int
    graph_radius: float = field(default=2000) 
    train_sim: str = field(default=None)
    observables_only: bool = field(default=False)
    hm_present: bool = field(default=False)
    target: str = field(default='HaloMass')
    non_target: str = field(default='HaloMass_z0')
    box_size: int = field(default=None)
    
    VALID_MODEL_TYPES = {"GNN", "GRAPH", "FCN", "BASELINE"}
    VALID_SIMS = {"AST", "ASTRID", "TNG", "ILLUSTRIS", "ILLUSTRISTNG"}
    
    MODEL_MAPPING = {
        "GRAPH": "GNN", 
        "BASELINE": "FCN"}
    SIM_MAPPING = {
        "AST": "ASTRID", 
        "ILLUSTRIS": "TNG", 
        "ILLUSTRISTNG": "TNG"}
    
    BOXSIZE = {
    "ASTRID": 250_000,
    "TNG": 75_000,
    }
    
    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.ckpt_dir = Path(self.ckpt_dir)
        
        if self.z <3 or self.z>6:
            raise ValueError(f"Redshift {self.z} is invalid. Must be one of 3, 4, 5, 6.")
        
        self.model_type = self.model_type.upper()
        self.sim = self.sim.upper()

        if self.model_type not in self.VALID_MODEL_TYPES:
            raise ValueError(f"Invalid model_type '{self.model_type}'. Must be one of {self.VALID_MODEL_TYPES}.")
        if self.sim not in self.VALID_SIMS:
            raise ValueError(f"Invalid sim '{self.sim}'. Must be one of {self.VALID_SIMS}.")
        
        self.model_type = self.MODEL_MAPPING.get(self.model_type, self.model_type)
        self.sim = self.SIM_MAPPING.get(self.sim, self.sim)
        self.train_sim = self.train_sim or self.sim 
        self.box_size = self.BOXSIZE.get(self.sim)
        
        if self.hm_present:
            if self.sim == 'ASTRID':
                raise ValueError("Incompatible arguments: Only TNG has z=0 targets.")
            self.target = 'HaloMass_z0'
            self.non_target = 'HaloMass'

FILE_SUFFIXES = {
    'GNN': {
        'data': 'all_graphs.pt',
        'stats': 'feature_stats.pt'
    },
    'FCN': {
        'data': 'raw.csv',
        'stats': 'stats.csv',
        'halos': 'halos.pkl'
    }
}

# -----------------
# FUNCTIONS
# -----------------

# def convert_to_float32(data_list):
#     """Convert all tensor attributes to float32."""
#     for data in data_list:
#         for key, value in data:
#             if key not in ['central_mask', 'edge_index', 'feature_names']:
#                 if isinstance(value, torch.Tensor):
#                     data[key] = value.clone().detach().float()
#                 else:
#                     data[key] = torch.tensor(value).float()
#     return data_list

def convert_to_float32(data_list):
    """Convert all tensor attributes in data_list to float32, except for excluded keys."""
    excluded_keys = {'central_mask', 'edge_index', 'feature_names'}
    
    for data in data_list:
        for key, value in data.items():
            if key in excluded_keys:
                continue
            if isinstance(value, torch.Tensor):
                # 
                if value.dtype != torch.float32:
                    data[key] = value.clone().detach().float().to(device)
            else:
                data[key] = torch.tensor(value, dtype=torch.float32, device=device)
    
    return data_list

def read_cat_data(data_path: Path, min_mass: float = 1e7) -> np.ndarray:
    """Load and filter galaxy catalog."""
    data = np.load(data_path)
    mask = (data['GalaxyMass'] >= min_mass) 
    return data[mask]

def get_split_indices(length: int, random_state: int =42, test_size: float = 0.1, val_size: float = 0.05) -> np.ndarray:
    indices = np.arange(length)
    train_idx, test_idx = train_test_split(
        indices, 
        random_state=random_state, 
        test_size=test_size,
    )
    train_idx, val_idx = train_test_split(
        train_idx, 
        random_state=random_state, 
        test_size=val_size,
    )
    return train_idx, val_idx, test_idx

def get_datapath(config: DataConfig, stat: bool = False) -> Path:
    """Generates data directory path based on config."""
    # TODO THESE WILL LIKELY CHANGE
    model_dir = f"{config.model_type.lower()}_{config.graph_radius:.1f}/" if config.model_type == 'GNN' else "baseline/"
    simname = f"{config.train_sim}_z{config.z}/" if stat else f"{config.sim}_z{config.z}/"
    data_path = Path(config.data_dir / model_dir / simname)

    if not data_path.exists():
        raise ValueError(f"Invalid path {data_path}. Check parameters.")
    return data_path

def get_file(config: DataConfig, data_type: str) -> Path:
    """
    Retrieves file path for the specified data. Cross-testing optionality
    
    Args:
        config -> (DataConfig): Provides all requisite specifications.
        data_type -> (str): 
    """
    stat = True if data_type=='stats' else False
    file_dir = get_datapath(config, stat=stat)
    suffix = FILE_SUFFIXES.get(config.model_type, {}).get(data_type)

    if not suffix:
        raise ValueError(f"Invalid data_type '{data_type}' for model_type '{config.model_type}'.")
        
    file = next(file_dir.glob(f"*{suffix}"), None)
    if not file:
        raise ValueError(f"File {suffix} not found. Check parameters.")
    return Path(file)

def get_modelpath(config: DataConfig) -> Path:
    """Generates the model path for specified parameters."""
    suff_one = "_z0" if config.hm_present else ""
    suff_two = "_observable_features_only" if config.observables_only else ""
    model_dir = f"{config.model_type}_{config.sim}_z{config.z}_target_HaloMass{suff_one + suff_two}/"
    model_path = config.ckpt_dir / model_dir

    if not model_path.exists():
        raise ValueError(f"Invalid path {model_path}. Ensure only listed parameters are used.")
    return model_path