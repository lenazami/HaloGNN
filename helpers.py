# -----------------
# Unitary utils to limit differences across data+models
# -----------------

# -----------------
# Imports
# -----------------
# general
import pandas as pd
from operator import itemgetter

# torch
from torch.utils.data import TensorDataset, DataLoader
from torch_geometric.loader import DataLoader as GeomDataLoader
from lightning import Trainer

# coverage
from lampe.diagnostics import expected_coverage_mc
from lampe.plots import coverage_plot

# internal
from utils import *

from gnn_model import GraphModel
from fcn_model import FlowModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# -----------------
# pre-training
# -----------------

# -------- load data --------
def read_data(config: DataConfig):
    filepath = get_file(config, 'data')
    if config.model_type == 'FCN':
        return pd.read_csv(filepath)
    data = torch.load(filepath)
    return convert_to_float32(data)

def load_train_stats(config: DataConfig):
    '''Load halo statistics for a given model type'''
    filepath = get_file(config, 'stats')
    if config.model_type == 'FCN':
        return pd.read_csv(filepath, index_col=0)
    return torch.load(filepath)

# -------- standardize features --------
def standardize_features(data, config: DataConfig):
    train_stats = load_train_stats(config)
    
    if config.model_type=='FCN':
        cols = list(data.columns)[:-1]
        data[cols] = (data[cols] - train_stats["mean"][cols]) / train_stats["std"][cols]
        return data
    
    elif config.model_type=='GNN':
        feature_names = data[0].feature_names
        # find feature indices in stats
        feature_indices = [train_stats['feature_names'].index(name) for name in feature_names if name in train_stats['feature_names']]
        means = torch.tensor(train_stats['means'], dtype=torch.float32, device=device)  # or use graph.x.device
        stds = torch.tensor(train_stats['stds'], dtype=torch.float32, device=device)
        halo_mass_idx = train_stats['feature_names'].index('HaloMass')
        
        for graph in data:
            graph = graph.to(device)
            graph.x[:,feature_indices] = (graph.x[:, feature_indices] - means[feature_indices]) / stds[feature_indices]
            graph.y = (graph.y - means[halo_mass_idx]) / stds[halo_mass_idx]
        return data

    

def unstandardize_features(data, config: DataConfig):
    '''Unstandardizes model predictions'''
    train_stats = load_train_stats(config)

    if config.model_type=='FCN':
        mean = train_stats.loc['HaloMass', 'mean']
        std = train_stats.loc['HaloMass', 'std']
        return (data * std) + mean
    elif config.model_type=='GNN':
        means = train_stats['means'].astype(np.float32)
        stds = train_stats['stds'].astype(np.float32)
        halo_mass_idx = train_stats['feature_names'].index('HaloMass')
        return data * stds[halo_mass_idx] + means[halo_mass_idx]
        
def observable_features(data, config: DataConfig):
    """Masks the non-observable features for a given model type and simulation."""
    features = {
        "FCN": ["HaloMass", "GalaxyMass_Sum", "GalaxyMass_Max", "GalaxyMass_Mean",
                "SFR_Sum", "SFR_Max", "SFR_Mean", "Velocity_Dispersion",
                "Velocity_Max", "Velocity_Mean", "HaloMass_z0"],
        "GNN": ["HaloMass", "GalaxyVel_1", "GalaxyVel_2", "GalaxyVel_3",
                "GalaxyRhalf", "GalaxyMass", "GalaxyVel", "SFR"]
    }
    features_to_mask = features[config.model_type]

    # FCN
    if config.model_type == 'FCN':
        return data.drop(columns=features_to_mask, errors='ignore')

    # GNN
    masked_data_list = []
    kept_indices = [i for i, name in enumerate(data[0].feature_names) if name not in features_to_mask]
        
    for dat in data:
        temp = dat.clone()
        temp.x = dat.x[:, kept_indices]
        temp.feature_names = [dat.feature_names[i] for i in kept_indices]
        masked_data_list.append(temp)

    return masked_data_list

# -------- dataloaders --------
def create_fcn_dataloader(config: DataConfig, data, batch_size: int, shuffle: bool, index=None) -> DataLoader:
    if index is not None:
        data = data.iloc[index]

    features = data.drop(columns=[config.target, config.non_target], errors='ignore')
    targets = data[config.target]

    dataset = TensorDataset(
        torch.tensor(features.to_numpy(), dtype=torch.float32, device=device),
        torch.tensor(targets.to_numpy(), dtype=torch.float32, device=device).unsqueeze(-1),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def create_gnn_dataloader(config: DataConfig, data, batch_size: int=64, shuffle: bool=False, index=None) -> GeomDataLoader:
    if index is not None:
        data = itemgetter(*index)(data)
    loader = GeomDataLoader(data, batch_size=batch_size, shuffle=shuffle)
    return loader

def load_data(
    config: DataConfig,
    batch_size: int = 64, 
    only_test: bool = None,
    all_data: bool = None
) -> DataLoader:
    """
    Loads data for a specific model, simulation suite and redshift.

    Args:
        config (DataConfig): Handles kwargs.
        
        batch_size (int): Number of items per batch. (default: 64)
        only_test (bool, optional): if true, only returns the test dataloader.
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Three PyTorch DataLoaders for training, validation, and testing.
    """
    # reads data
    data = read_data(config)

    # standardizes features
    data = standardize_features(data, config)
    if config.observables_only:
        data = observable_features(data, config)
    
    # dataloaders
    create_dataloader = create_fcn_dataloader if config.model_type == 'FCN' else create_gnn_dataloader

    if all_data==True:
        return create_dataloader(config, data, batch_size, shuffle=False)
    else:
        train_idx, val_idx, test_idx = get_split_indices(len(data), test_size=0.1 if config.sim == 'TNG' else 0.01, val_size=0.05)
        test_loader = create_dataloader(config, data, batch_size, shuffle=False, index=test_idx)
        if only_test==True:
            return test_loader
        train_loader = create_dataloader(config, data, batch_size, shuffle=True, index=train_idx)
        val_loader = create_dataloader(config, data, batch_size, shuffle=False, index=val_idx)
        
        return train_loader, val_loader, test_loader

# -----------------
# models
# -----------------
def load_model(config: DataConfig):
    data_path = get_modelpath(config)
    
    # find .ckpt file in folder
    ckpt_file = next(data_path.glob('*.ckpt'))
    
    if config.model_type.upper()=='GNN':
        model = GraphModel.load_from_checkpoint(ckpt_file)
        return model
    elif config.model_type.upper()=='FCN':
        model = FlowModel.load_from_checkpoint(ckpt_file)
        return model
    else:
        raise ValueError('Invalid model type.')

# -----------------
# post-training
# -----------------
def get_predictions(config: DataConfig, test_loader, model) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    trainer = Trainer()
    
    predictions_dict = trainer.predict(model, test_loader)
    predictions = torch.concatenate([pred['samples'] for pred in predictions_dict], dim=1).transpose(0,1)
    log_probs = torch.concatenate([pred['log_prob'] for pred in predictions_dict], dim=0)
    
    if config.model_type=='GNN':
        truths = torch.cat([batch.y for batch in test_loader], dim=0)
    elif config.model_type=='FCN':
        truths = torch.cat([batch[1] for batch in test_loader], dim=0)
    
    truths = unstandardize_features(truths, config)
    predictions = unstandardize_features(predictions, config)
    return truths.squeeze(), predictions.squeeze(), log_probs.squeeze()

def get_lampe_pairs(config, dataloader, model) -> zip:
    '''
    Generate parameter and posterior pairs 
    '''
    thetas = [] 
    exes = []  
    
    if config.model_type=='GNN':
        for batch in dataloader:
            thetas.append(batch.y.cpu())
            exes.append(model.model(batch).cpu())
    elif config.model_type=='FCN':
        for batch in dataloader:
            thetas.append(batch[1].cpu())
            exes.append(batch[0].cpu())
    theta = torch.cat(thetas, dim=0)
    x = torch.cat(exes, dim=0)
    return zip(theta.to(device), x.to(device))

def get_coverage(config: DataConfig):
    dataloader = load_data(config, only_test=True)
    model = load_model(config)
    pairs = get_lampe_pairs(config, dataloader=dataloader, model=model)
    return expected_coverage_mc(model.flow, pairs, device=device)

def all_coverages(model_type: str, suite: str, data_dir: str, model_dir: str):
    # TODO make this more flexible for different dimensions of plotting 
    # (eg different models, different suites, different redshifts)
    redshifts = [3,4,5,6]
    
    levels = []
    covers = []
    
    config = DataConfig(
            data_dir=data_dir, 
            ckpt_dir=model_dir,
            model_type=model_type,
            sim=suite,
            z=3
        )
    
    for zed in redshifts:
        config.z = zed
        lvl, cvr = get_coverage(config)
        levels.append(lvl)
        covers.append(cvr)
        
    return levels, covers