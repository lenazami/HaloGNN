# ------------------
# Imports
# ------------------
import os
print(f"{os.path.basename(__file__)} running")

import time
import numpy as np
import pandas as pd
import pickle
from scipy.spatial import cKDTree
import torch
from torch_geometric.data import Data
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("The GNN data script used: ",device)

# ------------------
# Dictionaries and Global Variables
# ------------------
graph_radius = 2000.
max_n_galaxies = 100

'''
Stores redshifts, sim suites, and filepaths for data made during processing
'''
data_dir = Path('/n/holystore01/LABS/itc_lab/Lab/galaxyGNN')

filenames = {
    'TNG': {
        6: data_dir / 'high-z-jwst-TNG/TNG100_galaxy_halo_catalog_z6.npy',
        5: data_dir / 'high-z-jwst-TNG/TNG100_galaxy_halo_catalog_z5.npy',
        4: data_dir / 'high-z-jwst-TNG/TNG100_galaxy_halo_catalog_z4.npy',
    },
    'ASTRID': {
        6: data_dir / 'high-z-jwst/ASTRID_galaxy_halo_catalog_047.npy',
        5: data_dir / 'high-z-jwst/ASTRID_galaxy_halo_catalog_107.npy',
        4: data_dir / 'high-z-jwst/ASTRID_galaxy_halo_catalog_147.npy',
        3: data_dir / 'high-z-jwst/ASTRID_galaxy_halo_catalog_214.npy'
    }
}

'''
For output filepaths
'''
base_path = '/n/home03/hbrittain/data_outs/gnn/'

# ------------------
# Functions and Classes
# ------------------
class DataFrameTensor:  
    '''
    Purpose: we want to preserve the useful properties of dataframes for data processing, 
    while avoiding repeated conversions to tensors w/in the HaloGraphDataset class
    
    Since this is a fairly bespoke datatype, it is not intended to be useable 
    outside of the context of this project. Many included functions are restricted:
    Notably: get_label and get_halo. Also the private __get_item__ function, 
    as no type errors will be thrown, since this dataset only has two data types anyway 
    (but you can boolean mask, if necessary). 
    '''
    def __init__(self, dataframe, means=None, stds=None):
        self.column_names = list(dataframe.columns)
        self.index = list(dataframe.index)
        self.tensor = torch.tensor(dataframe.values, dtype=torch.float)
        self.means = means
        self.stds = stds
    
    def get_column(self, column_name) -> torch.tensor:
        """
        Retrieve a column by name
        """
        col_idx = self.column_names.index(column_name)
        return self.tensor[:, col_idx]
    
    def drop_columns(self, columns) -> 'DataFrameTensor':
        """
        Drops specified columns and returns a new DataFrameTensor
        """
        # retrieves indices for dropping, and the names of the remaining columns
        col_indices = [self.column_names.index(col) for col in columns]
        remaining_cols = [col for col in self.column_names if col not in columns]
        
        # drops the specified columns
        remaining_tensor = torch.index_select(self.tensor, 1, torch.tensor([i for i in range(self.tensor.size(1)) if i not in col_indices]))
        new_df = pd.DataFrame(remaining_tensor.numpy(), columns=remaining_cols)
        return DataFrameTensor(new_df)
    
    def get_item(self, idx) -> torch.tensor:
        """
        Retrieve a certain row by index
        """
        return self.tensor[idx]
    
    def head(self) -> pd.DataFrame:
        '''
        Display first 5 rows of DataFrameTensor
        '''
        return pd.DataFrame(self.tensor[:5].numpy(), columns=self.column_names)
    
    def get_labels(self) -> torch.tensor:
        return self.get_column('HaloMass')
    
    def get_halos(self) -> 'DataFrameTensor':
        return self.drop_columns(['HaloMass']).tensor
    
    def get_position(self, idx=None) -> np.ndarray:
        """
        Get galaxy positions for specified indices. Returns all positions if no index specified
        """
        if idx is None:
            idx = slice(None)
        
        # get each position column
        pos1 = self.get_column('GalaxyPos_1')[idx].numpy() * self.stds['GalaxyPos_1'] + self.means['GalaxyPos_1']
        pos2 = self.get_column('GalaxyPos_2')[idx].numpy() * self.stds['GalaxyPos_2'] + self.means['GalaxyPos_2']
        pos3 = self.get_column('GalaxyPos_3')[idx].numpy() * self.stds['GalaxyPos_3'] + self.means['GalaxyPos_3']
        
        # Stack the positions into a NumPy array
        return np.array([pos1, pos2, pos3]).T
        

    def apply_mask(self, mask) -> 'DataFrameTensor':
        """
        Apply a boolean mask to filter rows and return a new DataFrameTensor
        """
        filtered_df = pd.DataFrame(self.tensor[mask].numpy(), columns=self.column_names)
        return DataFrameTensor(filtered_df)
    
    def __getitem__(self, key):
        """
        Retrieve col/row by input type
        """
        if isinstance(key, str):
            # column name
            return self.get_column(key)
        elif isinstance(key, int):
            # row index
            return self.get_item(key)
        elif torch.is_tensor(key) or isinstance(key, (pd.Series, np.ndarray)):
            # for boolean conditions
            return self.apply_mask(key)
        elif isinstance(key, list):
            if all(isinstance(i, int) for i in key):
                # list of row indices
                return self.tensor[key]
            elif all(isinstance(i, str) for i in key):
                # list of column names
                col_indices = [self.column_names.index(col) for col in key]
                return self.tensor[:, col_indices] 

    def __len__(self) -> int:
        return self.tensor.size(0) 

    def __gt__(self, other):
        return self.tensor > other
    
    def __lt__(self, other):
        return self.tensor < other
    
    def __ge__(self, other):
        return self.tensor >= other
    
    def __le__(self, other):
        return self.tensor <= other
    
    def __eq__(self, other):
        return self.tensor == other

    def __repr__(self):
        return f"DataFrameTensor(tensor={self.tensor}, column_names={self.column_names})"  

class HaloGraphDataset(Data):
    def __init__(self, data, boxsize, neighbour_idx, center_idx):
        super(HaloGraphDataset, self).__init__()
        self.gal_data = data.get_halos() # this is a tensor
        self.gal_labels = data.get_labels() # also a tensor
        self.neighbour_idx = neighbour_idx
        self.center_idx = center_idx
        self.positions = data.get_position() # numpy array w positions
        self.boxsize = boxsize
        
        self.data = [self.gal_data[nbhood] for nbhood in neighbour_idx]
        self.labels = self.gal_labels[center_idx]
        
        self.num_graphs = len(self.data) # length of data tensor (row-wise)
        
        
    def __len__(self):
        return self.num_graphs
    
    def apply_pbcs(self, delta):
        mask = np.abs(delta) > 0.5 * self.boxsize
        delta[mask] = np.where(delta[mask] > 0,
                             delta[mask] - self.boxsize,
                             delta[mask] + self.boxsize)
        return delta

    def get_periodic_distances_3d(self, pos_array):
        N = len(pos_array)
        idx_i, idx_j = np.meshgrid(np.arange(N), np.arange(N))
        idx_i, idx_j = idx_i.flatten(), idx_j.flatten()
        
        pos_i = pos_array[idx_i]  # (N*N, 3)
        pos_j = pos_array[idx_j]  # (N*N, 3)
        
        delta = pos_i - pos_j  # (N*N, 3)
        delta = self.apply_pbcs(delta)
        normalized_delta = delta / (2 * graph_radius)
        return torch.tensor(normalized_delta, dtype=torch.float32)
    
    def __getitem__(self, idx):
        x = self.data[idx]  # node features
        # Add relative positions respect to the fof group
        delta_pos = self.positions[self.neighbour_idx[idx]] - self.positions[self.center_idx[idx]]
        delta_pos = self.apply_pbcs(delta_pos) / graph_radius
        x = torch.cat([x, torch.tensor(delta_pos, dtype=torch.float32)], dim=1)
        n_nodes = len(x)
        
        edge_index = torch.tensor(np.array(np.meshgrid(
            np.arange(n_nodes), 
            np.arange(n_nodes)
        )).reshape(2, -1))
        
        neighborhood_positions = self.positions[self.neighbour_idx[idx]]
        edge_attr = self.get_periodic_distances_3d(neighborhood_positions)
        y = self.labels[idx]  # graph label
        n_galaxies = len(x)
        
        # stores the central galaxy 
        central_pos = np.where(self.neighbour_idx[idx] == self.center_idx[idx])[0]
        central_mask = torch.zeros(n_nodes, dtype=torch.bool)
        central_mask[central_pos] = True

        return Data(
            x=x, 
            edge_index=edge_index, 
            edge_attr=edge_attr, 
            y=y, 
            global_attr=torch.tensor([n_galaxies/max_n_galaxies], dtype=torch.float32),
            central_mask=central_mask,
        )
    
def get_time(elapsed):
    '''
    Gets time elapsed in text format
    '''
    hours, remainder = divmod(elapsed, 3600)  # 3600 seconds in an hour
    minutes, seconds = divmod(remainder, 60)       # 60 seconds in a minute
    return f'{int(hours)} hours, {int(minutes)} minutes, {seconds:.2f} seconds'

def process(sim, z, filepath, boxsize):
    start = time.time()
    
    # Housewarming
    print(f"\033[35mCurrently processing redshift z={z} for {sim} Suite\033[37m")
    cat_data = np.load(filepath)
    cat_data = cat_data[cat_data['GalaxyMass']>1e8]
    # For now remove galaxies with crazy Rhalf
    cat_data = cat_data[cat_data['GalaxyRhalf'] < 6000.]
    print(f"This catalogue contains {cat_data.shape} galaxies")
        
    out_path = f"{base_path}/{sim}_z{z}"
    os.makedirs(out_path, exist_ok=True)

    # ------------------
    # Centers halo at largest galaxy 
    center = pd.DataFrame({
        'GalaxyMass': cat_data['GalaxyMass'],
        'FOFID': cat_data['FOFID'],
    })
    central_idx = center.groupby('FOFID')['GalaxyMass'].idxmax().values

    # -------------
    # Standardize data
    data_dict = {
        'HaloMass': np.log10(cat_data['HaloMass']),
        'GalaxyMass': np.log10(cat_data['GalaxyMass']),
        'GalaxyPos_1': cat_data['GalaxyPos'][:, 0] % boxsize,
        'GalaxyPos_2': cat_data['GalaxyPos'][:, 1] % boxsize,
        'GalaxyPos_3': cat_data['GalaxyPos'][:, 2] % boxsize,
        'GalaxyVel_1': cat_data['GalaxyVel'][:, 0],
        'GalaxyVel_2': cat_data['GalaxyVel'][:, 1],
        'GalaxyVel_3': cat_data['GalaxyVel'][:, 2],
        'GalaxyRhalf': np.log10(cat_data['GalaxyRhalf']),
        'SFR': np.log10(np.where(cat_data['SFR'] == 0, 1e-5, cat_data['SFR'])),
        'jwst_f090w': cat_data['jwst_f090w'],
        'jwst_f150w': cat_data['jwst_f150w'],
        'jwst_f277w': cat_data['jwst_f277w'],
        'jwst_f444w': cat_data['jwst_f444w'],
    }
    
    raw_df = pd.DataFrame(data_dict)

    means = raw_df.mean()
    stds = raw_df.std()
    stats_df = pd.concat([means, stds], keys=['means', 'stds'], axis=1)
    stats_df.to_csv(f"{out_path}/{sim}_z{z}_stats.csv", index=False)

    # Builds tree based on radius and then queries it at the positions of the central galaxies
    galpos = cat_data['GalaxyPos'] % boxsize
    print('BUILDING GRAPH')
    print('gal pos = ', galpos.min(), galpos.max())
    print('boxsize = ', boxsize)
    kdtree = cKDTree(galpos, boxsize=boxsize)
    neighbour_idx = kdtree.query_ball_point(galpos[central_idx], graph_radius)

    fofs = cat_data['FOFID']
    
    # Saving things to graph the halos later
    np.savez(f'{out_path}/{sim}_z{z}_displaygraphdata.npz', fofs=fofs, galpos=galpos, neighbour_idx=neighbour_idx, central_idx=central_idx)



    # -------------------
    # Creates a DataFrameTensor from standardized cat_data, then converts to a HaloGraphDataset, 
    # then saves to a torch file to be accessed later
    standard_columns = [
        'HaloMass', 
        'GalaxyMass', 
        'GalaxyPos_1', 'GalaxyPos_2', 'GalaxyPos_3', 
        'GalaxyVel_1', 'GalaxyVel_2', 'GalaxyVel_3', 
        'GalaxyRhalf', 
        'SFR',
        'jwst_f090w', 'jwst_f150w', 'jwst_f277w', 'jwst_f444w',
    ]
    # v this part is weird
    raw_df[standard_columns] = (raw_df[standard_columns] - means[standard_columns]) / stds[standard_columns]
    standard_df = raw_df
    # ^ to here
    standard_df.to_csv(f"{out_path}/{sim}_z{z}_standardized.csv", index=False)
    df_tensor = DataFrameTensor(standard_df, means=means, stds=stds)

    test_set = HaloGraphDataset(df_tensor, boxsize, neighbour_idx, central_idx)

    all_graphs = []
    for idx in range(len(test_set)):
        graph_data = test_set[idx]
        all_graphs.append(graph_data)
        
    torch.save(all_graphs, f'{out_path}/{sim}_z{z}_all_graphs.pt')
    
    
    end = time.time()
    print(f"{sim}z{z} halo processing took {get_time(end-start)}. Processed {len(test_set)/(end-start)} halos per second")

if __name__ == '__main__':
    # ------------------
    # Main Processing
    # ------------------

    start_whole = time.time()
    for sim, data in filenames.items():
        if sim=='ASTRID':
            boxsize=250_000
        elif sim=='TNG':
            boxsize = 75_000
        for z, filepath in data.items():
            process(sim, z, filepath, boxsize)

    # ------------------
    # Finishing
    # ------------------
    end_whole = time.time()
    print(f"This program took {get_time(end_whole-start_whole)}")