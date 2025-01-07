# ------------------
# Imports
# ------------------
import os
import time
start_whole = time.time()

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("The baseline data script used:",device)

# ------------------
# Dictionaries and Global Variables
# ------------------
'''
Stores redshifts, sim suites, and filepaths for data made during processing
Sorted by file size/galaxy count
'''
DATA_DIR = Path('/n/holystore01/LABS/itc_lab/Lab/galaxyGNN/')
base_path = Path('/n/holystore01/LABS/itc_lab/Lab/galaxyGNN/carol_processed_data/baseline')

filenames = {
    'TNG': {
        6: DATA_DIR / 'high-z-jwst-TNG/TNG100_galaxy_halo_catalog_z6.npy',
        5: DATA_DIR / 'high-z-jwst-TNG/TNG100_galaxy_halo_catalog_z5.npy',
        4: DATA_DIR / 'high-z-jwst-TNG/TNG100_galaxy_halo_catalog_z4.npy',
    },
    'ASTRID': {
        6: DATA_DIR / 'high-z-jwst/ASTRID_galaxy_halo_catalog_047.npy',
        5: DATA_DIR / 'high-z-jwst/ASTRID_galaxy_halo_catalog_107.npy',
        4: DATA_DIR / 'high-z-jwst/ASTRID_galaxy_halo_catalog_147.npy',
        3: DATA_DIR / 'high-z-jwst/ASTRID_galaxy_halo_catalog_214.npy',
    } 
}

'''
For output filepaths
'''

# ------------------
# Functions
# ------------------
def make_halo_array(dat):
    '''
    Takes in a data array and splits array when FOFID changes 
    (catalogues are sorted by FOFID)
    '''
    boundaries = np.flatnonzero(np.diff(dat['FOFID'])) + 1
    return np.split(dat, boundaries)

def sum_stats(halo):
    '''
    Computes summary statistics for a group of galaxies within a halo.
    Returns a dictionary of features, making it easy to add/remove features
    while maintaining consistency in the data pipeline.
    
    Parameters:
        halo: Structured numpy array containing galaxy properties within a halo
        
    Returns:
        dict: Dictionary of feature names and their values
    '''
    MIN_SFR = 1e-4
    halo['SFR'] = np.where(halo['SFR'] == 0., MIN_SFR, halo['SFR'])
    
    # Mass features
    features = {
        'HaloMass': np.log10(halo['HaloMass'].max()),
        'GalaxyMass_Sum': np.log10(halo['GalaxyMass'].sum()),
        'GalaxyMass_Max': np.log10(halo['GalaxyMass'].max()),
        'GalaxyMass_Mean': np.log10(halo['GalaxyMass'].mean()),
    }
    
    # Star formation features
    features.update({
        'SFR_Sum': np.log10(halo['SFR'].sum()),
        'SFR_Max': np.log10(halo['SFR'].max()),
        'SFR_Mean': np.log10(halo['SFR'].mean()),
    })
    
    # Velocity features
    features.update({
        'Velocity_Dispersion': halo['GalaxyVel'].std(),
        'Velocity_Max': halo['GalaxyVel'].max(),
        'Velocity_Mean': halo['GalaxyVel'].mean(),
    })
    
    # JWST photometry features
    for band, band_name in [('jwst_f090w', 'F090W'), 
                           ('jwst_f150w', 'F150W'),
                           ('jwst_f277w', 'F277W'),
                           ('jwst_f444w', 'F444W')]:
        band_data = halo[band]
        features.update({
            f'{band_name}_Sum': band_data.sum(),
            f'{band_name}_Max': band_data.max(),
            f'{band_name}_Mean': band_data.mean(),
        })
    
    # Structural features
    features.update({
        'N_Galaxies': halo['GalaxyMass'].shape[0],
    })
    return features

def get_time(elapsed):
    '''
    Gets time elapsed in readable text format
    '''
    hours, remainder = divmod(elapsed, 3600)  # 3600 seconds in an hour
    minutes, seconds = divmod(remainder, 60)  # 60 seconds in a minute
    return f'{int(hours)} hours, {int(minutes)} minutes, {seconds:.2f} seconds'

# ------------------
# Main Processing
# ------------------
for sim, data in filenames.items():
    for z, filepath in data.items():
        start = time.time()
        
        out_path = base_path / f"{sim}_z{z}"
        out_path.mkdir(parents=True, exist_ok=True)
        print(f"Currently processing redshift z={z} for {sim} Suite")
        
        cat_data = np.load(filepath)
        print(f"This catalogue is {os.path.getsize(filepath) / (1024 * 1024):.2f} MB, contains {cat_data.shape[0]:_} galaxies")
        
        # ---------
        # No need to batch; Max file size is <0.5GB
        # On the off chance it is not sorted, this shouldn't take more than 5 seconds
        cat_data = np.sort(cat_data, order='FOFID') 
        print('All galaxies = ', cat_data.shape)
        halos = make_halo_array(cat_data)
        print('All Halos = ', len(halos))
        print('Mean len halos = ', np.mean([len(halo) for halo in halos]))
        print('Max len halos = ', np.max([len(halo) for halo in halos]))
                
        # Ensures that if data is generated again, it will overwrite the previous file
        with open(f'{out_path}/{sim}_z{z}_halos.pkl', 'wb') as file:
            pickle.dump(halos, file)

        end = time.time()
        print(f"Halo processing took {get_time(end-start)}")

        # --------- 
        # Creates the summary statistics; Processes ~50,000 halos/sec
        column_names = halos[0].dtype.names
        
        halo_stats = []
        for halo in halos:
            halo_stats.append(sum_stats(halo))
        df = pd.DataFrame(halo_stats)
        df.to_csv(f"{out_path}/{sim}_z{z}_raw.csv", index=False)
        print('len df = ', len(df))

        means = df.mean()
        stds = df.std()
        stats = np.stack((means.values,stds.values))
        # stats_df = pd.concat([means, stds], keys=['means', 'stds'], axis=1)
        # stats_df.to_csv(f"{out_path}/{sim}_z{z}_stats.csv", index=False)

        # Standardize dataframe by subtracting mean and dividing by std
        scaled_df = (df - means) / stds
        scaled_df.to_csv(f"{out_path}/{sim}_z{z}_normalized.csv", index=False)
        
        
end_whole = time.time()
print(f"This program took {get_time(end_whole-start_whole)}")
