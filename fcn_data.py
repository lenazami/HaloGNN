# ------------------
# Imports
# ------------------
import os
import time
start_whole = time.time()

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
filenames = {
    'TNG': {
        6: '/n/home03/hbrittain/galaxyGNN/high-z-jwst-TNG/TNG100_galaxy_halo_catalog_z6.npy',
        5: '/n/home03/hbrittain/galaxyGNN/high-z-jwst-TNG/TNG100_galaxy_halo_catalog_z5.npy',
        4: '/n/home03/hbrittain/galaxyGNN/high-z-jwst-TNG/TNG100_galaxy_halo_catalog_z4.npy',
    },
    'ASTRID': {
        6: '/n/holystore01/LABS/itc_lab/Lab/galaxyGNN/high-z-jwst/ASTRID_galaxy_halo_catalog_047.npy',
        5: '/n/holystore01/LABS/itc_lab/Lab/galaxyGNN/high-z-jwst/ASTRID_galaxy_halo_catalog_107.npy',
        4: '/n/holystore01/LABS/itc_lab/Lab/galaxyGNN/high-z-jwst/ASTRID_galaxy_halo_catalog_147.npy',
        3: '/n/holystore01/LABS/itc_lab/Lab/galaxyGNN/high-z-jwst/ASTRID_galaxy_halo_catalog_214.npy',
    } 
}

'''
For output filepaths
'''
base_path = '/n/home03/hbrittain/outputs/baseline'

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
    Goes through each halo array and creates a 1d-vector of summary statistics.
    In case any 0's exist in the data, the summary statistics are first formed, 
    and then base log 10 is applied. This avoids any -inf that will mess up the data.
    '''
    halo['SFR'] = np.where(halo['SFR'] == 0, 1e-4, halo['SFR'])
    
    halomass = np.log10(halo['HaloMass'].max())
    galmass_sum, galmass_max, galmass_mean = np.log10([halo['GalaxyMass'].sum(), 
                                                       halo['GalaxyMass'].max(), halo['GalaxyMass'].mean()])
    sfr_sum, sfr_max, sfr_mean = np.log10([halo['SFR'].sum(), halo['SFR'].max(), halo['SFR'].mean()])
    vel_disp = halo['GalaxyVel'].std()
    f0 = halo['jwst_f090w'].mean()
    f1 = halo['jwst_f150w'].mean()
    f2 = halo['jwst_f277w'].mean()
    f4 = halo['jwst_f444w'].mean()
    return [halomass, galmass_sum, galmass_max, galmass_mean, sfr_sum, 
            sfr_max, sfr_mean, vel_disp, f0, f1, f2, f4] 

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
        
        out_path = f"{base_path}/{sim}_z{z}"
        os.makedirs(out_path, exist_ok=True)
        print(f"Currently processing redshift z={z} for {sim} Suite")
        
        cat_data = np.load(filepath)
        print(f"This catalogue is {os.path.getsize(filepath) / (1024 * 1024):.2f} MB, contains {cat_data.shape[0]:_} galaxies")
        
        # ---------
        # No need to batch; Max file size is <0.5GB
        # On the off chance it is not sorted, this shouldn't take more than 5 seconds
        cat_data = np.sort(cat_data, order='FOFID') 
        halos = make_halo_array(cat_data)
                
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
            halo_stats.append(sum_stats(halos))
        
        # Reformatting halo_stats to dataframe for file purposes; We want to retain column titles
        df = pd.DataFrame(halo_stats, columns=['HaloMass', 'GM_Sum', 'GM_Max', 'GM_Mean', 'SFR_Sum', 'SFR_Max', 
                                               'SFR_Mean', 'V_disp', 'f0', 'f1', 'f2', 'f4'])
        df.to_csv(f"{out_path}/{sim}_z{z}_raw.csv", index=False)

        means = df.mean()
        stds = df.std()
        stats = np.array([means],[stds])
        # stats_df = pd.concat([means, stds], keys=['means', 'stds'], axis=1)
        # stats_df.to_csv(f"{out_path}/{sim}_z{z}_stats.csv", index=False)

        # Standardize dataframe by subtracting mean and dividing by std
        scaled_df = (df - means) / stds
        scaled_df.to_csv(f"{out_path}/{sim}_z{z}_normalized.csv", index=False)
        
        
end_whole = time.time()
print(f"This program took {get_time(end_whole-start_whole)}")
