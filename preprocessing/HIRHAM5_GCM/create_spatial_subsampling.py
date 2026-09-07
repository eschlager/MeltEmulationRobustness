# -*- coding: utf-8 -*-
"""
@author: eschlager
Create classification of GrIS in different zones:
1. ablation (total SMB of the defined years < 0)
2. percolation (accumulation zone with non-negliglible melt, i.e. at least in one year there was melt >= 100 mm weq)
3. dry snow (accumulation zone with yearly melt < 100 mm weq)

make spatial subsampling mask with N locations, with higher sampling probability in ablation zone.
"""

#%% make imports

import os 

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import xarray as xr

base_dir = os.path.dirname(os.path.abspath('')).split(os.sep + 'preprocessing')[0]
data_dir = os.path.sep.join([base_dir, 'data', 'interim', 'CESM2', 'HIRHAM5', 'firnpack']) 
aux_dir = os.path.sep.join([base_dir, 'data', 'interim', 'CESM2', 'HIRHAM5', 'AuxFiles'])

#%% Create zones file based on SMB of following years:
years = range(1990, 1999)    # for historic data set
# years = range(2075, 2084)    # for future data set

# Load GrIS mask
ds_mask = xr.open_dataset(os.path.sep.join([aux_dir, 'GRLmask.nc']))
gris_mask = ds_mask[['glacGRL']]
mask_bool = ~np.isnan(gris_mask['glacGRL'])

# define snowmelt threshold
snmel_threshold = 100.

# Containers for annual masks
dry_snow_mask_list = []
smb_mask_list = []

for year in years:
    ds = xr.open_dataset(os.path.sep.join([data_dir, f'Daily2D_GRL_{year}.nc']))

    # Mean SMB for this year
    smb = ds['gld'].mean(dim='time')

    # Total snowmelt for this year
    yearly_snmelt = ds['snmel'].sum(dim='time')
    negligible_melt = (yearly_snmelt < snmel_threshold) & mask_bool

    # Track masks for combining later
    dry_snow_mask_list.append(negligible_melt)
    smb_mask_list.append(smb)

# Stack yearly masks into new dimensions
dry_snow_stack = xr.concat(dry_snow_mask_list, dim='year')
smb_mean = xr.concat(smb_mask_list, dim='year').mean(dim='year')

# Create final masks

# Accumulation: SMB >= 0 in majority of years
final_accumulation = (smb_mean>=0) & mask_bool

# Dry snow: consistently negligible melt across all years
dry_snow = dry_snow_stack.all(dim='year') 
final_dry_snow = dry_snow & final_accumulation

# Ablation: SMB < 0 in majority of years
final_ablation = (smb_mean<0) & mask_bool

# Percolation = not dry_snow AND accumulation
final_percolation = (~dry_snow) & final_accumulation

# Final zone map
zones = gris_mask['glacGRL']*np.nan 
zones = zones.where(~mask_bool, 0)  # leave outside of GrIS as NaN
zones = zones.where(~final_dry_snow, 3)        # dry snow = 3
zones = zones.where(~final_percolation, 2)     # percolation = 2
zones = zones.where(~final_ablation, 1)        # ablation = 1

# Combine into a final mask dataset
final_mask = xr.Dataset({
    'zones': zones,
    'dry_snow': xr.where(final_dry_snow, 1.0, np.nan),
    'accumulation': xr.where(final_accumulation, 1.0, np.nan),
    'ablation': xr.where(final_ablation, 1.0, np.nan),
    'percolation': xr.where(final_percolation, 1.0, np.nan)
})

final_mask['zones'].attrs['description'] = "1: ablation, 2: percolation, 3: dry snow"

# plot all zones
plt.figure(figsize=(10,10))
plt.pcolormesh(final_mask['zones'])
plt.gca().set_aspect(1)
plt.axis('off')
plt.savefig(os.path.sep.join([aux_dir, f'GRLzones.png']), bbox_inches='tight')
plt.show()

acc_locs = int(final_mask['accumulation'].sum().item())
abl_locs = int(final_mask['ablation'].sum().item())
percolation_acc_locs = int(final_mask['percolation'].sum().item())
dry_snow_locs = int(final_mask['dry_snow'].sum().item())
print(f"Number of accumulation locations: {acc_locs}")
print(f"Number of ablation locations: {abl_locs}")
print(f"Number of locations with percolation within accumulation: {percolation_acc_locs}")
print(f"Number of dry snow locations: {dry_snow_locs}")

assert acc_locs + abl_locs == gris_mask['glacGRL'].sum(), "accumulation and ablation locations do not sum up to total GrIS locations"
assert percolation_acc_locs + dry_snow_locs == acc_locs, "Percolation and dry snow accumulation locations do not sum up to all accumulation locations"

final_mask.to_netcdf(os.path.sep.join([aux_dir, f'GRLzones.nc']))



#%% Make spatial sub-sampling based on zones

ds_zones = xr.open_dataset(os.path.sep.join([aux_dir, f'GRLzones.nc']))
zone_cats = ds_zones['zones'].values.astype(float)

## If want to make weights based on number of pixels in each zone
# unique, counts = np.unique(zone_cats, return_counts=True)
# weights = 1 / counts
# weights = np.round(weights / weights.sum(), 2)
# weight_map = {float(k): float(w) for k, w in zip(unique, weights)}

# define sampling weights for each zone manually:
weight_map = {1.0: 0.65, 2.0: 0.3, 3.0: 0.05}

def safe_lookup(z):
    if np.isnan(z):
        return np.nan
    else:
        return weight_map[float(z)]


vectorized_lookup = np.vectorize(safe_lookup)

zone_weight_array = vectorized_lookup(zone_cats)

zone_weight_array = zone_weight_array / zone_weight_array[~np.isnan(zone_weight_array)].sum()

# sample N locations
N = 2500

# Flatten and normalize
prob_flat = zone_weight_array.flatten()

sampled_indices = np.random.choice(len(prob_flat), size=N, p=np.nan_to_num(prob_flat), replace=False)

flat_mask = np.zeros(len(prob_flat), dtype=bool)
flat_mask[sampled_indices] = True

# Reshape back to 2D mask
mask_2d = flat_mask.reshape((602, 402))
mask_2d

plt.figure(figsize=(4,6))
float_mask = mask_2d.astype(float)
float_mask[~mask_2d] = np.nan
plt.pcolormesh(~np.isnan(ds_zones['zones'].values), cmap=ListedColormap(['white', 'silver']))
plt.pcolormesh(float_mask, cmap=ListedColormap(['red']), vmin=1)
plt.savefig(os.path.sep.join([aux_dir, f'GRL_subsample_{N}.png']), bbox_inches='tight')
plt.show()


ds_sub = xr.Dataset(coords={'y':ds_zones.y.values, 'x':ds_zones.x.values})
ds_sub['subsampling'] = (('y','x'), mask_2d)
ds_sub.to_netcdf(os.path.sep.join([aux_dir, f'GRL_subsampleidx_{N}.nc']))
