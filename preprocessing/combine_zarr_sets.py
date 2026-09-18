# -*- coding: utf-8 -*-
"""
Created on 19.02.2026
@author: eschlager
To combine data from different forcings to train combined models (ERA-Interim & CESM2, ERA-Interim & CESM2 future)
"""

import xarray as xr
import numpy as np
import os
import sys
import pandas as pd
from dask.diagnostics import ProgressBar
from prepare_trainset import ZarrDataset
script_dir = os.path.abspath(os.path.dirname(__file__))
project_dir = os.path.sep.join([script_dir, '..'])
sys.path.append(os.path.sep.join([project_dir , 'src']))
import logging_config
import read_yaml
import logging

# Open both datasets using xarray (which works with Zarr)
script_dir = os.path.abspath(os.path.dirname(__file__))
project_dir = os.path.sep.join([script_dir, '..'])
data_dir = os.path.sep.join([project_dir, 'data', 'processed', 'crossval', 'v_01'])


# datasets = ['ERAI', 'CESM2']
datasets = ['ERAI', 'CESM2future']

out_dir = os.path.join(data_dir, f"{'_'.join(datasets)}_meltNN")
os.makedirs(out_dir, exist_ok=True)


# create respective data sets with 2500 locs subsampling
for dataset in datasets:
    specs = read_yaml.read_yaml_file(os.path.sep.join([script_dir, 'spec_files_transfer', f'specs_melt_modularNNEBM_{dataset}_2500.yml']))
    zarrset = ZarrDataset(specs)
    file_dir = zarrset.get_file_dir()
    if 'future' in dataset:
        dates = pd.date_range(start='2075-01-01', end='2100-12-31', freq='D')
        base_dir = os.path.abspath(os.path.sep.join([project_dir, 'data', 'processed', f'HIRHAM5-{dataset.split("future")[0]}', 'future']))
    else:
        dates = pd.date_range(start='1990-01-01', end='2016-12-31', freq='D')
        base_dir = os.path.abspath(os.path.sep.join([project_dir, 'data', 'processed', f'HIRHAM5-{dataset}', 'v_04']))
    daily_file = os.path.abspath(os.path.sep.join([base_dir, 'base_dataset.zarr']))
    spinup_file = os.path.abspath(os.path.sep.join([base_dir, 'base_dataset_10yr.zarr']))

    file_name_sub = f'data_sub_2500.zarr'

    try:
        zarrset.check_file(file_name_sub)
        logging.info('   succesful!')
    except (FileNotFoundError, KeyError) as e:
        zarrset.prepare_prediction_file(dates, daily_file, spinup_file=spinup_file, use_spatial_subsampling=True, savedir=os.path.sep.join([file_dir, file_name_sub]))
    except:
        logging.error(f"There is a problem with the file {file_name_sub} - please check!")
        os._exit(0)


# combine the two data sets; use directories as defined in the spec files
ds1 = xr.open_zarr(os.path.sep.join([project_dir, 'data', 'processed', 'crossval', 'v_01', f'{datasets[0]}_meltNN_2500', 'data_sub_2500.zarr']))
ds2 = xr.open_zarr(os.path.sep.join([project_dir, 'data', 'processed', 'crossval', 'v_01', f'{datasets[1]}_meltNN_2500', 'data_sub_2500.zarr']))

Nz = ds1.sizes["z"]
assert Nz == ds2.sizes["z"]
assert (ds1.z.values == ds2.z.values).all()

# for datasets with different spatial subsamplig, there are different x and y values associated with z!
ds1 = ds1.drop_vars(["x", "y", "lon", "lat"], errors="ignore")
ds2 = ds2.drop_vars(["x", "y", "lon", "lat"], errors="ignore")

combined = xr.concat([ds1, ds2], dim='time', coords='minimal')

combined = combined.chunk({"time": 1, "z": -1, "variable_names": -1})

with ProgressBar():
    combined.to_zarr(os.path.join(out_dir, f'data_sub.zarr'))