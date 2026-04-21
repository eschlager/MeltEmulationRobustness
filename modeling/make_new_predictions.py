# -*- coding: utf-8 -*-
"""
Created on 15.11.2025
@author: eschlager
Main script to prepare data and apply trained model
"""

import os
import xarray as xr
import sys
from dask.diagnostics import ProgressBar
import pandas as pd
import torch
import matplotlib.pyplot as plt
script_dir = os.path.abspath(os.path.dirname(__file__))
project_dir = os.path.sep.join([script_dir, '..'])
sys.path.append(os.path.sep.join([project_dir , 'src']))
sys.path.append(os.path.sep.join([project_dir , 'modeling', 'models']))
sys.path.append(os.path.sep.join([project_dir, 'modeling']))
import read_yaml
from prepare_trainset import ZarrDataset
import eval_meltNN

# Create predictions for 1990-2016

# define which model to use
model_rel_dir = os.path.sep.join(['output', 'optuna_melt_modularNN_EBM/trial_13'])
model_dir = os.path.sep.join([project_dir, model_rel_dir])
specs = read_yaml.read_yaml_file(os.path.sep.join([model_dir, 'specs.yml']))
model_data_dir = os.path.abspath(os.path.sep.join([project_dir, specs['directories']['base_dir']]))
training_data_dir = os.path.sep.join([model_data_dir, specs['directories']['data_file']])
scaler_dir = os.path.sep.join([training_data_dir, 'std_scaler.npz'])

# prepare data for predictions with trained model: getting previous days, transforming, scaling ....
YEARS = list(range(1990,2016+1))

FILENAME = f'data_{YEARS[0]}-{YEARS[-1]}'

start_date = pd.to_datetime(f'{YEARS[0]}-01-01T12:00:00.000000000')
end_date = pd.to_datetime(f'{YEARS[-1]}-12-31T12:00:00.000000000')
dates = pd.date_range(start=start_date, end=end_date, freq='D')

# define base data files
daily_file = os.path.sep.join([model_data_dir, f'base_dataset.zarr'])
spinup_dir = os.path.sep.join([model_data_dir, f'base_dataset_10yr.zarr'])

zarrset = ZarrDataset(specs)
zarrset.prepare_prediction_file(dates, daily_file, spinup_file=spinup_dir, scaler_file=scaler_dir, savedir=os.path.sep.join([training_data_dir, f'{FILENAME}.zarr']))

# run eval_NNmelt to make predictions and evaluation plots
rmse, mae, mbe, r2 = eval_meltNN.main(model_rel_dir, mode=FILENAME, reconstruct_coords=True)