# -*- coding: utf-8 -*-
"""
Created on 15.11.2025
@author: eschlager
Main script to prepare data and apply trained model
"""

import logging
import os
import xarray as xr
import sys
from dask.diagnostics import ProgressBar
import pandas as pd
from torch.utils.data import DataLoader
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


# import local modules
script_dir = os.path.abspath(os.path.dirname(__file__))
project_dir = os.path.sep.join([script_dir, '..'])
sys.path.append(os.path.sep.join([project_dir , 'src']))
import logging_config
from create_dataset import FirnpackCellsDataset, my_collate_fn
import predictor


# Create predictions for 1990-2016

# define which model to use
out_dir = os.path.sep.join(['dmidata', 'projects', 'smb_and_polar_rcm', 'elkesc', 'output', 'Crossval', 'crossval_ERAI_modularNNEBM'])
data_dir = os.path.sep.join(['dmidata', 'users', 'elkesc', 'MeltEmulation', 'data', 'processed', 'crossval', 'v_01', 'ERAI_meltNN', 'data.zarr'])

RECONSTRUCT_COORDS = False


# only few models and 2 years for testing purpose
years = [1992, 1994]
iters = range(2)
dates = (pd.date_range(start="1990-01-01", end="1991-12-31")+pd.Timedelta(hours=12)).tolist()

for y in years:
    for it in iters:
        fold = f'fold_{y}'
        
        model_dir = os.path.sep.join([out_dir, fold, f'iter_{it}'])
        specs = read_yaml.read_yaml_file(os.path.sep.join([model_dir, 'specs.yml']))

        zarrset = ZarrDataset(specs)
        var_dict = zarrset.get_variable_dict()

        scalerpath = os.path.sep.join([model_dir, '..', 'std_scaler.npz'])
        val_data = FirnpackCellsDataset(data_dir, dates=dates, variable_names=var_dict, scaling=scalerpath)

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        logging.info(f'Using device {device}')   
        # dataloader = DataLoader(val_data, batch_size=32, shuffle=False, collate_fn=my_collate_fn, num_workers=4, persistent_workers=True,
        #                             pin_memory=pin_memory, worker_init_fn=AffinityInitializer(base_offset=12, cores_per_worker=1))
        dataloader = DataLoader(val_data, batch_size=32, shuffle=False, collate_fn=my_collate_fn, num_workers=0)

        model_predictor = predictor.ModelPredictor(model_dir, var_dict, dataloader, filedir=os.path.sep.join([model_dir, 'pred_data.zarr']), load='best', device=device)
        x_coords = xr.open_zarr(val_data.file_dir)['x'].values
        y_coords = xr.open_zarr(val_data.file_dir)['y'].values
        
        xy_coords_ref = None
        lonlat_coords_ref = None
        if RECONSTRUCT_COORDS:
            try:
                coords_file = os.path.sep.join([data_dir, '..', '..', 'coords_only.zarr'])
                coords_ref = xr.open_dataset(coords_file)
                xy_coords_ref = (coords_ref['x'], coords_ref['y'])
                logging.info(f"Use reference coordinates x [{len(xy_coords_ref[0])}], y [{len(xy_coords_ref[1])}].")
                if 'lon' in coords_ref and 'lat' in coords_ref:
                    lonlat_coords_ref = (coords_ref['lon'], coords_ref['lat'])
                    logging.info("Use reference coordinates lon/lat.")
                else:
                    lonlat_coords_ref = None
                coords_ref.close()
            except Exception as e:
                logging.error(f"Could not read coordinates from reference file {coords_file}: {e}")
                raise ValueError(f"Could not read coordinates from reference file {coords_file}: {e}")


        model_predictor.make_predictions_file(coords=(x_coords, y_coords), 
                                                ref_coords=xy_coords_ref, 
                                                extra_coords=lonlat_coords_ref)