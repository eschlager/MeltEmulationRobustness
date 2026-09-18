# -*- coding: utf-8 -*-
"""
Created on 30.07.2026
@author: eschlager
Script to create predictions from perturbed inputs for sensitivity study
"""

import os
import sys

import pandas as pd
import xarray as xr
from torch.utils.data import DataLoader
import logging

script_dir = os.path.abspath(os.path.abspath(''))
project_dir = os.path.sep.join([script_dir, '..'])
sys.path.append(os.path.sep.join([project_dir , 'src']))
sys.path.append(os.path.sep.join([project_dir , 'modeling', 'models']))
sys.path.append(os.path.sep.join([project_dir, 'modeling']))
import read_yaml
from prepare_trainset import ZarrDataset
from create_dataset import FirnpackCellsDataset, my_collate_fn
import predictor
import logging_config
from my_utils import shutdown_dataloader


def make_predictions(model_specs, data_spec):
    for model_name, iter in model_specs.items():

        model_dir = os.path.sep.join([base_dir, 'output', f'repeated_{data_spec}_{model_name}_log', f'iter_{iter}'])
        logging_config.define_root_logger(os.path.join(model_dir, '..', f'log_perturbation_study.txt'))
        logging.getLogger().setLevel(logging.INFO)
        scalerpath = os.path.sep.join([model_dir, '..', 'std_scaler.npz'])
        specs = read_yaml.read_yaml_file(os.path.sep.join([model_dir, 'specs.yml']))

        zarrset = ZarrDataset(specs)
        var_dict = zarrset.get_variable_dict()

        if 'future' in data_spec:
            dates = (pd.date_range(start="2075-01-01", end="2100-12-31")+pd.Timedelta(hours=12)).tolist()
        else:
            dates = (pd.date_range(start="1990-01-01", end="2016-12-31")+pd.Timedelta(hours=12)).tolist()

        data_dir = os.path.sep.join([base_dir, 'data', 'processed', 'crossval', 'v_01', f'{data_spec}_meltNN', 'data.zarr'])
        val_data = FirnpackCellsDataset(data_dir, dates=dates, variable_names=var_dict, scaling=scalerpath)

        dataloader = DataLoader(val_data, batch_size=100, shuffle=False, collate_fn=my_collate_fn)

        save_dir = os.path.sep.join([model_dir, '..', f'perturbation_study_{iter}'])
        os.makedirs(save_dir, exist_ok=True)

        pred_dir = os.path.sep.join([save_dir, f'pred_perturbed.zarr'])
                
        with xr.open_zarr(val_data.file_dir) as ds:
            x_coords = ds['x'].values.copy()
            y_coords = ds['y'].values.copy()

        model_predictor = predictor.ModelPredictor(model_dir, var_dict, dataloader, filedir=pred_dir, load=f'best_train', device='cpu')
        model_predictor.make_predictions_file(coords=(x_coords, y_coords), perturb='dswrad')
        model_predictor.make_predictions_file(coords=(x_coords, y_coords), perturb='energyin', append_mode=True)

        # for d in range(1,10):   # to get sensitivity w.r.t. the preceding days
        #     model_predictor.make_predictions_file(coords=(x_coords, y_coords), perturb=f'energyin_d-{d}', append_mode=True)
        #     model_predictor.make_predictions_file(coords=(x_coords, y_coords), perturb=f'dswrad_d-{d}', append_mode=True)

        shutdown_dataloader(dataloader)
        val_data.close()
        del val_data



if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath('')).split(os.sep + 'modeling')[0]

    # # models with their best iteration; choose one to perform the perturbation study on
    # MODEL_SPECS = {'modularNNEBM': 18}                         # default emulator
    # MODEL_SPECS = {'modularNNEBMalbedo': 1}                    # with albedo 
    # MODEL_SPECS = {'shorttermNN': 8}                           # shorttermNN
    # MODEL_SPECS = {'modularNNEBMmono100': 11}                  # with extra monotonic loss term
    MODEL_SPECS = {'modularNNEBMphysical20onlyabove10mm': 2}   # with extra physical loss term

    DATA_SPEC = 'CESM2future'
    make_predictions(MODEL_SPECS, DATA_SPEC)

