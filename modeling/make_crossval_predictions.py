# -*- coding: utf-8 -*-
"""
Created on 31.03.2026
@author: eschlager
Main script to prepare data and apply trained models of a crossvalidation study or a folder with repeated training runs
"""

import logging
import os
import gc
import xarray as xr
import sys
import pandas as pd
from torch.utils.data import DataLoader
import torch
script_dir = os.path.abspath(os.path.dirname(__file__))
project_dir = os.path.sep.join([script_dir, '..'])
sys.path.append(os.path.sep.join([project_dir , 'src']))
sys.path.append(os.path.sep.join([project_dir , 'modeling', 'models']))
sys.path.append(os.path.sep.join([project_dir, 'modeling']))
import read_yaml
from prepare_trainset import ZarrDataset
import shutil
import zarr

# import local modules
script_dir = os.path.abspath(os.path.dirname(__file__))
project_dir = os.path.sep.join([script_dir, '..'])
sys.path.append(os.path.sep.join([project_dir , 'src']))
import logging_config
from create_dataset import FirnpackCellsDataset, my_collate_fn
from my_utils import AffinityInitializer, shutdown_dataloader
import predictor

import multiprocessing as mp
logging.getLogger('multiprocessing').setLevel(logging.DEBUG)


if __name__ == "__main__":

    gc.collect()
    torch.cuda.empty_cache()

    try:
        mp.set_start_method("spawn")
        logging.info("Set multiprocessing start method to 'spawn'.")
        import warnings
        warnings.simplefilter('always')
    except RuntimeError:     
        pass


    DATASET = 'ECEfuture'
    MODEL = 'CESM2future'

    # directory of trained models
    out_dir = os.path.sep.join([project_dir, 'output', f'repeated_{MODEL}_modularNNEBMphysical20onlyabove10mm_log'])

    # directory of data to apply the models
    data_dir = os.path.sep.join([project_dir, 'data', 'processed', 'crossval', 'v_01', f'{DATASET}_meltNN', 'data.zarr'])

    if 'future' in DATASET:
        dates = (pd.date_range(start="2075-01-01", end="2100-12-31")+pd.Timedelta(hours=12)).tolist()
    else:
        dates = (pd.date_range(start="1990-01-01", end="2016-12-31")+pd.Timedelta(hours=12)).tolist()
        
    ## ----- if data.zarr does not exist, can also create it here ----- ##
    # os.makedirs(os.path.join(data_dir, '..'), exist_ok=True)
    # logging_config.define_root_logger(os.path.join(data_dir, '..', 'log_data.txt'))
    # logging.getLogger().setLevel(logging.INFO)

    # # define the correct spec file:
    # specs = read_yaml.read_yaml_file(os.path.sep.join([project_dir, 'modeling', 'spec_files_transfer', f'specs_melt_modularNNEBMalbedo_{MODEL}_log.yml']))
    # zarrset = ZarrDataset(specs)

    # # define the correct base files
    # daily_file = os.path.abspath(os.path.sep.join([project_dir, 'data', 'processed','HIRHAM5-ECE', 'future', 'base_dataset.zarr']))
    # spinup_file = os.path.abspath(os.path.sep.join([project_dir, 'data', 'processed','HIRHAM5-ECE', 'future', 'base_dataset_10yr.zarr']))
    # zarrset.prepare_prediction_file(dates, daily_file, spinup_file=spinup_file, chunksize={"time":32}, savedir=data_dir)
 
    logging_config.define_root_logger(os.path.join(out_dir, f'log_predictions_{DATASET}_data_{MODEL}_model.txt'))
    logging.getLogger().setLevel(logging.INFO)

    RECONSTRUCT_COORDS = False  # to save space

    # years = range(1992, 2017, 2)     # define folds of cross-validation study
    years = [2016]                     # for repeated training; just a place-holder of one element for correct folder access

    iters = range(20)
    
    for y in years:   # iterate through validation folds
        # fold = f'fold_{y}'   # for cross-validation
        fold = ''              # for repeated training
        
        model_dir = os.path.sep.join([out_dir, fold])
        scalerpath = os.path.sep.join([model_dir, 'std_scaler.npz'])

        specs = read_yaml.read_yaml_file(os.path.sep.join([model_dir, f'iter_{iters[0]}',  'specs.yml']))

        zarrset = ZarrDataset(specs)
        var_dict = zarrset.get_variable_dict()
        # data_dir = os.path.sep.join([project_dir, specs['directories']['base_dir'], specs['directories']['data_file'], 'data.zarr'])
        val_data = FirnpackCellsDataset(data_dir, dates=dates, variable_names=var_dict, scaling=scalerpath)

        dataloader = DataLoader(val_data, batch_size=100, shuffle=False, collate_fn=my_collate_fn, num_workers=4, prefetch_factor=1,
                                persistent_workers=True)

        ds_x = xr.open_zarr(val_data.file_dir)
        x_coords = ds_x['x'].values.copy()
        y_coords = ds_x['y'].values.copy()
        ds_x.close()
        del ds_x

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

        for it in iters:   # iterate through the repeated trainings and make prediction file for each of them
            model_iter_dir = os.path.sep.join([model_dir, f'iter_{it}'])
            filedir=os.path.sep.join([model_iter_dir, f'pred_data_{DATASET}.zarr'])
            if not os.path.exists(filedir):
                model_predictor = predictor.ModelPredictor(model_iter_dir, var_dict, dataloader, filedir=filedir, load='best_train', device='cpu')

                model_predictor.make_predictions_file(coords=(x_coords, y_coords), 
                                                        ref_coords=xy_coords_ref, 
                                                        extra_coords=lonlat_coords_ref)
                del model_predictor
        
        shutdown_dataloader(dataloader)
        val_data.close()
        del val_data
                
        # merge all predictions across the iterations in one file
        filedir = os.path.sep.join([out_dir, fold, f'pred_data_{DATASET}.zarr'])

        for it in iters:
            model_dir = os.path.sep.join([out_dir, fold, f'iter_{it}'])
            file_iter = os.path.sep.join([model_dir, f'pred_data_{DATASET}.zarr'])

            if not os.path.exists(filedir):
                # copy file of first iteration
                if os.path.exists(file_iter):
                    # get file of first model but rename variable name snmel_pred
                    logging.info(f'Copying predictions from {file_iter} to {filedir} ...')
                    ds = xr.load_dataset(file_iter)
                    ds_true = ds[['snmel_true']]
                    ds_pred = ds[['snmel_pred']].expand_dims(iteration=[it])
                    ds_merged = xr.merge([ds_true, ds_pred])
                    ds_merged.to_zarr(filedir,                                      
                                    encoding={
                                            var: {"compressors": zarr.codecs.BloscCodec(cname="zstd", clevel=5)}
                                            for var in ds_merged.data_vars
                                        })
                    del ds, ds_true, ds_pred, ds_merged                    
                else:
                    logging.warning(f'First iteration predictions file {file_iter} does not exist.')
            else:
                if os.path.exists(file_iter):

                    logging.info(f'Appending predictions from {file_iter} to {filedir} ...')
                    ds = xr.load_dataset(file_iter)[['snmel_pred']]
                    ds_pred = ds[['snmel_pred']].expand_dims(iteration=[it])
                    
                    ds_pred.to_zarr(filedir, mode='a', append_dim='iteration')
                    
                    ## in case of replacing an already written prediction:
                    # ds_pred = ds_pred.drop_vars(['time', 'x', 'y'], errors='ignore')
                    # ds_pred.to_zarr(filedir, mode='r+',region={'iteration': slice(it, it+1)})

                    del ds, ds_pred
                else:
                    logging.warning(f'Predictions file of iteration {file_iter} does not exist.')

        gc.collect()

        # delete the individual prediction files if merging was successful
        success = False
        try:
            ds_check = xr.open_zarr(filedir)
            assert 'snmel_pred' in ds_check
            assert 'iteration' in ds_check.dims
            assert all(it in ds_check['iteration'] for it in iters)
            success = True
        except AssertionError as e:
            logging.error(f"Validation failed: {e}")
        except Exception as e:
            logging.error(f"Error reading zarr: {e}")

        if success:
            for it in iters:
                model_dir = os.path.sep.join([out_dir, fold, f'iter_{it}'])
                file_iter = os.path.sep.join([model_dir, f'pred_data_{DATASET}.zarr'])
                shutil.rmtree(file_iter)
        else:
            logging.warning("Not deleting source file because validation failed.")