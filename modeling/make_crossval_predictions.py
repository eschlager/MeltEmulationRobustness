# -*- coding: utf-8 -*-
"""
Created on 15.11.2025
@author: eschlager
Main script to prepare data and apply trained model
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
import eval_meltNN
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

    # Create predictions for 1990-2016

    # # define which model to use
    # out_dir = os.path.sep.join(['/dmidata', 'projects', 'smb_and_polar_rcm', 'elkesc', 'output', 'Crossval', 'crossval_CESM2_modularNNEBM'])
    # data_dir = os.path.sep.join(['/dmidata', 'users', 'elkesc', 'MeltEmulation', 'data', 'processed', 'crossval', 'v_01', 'CESM2_meltNN', 'data.zarr'])
    out_dir = os.path.sep.join([project_dir, 'output', 'crossval_ERAI_modularNNEBM_noshuffle_smallNN_transform_20260515_144453'])
    data_dir = os.path.sep.join([project_dir, 'data', 'processed', 'crossval', 'v_01', 'ERAI_meltNN', 'data.zarr'])


    #out_dir = os.path.sep.join([project_dir, 'output', 'crossval_ERAI_modularNNEBM_stabilized'])

    logging_config.define_root_logger(os.path.join(out_dir, f'log_predictions_ERA_data_ERAI_model.txt'))
    logging.getLogger().setLevel(logging.INFO)

    RECONSTRUCT_COORDS = False  # to save space?

    # only few models and 2 years for testing purpose
    years = range(2002, 2017, 2) #range(1992, 2017, 2)
    years = [1996, 2004]
    iters = range(20)
    dates = (pd.date_range(start="1990-01-01", end="2016-12-31")+pd.Timedelta(hours=12)).tolist()

    for y in years:
        fold = f'fold_{y}'
        model_dir = os.path.sep.join([out_dir, fold])
        scalerpath = os.path.sep.join([model_dir, 'std_scaler.npz'])

        specs = read_yaml.read_yaml_file(os.path.sep.join([model_dir, f'iter_{iters[0]}',  'specs.yml']))

        zarrset = ZarrDataset(specs)
        var_dict = zarrset.get_variable_dict()
        # data_dir = os.path.sep.join([project_dir, specs['directories']['base_dir'], specs['directories']['data_file'], 'data.zarr'])
        val_data = FirnpackCellsDataset(data_dir, dates=dates, variable_names=var_dict, scaling=scalerpath)

        # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        device = 'cpu'
        logging.info(f'Using device {device}')   
        #dataloader = DataLoader(val_data, batch_size=32, shuffle=False, collate_fn=my_collate_fn, num_workers=4, prefetch_factor=2)
        dataloader = DataLoader(val_data, batch_size=366, shuffle=False, collate_fn=my_collate_fn, num_workers=4, prefetch_factor=2,
                                persistent_workers=True, worker_init_fn=AffinityInitializer(base_offset=0, cores_per_worker=4))
        # dataloader = DataLoader(val_data, batch_size=366, shuffle=False, collate_fn=my_collate_fn, num_workers=4, prefetch_factor=2,)
                                # persistent_workers=True, worker_init_fn=AffinityInitializer(base_offset=0, cores_per_worker=4))
        
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

        for it in iters:
            model_iter_dir = os.path.sep.join([model_dir, f'iter_{it}'])
            filedir=os.path.sep.join([model_iter_dir, 'pred_data.zarr'])
            if not os.path.exists(filedir):
                model_predictor = predictor.ModelPredictor(model_iter_dir, var_dict, dataloader, filedir=filedir, load='best', device=device)

                model_predictor.make_predictions_file(coords=(x_coords, y_coords), 
                                                        ref_coords=xy_coords_ref, 
                                                        extra_coords=lonlat_coords_ref)
                del model_predictor
        
        shutdown_dataloader(dataloader)
        val_data.close()
        del val_data 
                
        # merge all predictions across the iterations in one file
        filedir = os.path.sep.join([out_dir, fold, 'pred_data.zarr'])


        for it in iters:
            model_dir = os.path.sep.join([out_dir, fold, f'iter_{it}'])
            file_iter = os.path.sep.join([model_dir, 'pred_data.zarr'])

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
                                            var: {"compressor": zarr.codecs.BloscCodec(cname="zstd", clevel=5)}
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
                    del ds, ds_pred
                else:
                    logging.warning(f'First iteration predictions file {file_iter} does not exist.')

        gc.collect()

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
                file_iter = os.path.sep.join([model_dir, 'pred_data.zarr'])
                shutil.rmtree(file_iter)
        else:
            logging.warning("Not deleting source file because validation failed.")