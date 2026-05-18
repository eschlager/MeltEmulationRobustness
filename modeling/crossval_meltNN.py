# -*- coding: utf-8 -*-
"""
Created on 31.03.2026
@author: eschlager
main script for crossvalidation
"""

import argparse
import glob
import gc
import json
import logging
import multiprocessing as mp
import os
import re
import shutil
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from datetime import datetime
import xarray as xr
from yaml import safe_dump as dump
import matplotlib.pyplot as plt

# import local modules
script_dir = os.path.abspath(os.path.dirname(__file__))
project_dir = os.path.sep.join([script_dir, '..'])
sys.path.append(os.path.sep.join([project_dir, 'src']))
from train_model import ModelTrainer
from create_dataset import FirnpackCellsDataset, my_collate_fn
from my_utils import AffinityInitializer, shutdown_dataloader
import read_yaml
import logging_config
import predictor
import eval_model
from prepare_trainset import ZarrDataset
import meltNN

torch.backends.cudnn.allow_tf32 = True

def delete_path_tree(path: Path):
    path = Path(path)
    for p in sorted(path.rglob("*"), reverse=True):
        if p.is_file() or p.is_symlink():
            p.unlink()
        elif p.is_dir():
            p.rmdir()
    path.rmdir()

def get_specs(config):
    if isinstance(config, str):
        yaml_file = os.path.sep.join([script_dir, config])
        if os.path.isfile(yaml_file):
            if config.lower().endswith(('.yaml', '.yml')):
                print("Read specifications from YAML file:", yaml_file)
                specs = read_yaml.read_yaml_file(yaml_file)
            else:
                print("Input is not a YAML file:", yaml_file)
        else:
            print("File does not exist:", yaml_file)
    elif isinstance(config, dict):
        specs = config
    else:
        print("Input type is not supported:", type(config))
    return specs

def mk_my_outdir(specs):
    out_dir_new = specs['directories']['out_dir']
    out_dir_abs = os.path.abspath(os.path.sep.join([project_dir, out_dir_new]))
    
    # if continue training from previous results, make directory with suffix _contd
    if 'continue_training' in specs['training']:
        if specs['training']['continue_training']:
            if not os.path.exists(out_dir_abs):
                raise OSError(
                    f'Cannot continue training from non-existing directory {out_dir_abs}!')
            else:
                out_dir_new = out_dir_new + '_contd'
                source_dir = out_dir_abs
                out_dir_abs = os.path.abspath(os.path.sep.join([project_dir, out_dir_new]))

    # if out_dir already exits:
    # - if not in overwrite mode, add timestamp to out_dir
    # - if in overwrite mode, delete existing directory and create new one with that name
    if os.path.exists(out_dir_abs):
        if not specs['directories']['overwrite']:
            out_dir_new = out_dir_new + '_' + str(datetime.now().strftime('%Y%m%d_%H%M%S'))
            
        else:
            delete_path_tree(out_dir_abs)

    # if continue training, copy all files from previous results to new folder,
    # to have whole loss history, best model, etc. in the continued training folder
    out_dir_abs = os.path.abspath(os.path.sep.join([project_dir, out_dir_new]))
    os.makedirs(out_dir_abs)
    if 'continue_training' in specs['training']:
        if specs['training']['continue_training']:
            for root, dirs, files in os.walk(source_dir):
                # Calculate the destination directory
                rel_path = os.path.relpath(root, source_dir)
                dest_dir = os.path.join(out_dir_abs, rel_path)
                os.makedirs(dest_dir, exist_ok=True)
                for file in files:
                    src_file = os.path.join(root, file)
                    dest_file = os.path.join(dest_dir, file)
                    shutil.copy2(src_file, dest_file)

    return out_dir_new, out_dir_abs


def perform_training(specs):
    torch.cuda.empty_cache() 
    
    # ---------------------------- Copy specs file to out_dir ----------------------------
    out_dir_abs = os.path.abspath(os.path.sep.join([project_dir, specs['directories']['out_dir']]))

    inputs = specs['data'].get('input')
    if isinstance(inputs, str):
        specs['data']['input'] = re.split(r',\s*', inputs)

    layers_daily = specs['model'].get('layers_daily_feat_extractor')
    if isinstance(layers_daily, str):
        specs['model']['layers_daily_feat_extractor'] = [int(x) for x in re.split(r',\s*', layers_daily)]

    layers_medrange = specs['model'].get('layers_medrange_feat_extractor')
    if isinstance(layers_medrange, str):
        specs['model']['layers_medrange_feat_extractor'] = [int(x) for x in re.split(r',\s*', layers_medrange)]

    layers_spinup = specs['model'].get('layers_spinup_feat_extractor')
    if isinstance(layers_spinup, str):
        specs['model']['layers_spinup_feat_extractor'] = [int(x) for x in re.split(r',\s*', layers_spinup)]
    
    with open(os.path.join(out_dir_abs, f'specs.yml'), 'w') as f:
        dump(specs, f, default_flow_style=False)

    # ---------------------------- Set up Cuda if available ----------------------------
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # ---------------------------- Prep zarr files from base file --------------------------

    zarrset = ZarrDataset(specs)
    file_dir = zarrset.get_file_dir()
    dates = pd.date_range(start='1990-01-01', end='2016-12-31', freq='D')
    base_dir = os.path.abspath(os.path.sep.join([project_dir, specs['directories']['base_dir']]))
    base_dir = os.path.abspath(os.path.sep.join([project_dir, 'data', 'processed', 'HIRHAM5-ERAI', 'v_04']))
    daily_file = os.path.abspath(os.path.sep.join([base_dir, 'base_dataset.zarr']))
    spinup_file = os.path.abspath(os.path.sep.join([base_dir, 'base_dataset_10yr.zarr']))

    file_name = 'data.zarr'
    file_name_sub = 'data_sub.zarr'

    try:
        zarrset.check_file(file_name)
        logging.info('   succesful!')
    except (FileNotFoundError, KeyError) as e:
        zarrset.prepare_prediction_file(dates, daily_file, spinup_file=spinup_file, chunksize={"time":32}, savedir=os.path.sep.join([file_dir, file_name]))
    except:
        logging.error(f"There is a problem with the file {file_name} - please check!")
        os._exit(0)

    try:
        zarrset.check_file(file_name_sub)
        logging.info('   succesful!')
    except (FileNotFoundError, KeyError) as e:
        zarrset.prepare_prediction_file(dates, daily_file, spinup_file=spinup_file, use_spatial_subsampling=True, savedir=os.path.sep.join([file_dir, file_name_sub]))
    except:
        logging.error(f"There is a problem with the file {file_name_sub} - please check!")
        os._exit(0)

    # Get variables names to be used according to specs file
    var_dict = zarrset.get_variable_dict()

    # ---------------------------- Define crossvalidation folds ----------------------------
    
    temp_sub_file = os.path.sep.join([file_dir, '..', specs['directories']['temp_sub_file']])
    with open(temp_sub_file, 'r') as f: 
        temp_sub_dates = pd.DatetimeIndex(json.load(f)['dates_sub'])

    with xr.open_zarr(file_dir+'/'+file_name, consolidated=True) as ds:
        data_dates = pd.DatetimeIndex(ds['time'].values)
    data_years = data_dates.year

    # stop_years = [2014, 2012, 2010, 2008, 2006, 2004, 2002, 2000, 1998, 1996, 1994, 1992, 1990]
    # val_years = [2016, 2014, 2012, 2010, 2008, 2006, 2004, 2002, 2000, 1998, 1996, 1994, 1992]
    # assert np.isin(val_years, data_years.values).all(), "Validation years must be in the dataset."
    # assert np.isin(stop_years, data_years.values).all(), "Stop years must be in the dataset."

    def time_series_cv_split(data_dates, sub_dates, stop_year, val_year):
        # use only sub-sampled dates for training
        if not isinstance(stop_year, list):
            stop_year = [stop_year]
        exclude_years = {val_year, *stop_year}
        train_dates = [d for d in sub_dates if d.year not in exclude_years]
        stop_dates = [d for d in sub_dates if d.year in stop_year]
        # use all available dates for validation set
        val_dates = [d for d in data_dates if d.year == val_year]
        return train_dates, stop_dates, val_dates

    # stop_years = [2012, 2010, 2008, 2006, 2004, 2002, 2000, 1998, 1996, 1994, 1992]
    val_years = [1996, 2004]
    stop_years = [1996, 2004]
    assert np.isin(val_years, data_years.values).all(), "Validation years must be in the dataset."
    assert np.isin(stop_years, data_years.values).all(), "Stop years must be in the dataset."

    batch_size = specs['training']['batch_size']

    

    # loop over folds
    logging_config.close_all_file_handlers()
    logging.shutdown()
    for foldnr, (stop_year, val_year) in enumerate(zip(stop_years, val_years)):
        out_dir_fold = os.path.abspath(os.path.sep.join([out_dir_abs, f'fold_{val_year}']))
        os.makedirs(out_dir_fold, exist_ok=True)
        logging_config.define_root_logger(os.path.join(out_dir_fold, f'log.txt'))
        logging.getLogger().setLevel(logging.INFO)

        df = pd.DataFrame(columns=['fold_nr', 'val_year', 'iter', 'rmse', 'mae', 'mbe', 'r2', 'train_loss', 'stop_loss'])

        # ---------------------------- Set up data for fold ----------------------------
        logging.info(f'------------ Fold {foldnr + 1}/{len(val_years)} with validation year {val_year} ------------')
        train_dates, stop_dates, val_dates = time_series_cv_split(data_dates, temp_sub_dates, stop_year, val_year)

        logging.info(f'Initialize sub-sampled training dataset from {file_dir} using {len(train_dates)} training days...')
        train_data = FirnpackCellsDataset(file_dir+'/'+file_name_sub, dates=train_dates, variable_names=var_dict, scaling='fit')
        scalerpath = os.path.sep.join([out_dir_fold, 'std_scaler.npz'])
        logging.info(f'  Save scaling parameters in: {scalerpath}.')
        np.savez(scalerpath, **train_data.scaler_vals)

        logging.info(f'Initialize sub-sampled dataset for early stopping using {len(stop_dates)} days from {stop_year}...')
        stop_data = FirnpackCellsDataset(file_dir+'/'+file_name_sub, dates=stop_dates, variable_names=var_dict, scaling=scalerpath)

        logging.info(f'Initialize validation data using {len(val_dates)} days from {val_year}...')
        val_data = FirnpackCellsDataset(file_dir+'/'+file_name, dates=val_dates, variable_names=var_dict, scaling=scalerpath)

        logging.info('Initialize DataLoaders ...')
        pin_memory = device.type == 'cuda'   # pin_memory if using GPU; if use pin_memory with CPU it just creates overhead!
        
        # shuffle training data once and then keep order fixed during training
        import random
        random.seed(42)
        shuffled_indices = list(range(len(train_data)))
        random.shuffle(shuffled_indices)
        shuffled_train_data = torch.utils.data.Subset(train_data, shuffled_indices)
        train_dataloader = DataLoader(shuffled_train_data, batch_size=batch_size, shuffle=False, collate_fn=my_collate_fn, num_workers=8, persistent_workers=True,
                                    pin_memory=pin_memory, worker_init_fn=AffinityInitializer(base_offset=1, cores_per_worker=1, name='train'))
        
        
        # train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=my_collate_fn, num_workers=8, persistent_workers=True,
        #                             pin_memory=pin_memory, worker_init_fn=AffinityInitializer(base_offset=1, cores_per_worker=1, name='train'))
        stop_dataloader = DataLoader(stop_data, batch_size=batch_size, shuffle=False, collate_fn=my_collate_fn, num_workers=4, persistent_workers=True,
                                    pin_memory=pin_memory, worker_init_fn=AffinityInitializer(base_offset=9, cores_per_worker=1, name='earlystop'))
        val_dataloader = DataLoader(val_data, batch_size=32, shuffle=False, collate_fn=my_collate_fn, num_workers=0)


        # ---------------------------- Perform training ----------------------------

        for iter in range(specs['training']['restarts']):     # restart each fold multiple time for robustness
            logging.info(f'-------------- START ITERATION NR {iter} --------------')
            out_dir_iter = os.path.abspath(os.path.sep.join([out_dir_fold, f'iter_{iter}']))
            os.makedirs(out_dir_iter, exist_ok=True)
            with open(os.path.join(out_dir_iter, f'specs.yml'), 'w') as f:   # copy specs file also to folder where model is saved
                dump(specs, f, default_flow_style=False)
            
            logging.info('Initialize meltNN model ...')
            model = meltNN.init_model(var_dict, specs).to(device)
            logging.info(model)

            logging.info(f"Start training ...")
            trainer = ModelTrainer(model, train_dataloader, stop_dataloader, specs['training'], out_dir_iter, device=device)
            start_time = time.time()
            (train_loss, stop_loss, epoch, success) = trainer.train()
            torch.cuda.current_stream().synchronize()
            end_time = time.time()
            trainer.close()
            del trainer, model
            logging.info(f"Training completed in {(end_time - start_time)/60:.2f} minutes.")
            
            gc.collect()
            torch.cuda.empty_cache()

            # ---------------------------- Evaluate model on the val dataset ----------------------------

            logging.info('Initialize validation dataset ...')

            pred_dir = os.path.sep.join([out_dir_iter, f'pred_val.zarr'])
                
            logging.info(f"Predictions haven't been created - create predictions file... ")

            # val_data = FirnpackCellsDataset(file_dir+'/data.zarr', dates=val_dates, variable_names=var_dict)
            # val_dataloader = DataLoader(val_data, batch_size=1, shuffle=False, collate_fn=my_collate_fn, num_workers=0)
            
            xy_coords_ref = None
            lonlat_coords_ref = None
            try:
                coords_file = os.path.sep.join([file_dir, '..', 'coords_only.zarr'])
                with xr.open_dataset(coords_file) as coords_ref:
                    if 'x' in coords_ref and 'y' in coords_ref:
                        xy_coords_ref = (coords_ref['x'].values.copy(), coords_ref['y'].values.copy())
                        if 'lon' in coords_ref and 'lat' in coords_ref:
                            lonlat_coords_ref = (coords_ref['lon'].values.copy(), coords_ref['lat'].values.copy())
                        else:
                            lonlat_coords_ref = None
            except Exception as e:
                logging.warning(f"Could not read coordinates from reference file {coords_file}: {e}")

            with xr.open_zarr(val_data.file_dir) as ds:
                x_coords = ds['x'].values.copy()
                y_coords = ds['y'].values.copy()
            model_predictor = predictor.ModelPredictor(out_dir_iter, var_dict, val_dataloader, filedir=pred_dir, load=f'best_train', device=device)
            model_predictor.make_predictions_file(coords=(x_coords, y_coords), 
                                                ref_coords=xy_coords_ref, extra_coords=lonlat_coords_ref)

            
            try:
                logging.info(f"Load predictions file: {pred_dir}")
                with xr.open_zarr(pred_dir) as ds:
            
                    eval_model.plot_loss(out_dir_iter)
                    eval_model.plot_loss(out_dir_iter, log_scale=True)

                    target_names = var_dict['target']
                    scores = {}

                    for target_name in target_names:
                        m_eval = eval_model.ModelEvaluator(ds, target_name=target_name, batch_size=128)
                    
                        rmse = m_eval.get_rmse()
                        scores[target_name+'_rmse'] = rmse
                        logging.info(f'RMSE {target_name}: {rmse}')
                        mae = m_eval.get_mae()
                        scores[target_name+'_mae'] = mae
                        logging.info(f'MAE {target_name}: {mae}')
                        mbe = m_eval.get_mbe()
                        scores[target_name+'_mbe'] = mbe
                        logging.info(f'MBE {target_name}: {mbe}')
                        r2 = m_eval.get_r2()
                        scores[target_name+'_r2'] = r2
                        logging.info(f'R2 {target_name}: {r2}')

                        # plot density of predictions vs target
                        ax = m_eval.plot_pred_vs_target_density(f'{target_name}_true', f'{target_name}_pred', ref_line='equal')
                        fig_dir = os.path.sep.join([out_dir_iter, f"true_vs_pred_{target_name}_density.png"])
                        ax.get_figure().savefig(fig_dir, bbox_inches="tight", dpi=200)
                        logging.info(f'Saved plot to {fig_dir}.')
                        plt.close()

            except Exception as e:
                logging.error(f"Error loading predictions: {e}")
                sys.exit()

            df.loc[len(df)] = {'fold_nr':foldnr, 'val_year':val_year,'iter':iter, 
                                                      'rmse':scores['snmel_rmse'], 'mae':scores['snmel_mae'], 
                                                      'mbe':scores['snmel_mbe'], 'r2':scores['snmel_r2'],
                                                      'train_loss':train_loss, 'stop_loss':stop_loss}
            df.to_csv(os.path.sep.join([out_dir_fold, 'scores_val.csv']), sep=',', index=False)

        scores_file = os.path.sep.join([out_dir_abs, 'scores_val.csv'])
        if os.path.exists(scores_file):
            mode = 'a'
            header = False
        else:
            mode = 'w'
            header = True
        df.to_csv(
            scores_file,
            sep=',',
            index=False,
            mode=mode,
            header=header
        )


        # clean up
        logging_config.close_all_file_handlers()
        logging.shutdown()
        gc.collect()
        torch.cuda.empty_cache()
        

        
        shutdown_dataloader(train_dataloader)
        shutdown_dataloader(stop_dataloader)
        shutdown_dataloader(val_dataloader)
        train_data.close()
        stop_data.close()
        val_data.close()
        del train_data, stop_data, val_data
        model_predictor.close()
        del model_predictor
        del x_coords, y_coords           

        gc.collect()
        torch.cuda.empty_cache()


parser = argparse.ArgumentParser(description='Train meltNN model')
parser.add_argument('-s', '--specifications', default='./spec_files_crossval/specs_melt_modularNNEBM_ERAI_small_transform.yml',
                    help='name of yaml-file with model and training initialisation')
args = parser.parse_args("")



if __name__ == "__main__":
    
    torch.cuda.empty_cache() 
    try:
       if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("spawn")
    except RuntimeError:     
        pass

    # get training specifications and set up logger file
    specs = get_specs(args.specifications)
    specs['directories']['out_dir'], out_dir_abs = mk_my_outdir(specs)
    logging_config.define_root_logger(os.path.join(out_dir_abs, f'log.txt'))
    logging.getLogger().setLevel(logging.INFO)
    logging.info(specs)
    
    perform_training(specs)