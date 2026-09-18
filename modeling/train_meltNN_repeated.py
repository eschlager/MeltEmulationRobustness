# -*- coding: utf-8 -*-
"""
Created on 29.01.2025
@author: eschlager
main script for training repeatedly with random layer initialization and training data shuffle
"""

import argparse
import glob
import gc
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
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from datetime import datetime
import xarray as xr
from yaml import safe_dump as dump

# import local modules
script_dir = os.path.abspath(os.path.dirname(__file__))
project_dir = os.path.sep.join([script_dir, '..'])
sys.path.append(os.path.sep.join([project_dir, 'src']))
from train_model import ModelTrainer
from create_dataset import FirnpackCellsDataset, my_collate_fn
from my_utils import AffinityInitializer, shutdown_dataloader
import read_yaml
import logging_config
import eval_model
import predictor
from prepare_trainset import ZarrDataset
import meltNN

torch.backends.cudnn.allow_tf32 = True


import psutil


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
            # shutil.copytree(source_dir, out_dir_abs)
            for filename in glob.glob(os.path.join(source_dir, '*.*')):
                try:
                    _, fname = os.path.split(filename)
                    shutil.copy2(filename, os.path.join(out_dir_abs, fname))
                except:
                    pass

    return out_dir_new, out_dir_abs

def dump_specs_file(specs, out_dir_abs):
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

def get_scaler(scaler_dir):
    scaler_path = os.path.sep.join([scaler_dir, 'std_scaler.npz'])
    try:
        with np.load(scaler_path) as npz:
            scaler = {k: npz[k].copy() for k in npz.files}
            logging.info(f"Load scaler file {scaler_path}")
        return scaler
    except FileNotFoundError:
        logging.warning("No scaler file found.")
        return None


def perform_training(specs):
    torch.cuda.empty_cache() 
    process = psutil.Process()
    
    # ---------------------------- Copy specs file to out_dir ----------------------------
    out_dir_abs = os.path.abspath(os.path.sep.join([project_dir, specs['directories']['out_dir']]))
    dump_specs_file(specs, out_dir_abs)

    
    # ---------------------------- Set up Cuda if available ----------------------------
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # ---------------------------- Prep training files from base file --------------------------

    zarrset = ZarrDataset(specs)
    file_dir = zarrset.get_file_dir()
    source_dir = os.path.abspath(os.path.sep.join([project_dir, specs['directories']['source_dir']]))
    daily_file = os.path.abspath(os.path.sep.join([source_dir, 'base_dataset.zarr']))
    spinup_file = os.path.abspath(os.path.sep.join([source_dir, 'base_dataset_10yr.zarr']))
    
    batch_size = specs['training']['batch_size']
       
    ## Use scaler from pre-trained model
    #scaler_file = os.path.abspath(os.path.sep.join([project_dir, specs['directories']['out_dir'], '..', 'std_scaler.npz']))
    scaler_file = None   # set none to fit new scaler to training data
      
    zarrset = ZarrDataset(specs)
    try:
        zarrset.check_file()
    except (FileNotFoundError, KeyError) as e:
        zarrset.make_file(scaler_file=scaler_file)       
    except:
        logging.error(f"There is a problem with the file - please check!")
        os._exit(0)

    # Get variables names to be used according to specs file
    var_dict = zarrset.get_variable_dict()
    
    # needed for physical loss
    if scaler_file is None:
        scaler = get_scaler(file_dir)
        logging.info(scaler)
        

    # ---------------------------- Set up data loaders --------------------------
    
    batch_size = specs['training']['batch_size']
    
    train_data_zarr = os.path.sep.join([file_dir, 'train_sub.zarr'])
    logging.info(f'Initialize (sub-sampled) training dataset from {train_data_zarr} ...')
    train_data = FirnpackCellsDataset(train_data_zarr, variable_names=var_dict)
    
    val_data_zarr = os.path.sep.join([file_dir, 'val_sub.zarr'])
    logging.info(f'Initialize (sub-sampled) validation dataset from {val_data_zarr} ...')
    val_data_sub = FirnpackCellsDataset(val_data_zarr, variable_names=var_dict)

    logging.info('Initialize DataLoaders ...')

    
    pin_memory = device.type == 'cuda'   # pin_memory if using GPU; if use pin_memory with CPU it just creates overhead!
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=my_collate_fn, num_workers=14, persistent_workers=True, prefetch_factor=4,
                                  pin_memory=pin_memory, worker_init_fn=AffinityInitializer(base_offset=1, cores_per_worker=1, name='train'))
    val_sub_dataloader = DataLoader(val_data_sub, batch_size=batch_size, shuffle=False, collate_fn=my_collate_fn, num_workers=2, persistent_workers=True,
                                pin_memory=pin_memory, worker_init_fn=AffinityInitializer(base_offset=9, cores_per_worker=2, name='val'))

    val_data = FirnpackCellsDataset(file_dir+'/val.zarr', variable_names=var_dict)
    val_dataloader = DataLoader(val_data, batch_size=1, shuffle=False, collate_fn=my_collate_fn, num_workers=0)
    
    # ---------------------------- Train multiple times --------------------------
    
    df = pd.DataFrame(columns=['iter', 'rmse', 'mae', 'mbe', 'r2', 'train_loss', 'stop_loss'])
    
    for iter in range(specs['training']['restarts']):
    
        logging.info(f'-------------- START ITERATION NR {iter} --------------')
        out_dir_iter = os.path.abspath(os.path.sep.join([out_dir_abs, f'iter_{iter}']))
        os.makedirs(out_dir_iter, exist_ok=True)
        with open(os.path.join(out_dir_iter, f'specs.yml'), 'w') as f:   # copy specs file also to folder where model is saved
            dump(specs, f, default_flow_style=False)
        
        logging.info('Initialize meltNN model ...')
        model = meltNN.init_model(var_dict, specs).to(device)
        logging.info(model)
        
        logging.info(f"Start training ...")
        trainer = ModelTrainer(model, train_dataloader, val_sub_dataloader, specs['training'], out_dir_iter, device=device, scaler=scaler)
        start_time = time.time()
        (train_loss, val_loss, epoch, success) = trainer.train()
        torch.cuda.current_stream().synchronize()
        end_time = time.time()
        logging.info(f"Training completed in {(end_time - start_time)/60:.2f} minutes.")
        
        trainer.close()
        del trainer
        del model
        gc.collect()
        torch.cuda.empty_cache()

        eval_model.plot_loss(out_dir_iter)
        eval_model.plot_loss(out_dir_iter, log_scale=True)

        # ---------------------------- Evaluate model on the validation dataset ----------------------------

        logging.info('Initialize validation dataset ...')

        pred_dir = os.path.sep.join([out_dir_iter, f'pred_val.zarr'])
            
        logging.info(f"Predictions haven't been created - create predictions file... ")

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

        model_predictor.close()
        del model_predictor
        del x_coords, y_coords 
        gc.collect()
        torch.cuda.empty_cache()
        
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

        df.loc[len(df)] = { 'iter':iter, 
                            'rmse':scores['snmel_rmse'], 'mae':scores['snmel_mae'], 
                            'mbe':scores['snmel_mbe'], 'r2':scores['snmel_r2'],
                            'train_loss':train_loss, }
        df.to_csv(os.path.sep.join([out_dir_abs, 'scores_val.csv']), sep=',', index=False)
        

        logging.info(
            f"RAM={process.memory_info().rss/1024**3:.2f} GB, \n "
            f"GPU alloc={torch.cuda.memory_allocated()/1024**3:.2f} GB, \n"
            f"GPU reserved={torch.cuda.memory_reserved()/1024**3:.2f} GB, \n"
            f"nr files: {len(psutil.Process().open_files())}"
        )
        
        
        

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
    shutdown_dataloader(val_dataloader)
    shutdown_dataloader(val_sub_dataloader)
    train_data.close()
    val_data.close()
    val_data_sub.close()
    del train_data, val_data_sub
    del val_data
          

    gc.collect()
    torch.cuda.empty_cache()
    
    


parser = argparse.ArgumentParser(description='Train meltNN model')
parser.add_argument('-s', '--specifications', default='./spec_files_transfer/specs_melt_shorttermNN_ECEfuture_log.yml',
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