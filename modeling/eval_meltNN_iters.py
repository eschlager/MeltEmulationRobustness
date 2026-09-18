# -*- coding: utf-8 -*-
"""
Created on 15.08.2024
@author: eschlager
Evaluation script for repeated trainings
"""

import argparse
import calendar
import json
import logging
import os
import sys

import numpy as np
import pandas as pd
from dateutil.parser import parse
import xarray as xr
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# import local modules
script_dir = os.path.abspath(os.path.dirname(__file__))
project_dir = os.path.sep.join([script_dir, '..'])
sys.path.append(os.path.sep.join([project_dir , 'src']))
import logging_config
import read_yaml
from create_dataset import FirnpackCellsDataset, my_collate_fn
import predictor
import eval_model
from prepare_trainset import ZarrDataset


def main(out_dir, iters=[None], mode='val', reconstruct_coords=False):
    """ Main evaluation function
    mode: 'val' or 'test' to evaluate on validation or test set;
          but can also be the filename of a preprocessd zarr file
    reconstruct_coords: if True, use reference coordinates to reconstruct lon/lat from x/y
    """
    model_dir = os.path.abspath(os.path.sep.join([project_dir, out_dir]))
    specs = read_yaml.read_yaml_file(os.path.sep.join([model_dir, 'specs.yml']))
    logging_config.define_root_logger(os.path.join(model_dir, f'log_eval_{mode}.txt'))
    logging.getLogger().setLevel(logging.INFO)
    
    # ---------------------------- Set up Cuda if available ----------------------------
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')   
    
    # ---------------------------- Plot loss history ----------------------------
    # plot loss history if available
    try:
        eval_model.plot_loss(model_dir)
        eval_model.plot_loss(model_dir, log_scale=True)
    except:
        pass
    
    # ---------------------------- Read data ----------------------------
    base_dir = os.path.abspath(os.path.sep.join([project_dir, specs['directories']['base_dir']]))
    
    zarrset = ZarrDataset(specs)
    var_dict = zarrset.get_variable_dict()
    file_dir = zarrset.get_file_dir()
    
    logging.info(f'Initialize {mode} Dataset ...')

    auto = specs['model'].get('auto')
    if auto is not None:
        # if model is auto-regressive, evaluate in teacher-forced (eval_modes_auto=False)
        # and auto-regressive (eval_modes_auto=True) mode
        model_is_auto = True
        eval_modes_auto = [False, True]
    else:
        model_is_auto = False
        eval_modes_auto = [False]
        


    # ---------------------------- Evaluation ----------------------------
    ref_coords = None
    extra_coords = None
    coords_file = os.path.sep.join([project_dir, 'data', 'processed', 'HIRHAM5-ERAI', 'v_04', 'coords_only.zarr'])
    coords_ref = xr.open_dataset(coords_file)
    ref_coords = (coords_ref['x'], coords_ref['y'])
    logging.info(f"Use reference coordinates x [{len(ref_coords[0])}], y [{len(ref_coords[1])}].")
    if 'lon' in coords_ref and 'lat' in coords_ref:
        extra_coords = (coords_ref['lon'], coords_ref['lat'])
        logging.info("Use reference coordinates lon/lat.")
    coords_ref.close()



    for eval_mode in eval_modes_auto:  # for autoregressive models evaluate in teacher-forced and auto-regressive mode
        eval_specifier = f'auto_{mode}' if eval_mode else mode
        pred_dir = os.path.sep.join([model_dir, f'pred_{eval_specifier}.zarr'])

        # read prediction file
        try:
            logging.info(f"Load predictions file: {pred_dir}")
            ds = xr.open_zarr(pred_dir)
        except Exception as e:
            logging.error(f"Error loading predictions: {e}")
            sys.exit()

        if ref_coords is not None:
            try:
                ds = ds.reindex({'x': ref_coords[0], 'y': ref_coords[1]})   # missing cells -> NaN
            except Exception as e:
                logging.info(f'Error reindexing to reference coordinates: {e}')
                raise ValueError(f'Error reindexing to reference coordinates: {e}')
        
        if extra_coords is not None:
            try:
                ds = ds.assign_coords(lon=extra_coords[0], lat=extra_coords[1])
            except Exception as e:
                logging.info(f'Error assigning extra coordinates: {e}')
                raise ValueError(f'Error reindexing to extra coordinates: {e}')

        # run evaluation for all target variables
        target_name = var_dict['target'][0]
        for it in iters:
            logging.info(f'Initialize ModelEvaluator for {target_name} ...')
            
            ## to create DJF map plot
            # ds = ds.sel(time=slice('2015-12-01', '2016-02-28'))
            # time = ds.indexes["time"]
            # new_time = time.where(time.year != 2015, time.map(lambda t: t.replace(year=2016)))
            # ds = ds.assign_coords(time=new_time)
            
            if it is not None:
                dss = ds.sel(iteration=it)
                save_spec = f'iter{it}'
            else:
                dss = ds
                save_spec = ''
            m_eval = eval_model.ModelEvaluator(dss, target_name=target_name, batch_size=128)
            
            val_fig_dir = os.path.sep.join([model_dir, f'{eval_specifier}_figures'])
            os.makedirs(val_fig_dir, exist_ok=True)

            val_fig_dir_hexbins = os.path.sep.join([val_fig_dir, 'hexbins'])
            os.makedirs(val_fig_dir_hexbins, exist_ok=True)

            # plot density of predictions vs target per year
            years = np.unique(ds.time.dt.year.values)
            ax_lims = {'snmel': (-7,380), 'albedom': None}
            #ax_lims = {'snmel': (-7,230), 'albedom': None}
            for y in years:
                ax = m_eval.plot_pred_vs_target_density(f'{target_name}_true', f'{target_name}_pred', year=y, ref_line='equal', x_lims=ax_lims[target_name])
                fig_dir = os.path.sep.join([val_fig_dir_hexbins, f"true_vs_pred{save_spec}_{target_name}_density_year{y}.png"])
                ax.get_figure().savefig(fig_dir, bbox_inches="tight", dpi=300)
                logging.info(f'Saved plot to {fig_dir}.')
                plt.close()

                                
            # make maps of predictions, true values, and differences true-pred
            val_fig_dir_map = os.path.sep.join([val_fig_dir, 'maps'])
            os.makedirs(val_fig_dir_map, exist_ok=True)
        
            ax = m_eval.plot_map(join_colorbar=True)
            plt.show()
            fig_dir = os.path.sep.join([val_fig_dir_map, f"map_{target_name}_{save_spec}.png"])
            ax.get_figure().savefig(fig_dir, bbox_inches="tight", dpi=300)
            logging.info(f'Saved plot to {fig_dir}.')
            plt.close()


            # totals per year
            years = np.unique(ds.time.dt.year.values)
            for y in years:
                ax = m_eval.plot_map(year=y, join_colorbar=True)
                plt.show()
                fig_dir = os.path.sep.join([val_fig_dir_map, f"map_{target_name}_{save_spec}_{y}.png"])
                ax.get_figure().savefig(fig_dir, bbox_inches="tight", dpi=300)
                logging.info(f'Saved plot to {fig_dir}.')
                plt.close()




        
    logging.shutdown()
    return


parser = argparse.ArgumentParser(description = 'Evaluate NN_firn model')
parser.add_argument('-d', '--out_dir', default='./output/test', type=str,
                                    help='directory to model ouptut')
args = parser.parse_args("")

if __name__ == "__main__":

    # # Run evaluation on test sets for best model per config:
    # main('./output/repeated_CESM2_modularNNEBM_log/', mode='data_CESM2', reconstruct_coords=True)
    # main('./output/repeated_CESM2_modularNNEBM_log/', mode='data_CESM2future', reconstruct_coords=True)
    # main('./output/repeated_CESM2_modularNNEBM_log/', mode='data_ERAI', reconstruct_coords=True)
    # main('./output/repeated_CESM2_modularNNEBM_log/', mode='data_ECE', reconstruct_coords=True)
    main('./output/repeated_CESM2future_modularNNEBM_log/', iters=[18], mode='data_CESM2future', reconstruct_coords=True)