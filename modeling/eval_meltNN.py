# -*- coding: utf-8 -*-
"""
Created on 15.08.2024
@author: eschlager
Evaluation script
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


def main(out_dir, mode='val', reconstruct_coords=False):
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
        
    # # use zones (ablation, percolation, dry-snow)
    # if 'zones_file' in specs['directories']:
    #     zone_dir = os.path.sep.join([base_dir, specs['directories']['zones_file']])
    #     zones = xr.open_dataset(zone_dir)['zones']
    #     if 'z' in zones.dims:
    #         zones = zones.set_index(z=['y', 'x']).unstack('z')
    #     logging.info(f'Loaded zones file from {zone_dir}.')
    #     zones_nr = np.unique(zones.values[~np.isnan(zones.values)]).tolist()
    # else:
    #     zones = None
    #     zones_nr = 0

    # use basins
    basins_dir = os.path.sep.join([base_dir, 'GRLmask.zarr'])
    basins = xr.open_zarr(basins_dir)['maskbas']
    if 'z' in basins.dims:
        basins = basins.set_index(z=['y', 'x']).unstack('z')
    logging.info(f'Loaded basins file from {basins_dir}.')
    basins_nr = np.unique(basins.values[~np.isnan(basins.values)]).tolist()

    # ---------------------------- Evaluation ----------------------------


    for eval_mode in eval_modes_auto:  # for autoregressive models evaluate in teacher-forced and auto-regressive mode
        eval_specifier = f'auto_{mode}' if eval_mode else mode
        pred_dir = os.path.sep.join([model_dir, f'pred_{eval_specifier}.zarr'])

        if not os.path.exists(pred_dir):   # if predictions file does not exist, create it!
            logging.info(f"Predictions haven't been created - create predictions file... ")
            
            if eval_mode:
                inference_mode = True
                logging.info(f"Evaluate model in full inference mode.")
            else:
                inference_mode = False
                if model_is_auto:
                    logging.info(f"Evaluate model in teacher-forced mode.")
                else:
                    logging.info(f"Evaluate model (model is not auto-regressive).")
            
            # set up dataloader
            date_indices = None
            if mode in ['train', 'val', 'test']:
                temp_data_split = os.path.sep.join([file_dir, '..', specs['directories']['temp_split_file']])
                train_val_test_indices = json.load(open(temp_data_split))
                if mode in train_val_test_indices.keys():
                    date_indices = train_val_test_indices[mode]
                    logging.info(f"Using {len(date_indices)} dates for {mode} set from split file {temp_data_split}.")


            if os.path.exists(file_dir+f'/{mode}.zarr'):
                datafile = file_dir+f'/{mode}.zarr'
                logging.info(f"Using data file {mode} in {file_dir}.")
            else:
                logging.error(f"No data file {mode} found in {file_dir}.")
                raise FileNotFoundError(f"No data file {mode} found in {file_dir}.")

            val_data = FirnpackCellsDataset(datafile, dates=date_indices, variable_names=var_dict, sequ_len=1, inference_mode=inference_mode)

            # use batch_size=1 for processing one day at a time (needed for auto-regressive mode)
            # writing results of multiple days to files is not implemented yet, but could be done for non-autoregressive model!
            dataloader = DataLoader(val_data, batch_size=1, shuffle=False, collate_fn=my_collate_fn, num_workers=0) 

            model_predictor = predictor.ModelPredictor(model_dir, var_dict, dataloader, filedir=pred_dir, load='best', device=device)
            x_coords = xr.open_zarr(val_data.file_dir)['x'].values
            y_coords = xr.open_zarr(val_data.file_dir)['y'].values
            
            xy_coords_ref = None
            lonlat_coords_ref = None
            if reconstruct_coords:
                try:
                    coords_file = os.path.sep.join([file_dir, '..', 'coords_only.zarr'])
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
                                                  extra_coords=lonlat_coords_ref,
                                                  inference_mode=inference_mode)


        # read prediction file
        try:
            logging.info(f"Load predictions file: {pred_dir}")
            ds = xr.open_zarr(pred_dir)
        except Exception as e:
            logging.error(f"Error loading predictions: {e}")
            sys.exit()
        
        # run evaluation for all target variables
        target_names = var_dict['target']
        for i,target_name in enumerate(target_names):
            logging.info(f'Initialize ModelEvaluator for {target_name} ...')
            
            ## to create DJF map plot
            # ds = ds.sel(time=slice('2015-12-01', '2016-02-28'))
            # time = ds.indexes["time"]
            # new_time = time.where(time.year != 2015, time.map(lambda t: t.replace(year=2016)))
            # ds = ds.assign_coords(time=new_time)

            m_eval = eval_model.ModelEvaluator(ds, target_name=target_name, batch_size=128, zones=zones)
            
            logging.info(f'{target_name}_true min: {ds[target_name+"_true"].min().compute().item()}, max: {ds[target_name+"_true"].max().compute().item()}.')
            logging.info(f'{target_name}_pred min: {ds[target_name+"_pred"].min().compute().item()}, max: {ds[target_name+"_pred"].max().compute().item()}.')
            
            # calculate scores
            rmse = m_eval.get_rmse()
            logging.info(f'RMSE {target_name}: {rmse}')
            mae = m_eval.get_mae()
            logging.info(f'MAE {target_name}: {mae}')
            mbe = m_eval.get_mbe()
            logging.info(f'MBE {target_name}: {mbe}')
            r2 = m_eval.get_r2()
            logging.info(f'R2 {target_name}: {r2}')

            if i == 0:  # get main target scores
                rmse_main = rmse
                mae_main = mae
                mbe_main = mbe
                r2_main = r2

            # val_fig_dir = os.path.sep.join([model_dir, f'{eval_specifier}_figures'])
            # os.makedirs(val_fig_dir, exist_ok=True)

            # # plot density of predictions vs target per year
            # years = np.unique(ds.time.dt.year.values)
            # ax_lims = {'snmel': (-7,180), 'albedom': None}
            # for y in years:
            #     ax = m_eval.plot_pred_vs_target_density(f'{target_name}_true', f'{target_name}_pred', year=y, ref_line='equal', x_lims=ax_lims[target_name])
            #     fig_dir = os.path.sep.join([val_fig_dir, f"true_vs_pred_{target_name}_density_year{y}.png"])
            #     ax.get_figure().savefig(fig_dir, bbox_inches="tight", dpi=300)
            #     logging.info(f'Saved plot to {fig_dir}.')
            #     plt.close()
            
            # # plot density of predictions vs target for each basin separately
            # logging.info(f'Plot groundtruth vs prediction density per basin ...')
            # for basin in basins_nr:
            #     ax = m_eval.plot_pred_vs_target_density(f'{target_name}_true', f'{target_name}_pred', ref_line='equal', zone_cat=basin)
            #     plt.show()
            #     fig_dir = os.path.sep.join([val_fig_dir, f"true_vs_pred_{target_name}_density_basin{str(basin)}.png"])
            #     ax.get_figure().savefig(fig_dir, bbox_inches="tight", dpi=300)
            #     logging.info(f'Saved plot to {fig_dir}.')
            #     plt.close()
            
            # # plot for test year per month
            # year = np.unique(ds.time.dt.year.values)[-1]  # last year in dataset
            # for m in list(calendar.month_name[1:]):
            #     ax = m_eval.plot_pred_vs_target_density(f'{target_name}_true', f'{target_name}_pred', year=year, month=m, ref_line='equal')
            #     fig_dir = os.path.sep.join([val_fig_dir, f"true_vs_pred_{target_name}_density_year{y}_month{m}.png"])
            #     ax.get_figure().savefig(fig_dir, bbox_inches="tight", dpi=300)
            #     logging.info(f'Saved plot to {fig_dir}.')
            #     plt.close()
                                
            # # make maps of predictions, true values, and differences true-pred
            # val_fig_dir_map = os.path.sep.join([val_fig_dir, 'maps'])
            # os.makedirs(val_fig_dir_map, exist_ok=True)
           
            # # totals per year
            # years = np.unique(ds.time.dt.year.values)
            # for y in years:
            #     ax = m_eval.plot_map(year=y, join_colorbar=True)
            #     plt.show()
            #     fig_dir = os.path.sep.join([val_fig_dir_map, f"map_{target_name}_{y}.png"])
            #     ax.get_figure().savefig(fig_dir, bbox_inches="tight", dpi=300)
            #     logging.info(f'Saved plot to {fig_dir}.')
            #     plt.close()


            # # test year per month
            # year = np.unique(ds.time.dt.year.values)[-1]
            # for m in list(calendar.month_name[1:]):
            #     ax = m_eval.plot_map(year=year, month=m, join_colorbar=True)
            #     plt.show()
            #     fig_dir = os.path.sep.join([val_fig_dir_map, f"map_{target_name}_{year}_month{m}.png"])
            #     ax.get_figure().savefig(fig_dir, bbox_inches="tight", dpi=300)
            #     logging.info(f'Saved plot to {fig_dir}.')
            #     plt.close()

            # # test year per season
            # seasons = {'JJA': ['June', 'July', 'August'], 'MAM': ['March', 'April', 'May'], 'SON': ['September', 'October', 'November'], 'DJF': ['December', 'January', 'February']}
            # # use pred file for entire time period, and select dec. 2015 to feb 2016 to use correct DJF data!
            # value_lims = {'JJA': (0,4900) , 'MAM': (0, 1900), 'SON': (0,1100), 'DJF': (0,270)}
            # for s, months in seasons.items():
            #     ax = m_eval.plot_map(month=months, value_lim=value_lims[s], join_colorbar=True)
            #     plt.show()
            #     fig_dir = os.path.sep.join([val_fig_dir_map, f"map_{target_name}_season{s}.png"])
            #     ax.get_figure().savefig(fig_dir, bbox_inches="tight", dpi=300)
            #     logging.info(f'Saved plot to {fig_dir}.')
            #     plt.close()

            # # some July days
            # residual_max = {'albedom':None, 'snmel':30}
            # value_lims = {'albedom':(0.35, 0.9), 'snmel':(0,100)}
            # for d in ds.time[180:221:10]:
            #     date_str = pd.to_datetime(d.astype('datetime64[D]').item()).strftime('%Y-%m-%d')
            #     ax = m_eval.plot_map(date=d, residual_max=residual_max[target_name], value_lim=value_lims[target_name])
            #     fig_dir = os.path.sep.join([val_fig_dir_map, f"map_{target_name}_date{date_str}.png"])
            #     ax.get_figure().savefig(fig_dir, bbox_inches="tight", dpi=300)
            #     logging.info(f'Saved plot to {fig_dir}.')
            #     plt.close()

            # # some single days across the year
            # residual_max = {'albedom':None, 'snmel':30}
            # value_lims = {'albedom':(0.35, 0.9), 'snmel':(0,100)}
            # # dates = pd.date_range(start="2016-01-01", end="2016-12-01", freq="1MS")+pd.Timedelta(days=20)
            # for d in dates:
            #     date_str = d.strftime('%Y-%m-%d')
            #     ax = m_eval.plot_map(date=d, residual_max=residual_max[target_name], value_lim=value_lims[target_name])
            #     fig_dir = os.path.sep.join([val_fig_dir_map, f"map_{target_name}_date{date_str}.png"])
            #     ax.get_figure().savefig(fig_dir, bbox_inches="tight", dpi=300)
            #     logging.info(f'Saved plot to {fig_dir}.')
            #     plt.close()

        
    logging.shutdown()
    return rmse_main, mae_main, mbe_main, r2_main


parser = argparse.ArgumentParser(description = 'Evaluate NN_firn model')
parser.add_argument('-d', '--out_dir', default='./output/test', type=str,
                                    help='directory to model ouptut')
args = parser.parse_args("")

if __name__ == "__main__":

    # # Run evaluation on test sets for best model per config:
    # main('./output/optuna_melt_regression/trial_0', mode='test', reconstruct_coords=True)
    # main('./output/optuna_melt_shorttermNN/trial_28', mode='test', reconstruct_coords=True)
    # main('./output/optuna_melt_modularNN/trial_21', mode='test', reconstruct_coords=True)
    # main('./output/optuna_melt_autoreg1noise/trial_18', mode='test', reconstruct_coords=True)
    # main('./output/optuna_melt_modularNN_multitarget_trainable/trial_23', mode='test', reconstruct_coords=True)
    # main('./output/optuna_melt_modularNN_noseason', mode='test', reconstruct_coords=True)
    # main('./output/optuna_melt_modularNN_albedo', mode='test', reconstruct_coords=True)
    # main('./output/optuna_melt_modularNN_EBMTd/trial_22', mode='test', reconstruct_coords=True)
    # main('./output/optuna_melt_modularNN_EBMT/trial_18', mode='test', reconstruct_coords=True)
    # main('./output/optuna_melt_modularNN_EBM/trial_13', mode='test', reconstruct_coords=True)
    # main('./output/optuna_melt_modularNN_EBS/trial_9', mode='test', reconstruct_coords=True)
    # main('./output/optuna_melt_modularNN_EBR/trial_22', mode='test', reconstruct_coords=True)
    main('./output/optuna_melt_modularNN_EBMnodoy/trial_21', mode='test', reconstruct_coords=True)