# -*- coding: utf-8 -*-
"""
Created on 23.07.2024
@author: eschlager
Script for creating dataset for firn emulator
"""
import sys
import os
import json
import logging
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
import xarray as xr
import re
from pathlib import Path

script_dir = os.path.abspath(os.path.dirname(__file__))
project_dir = os.path.sep.join([script_dir, '..'])
sys.path.append(os.path.sep.join([project_dir, 'modeling']))
import read_yaml
from my_utils import AffinityInitializer
from prepare_trainset import ZarrDataset


import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class FirnpackCellsDataset(Dataset):
    # for cellwise firnpack emulation
    
    def __init__(self, data_dir:str, variable_names:dict, dates:list=[], sequ_len:int=1, inference_mode:bool=False, scaling=None):
        """
        Arguments:
            data_dir (string): Directory of the data.
            dates (list): list of dates used for training/validation/test set; must be subset of data in data_dir.
            variable_names (dict): Dict of variable names for daily, medium-range, spinup, auto and target variables.
            sequ_len (int): number of auto-reg days; used for training, to load data of previous days.
            inference_mode (bool): if True, use whole timeseries for making predictions one after another; 
                                   if False, use sequ_len days of data for making predictions.
            scaling: None (no scaling), 'fit' (fit scaler to training data), or a dict (to apply)
        """
        
        self.file_dir = os.path.join(data_dir)
        logging.info(f'Initialize data files from {self.file_dir}')
        if dates:
            logging.info(f'Using {len(dates)} dates for dataset.')
            self.dates = pd.to_datetime(dates)
        else: 
            self.dates = None
            logging.info(f'Using all dates in dataset.')
        
        self.inference_mode = inference_mode # inference_mode: is use whole timeseries, and make one prediction after another
        if not self.inference_mode:
            # if not inference mode, need to load previous data for making predictions
            self.sequ_len = sequ_len
        else: 
            # if inference mode, do not need previous data, becaue make on step after another anyways
            self.sequ_len = 1    

        self.daily_inputs = variable_names['daily']
        self.medrange_inputs = variable_names['medrange']
        self.spinup_inputs = variable_names['spinup']
        self.auto_inputs = variable_names['auto']
        self.target_vars = variable_names['target']
        if 'trunoff' in variable_names:
            self.trunoff_var = variable_names['trunoff']
        else:
            self.trunoff_var = []
        
        
        
        with xr.open_zarr(self.file_dir, consolidated=True, chunks='auto') as ds:
            self.data_var_names = list(ds.coords["variable_names"].values)
            logging.info(f'Variable names in dataset: {self.data_var_names}')
            self.batch_dim = "time"
            time_values = pd.to_datetime(ds.time.values)
            self.z_values = ds.z.values
            self.nr_samples = len(time_values)*len(self.z_values)
            logging.info(f'Nr of samples: {self.nr_samples}')
            # self.chunks = len(ds.chunks[self.batch_dim])
            self.chunks = ds.chunks[self.batch_dim]
            logging.info(f'Nr of chunks in dataset: {len(self.chunks)}')

            # subset_index = mapping from subset_index to index of full dataset
            if self.dates is not None:
                self.subset_indices = [i for i, d in enumerate(time_values) if d in self.dates]
            else:
                self.dates = time_values
                self.subset_indices = list(range(len(time_values)))
            logging.info(f'Nr of chunks selected: {len(self.subset_indices)}')

            if "source" in ds.variables:
                self.source_values = ds["source"].values
            else:
                self.source_values = None
            
            if inference_mode: # check if data is a continous timeseries for inference mode
                sorted_times = sorted(time_values.drop_duplicates())
                expected_times = pd.date_range(start=sorted_times[0], end=sorted_times[-1], freq='D')
                is_complete = (
                    len(sorted_times) == len(expected_times) and
                    (sorted_times == expected_times).all()
                )
                if not is_complete:
                    logging.warning("Time series has missing dates. Cannot predict using auto-regressive mode with missing dates!.")
                    raise ValueError("Time series has missing dates. Cannot predict using auto-regressive mode with missing dates!.")

            self._get_variable_indices()
            if scaling == 'fit':
                ds_sel = ds.isel(time=self.subset_indices).sel(variable_names=self.all_needed_vars)
                self.fit_scaler(ds_sel)
            elif isinstance(scaling, str):
                try:
                    self.load_scaler(scaling)
                except Exception as e:
                    logging.error(f"Error loading scaler from {scaling}: {e}")
                    raise FileNotFoundError(f"Error loading scaler from {scaling}: {e}")
            else:
                self.scaler_vals = None

        self.data = None

        logging.info("Finished Dataset __init__")


    def fit_scaler(self, _ds):
        logging.info("Fit scaler ...")
        main_vars = [vv.split('_d')[0].split('_med')[0] for vv in self.all_needed_vars]
        main_vars = [vv for vv in main_vars if (('mask' not in vv) and ('trunoff' not in vv))]
        main_vars = list(set(main_vars))
        logging.info(f"  main variables to scale: {main_vars}")
        self.scaler_vals = dict.fromkeys(self.all_needed_vars)
        for v in main_vars:
            vars_group = [var for var in self.all_needed_vars if re.fullmatch(rf'{v}(_d-\d+)?', var)]
            logging.info(f"    fit standard scaler on {v} for variable group {vars_group}...")                                
            combined = _ds.sel(variable_names=v).data
            combined_mean = combined.mean().compute().item()
            combined_std = combined.std().compute().item()
            for vg in vars_group:
                ## ATTENTION: USE IF MODEL PREDICTS NON_NEGATIVE OUTPUTS ONLY!!!
                # if v == 'snmel':
                #     logging.info(f"Set mean of snmel to 0, to keep targets non-negative!")
                #     combined_mean = 0.
                self.scaler_vals[vg] = (combined_mean, combined_std)         
        
        logging.info("  resulting mean and std per variable:")
        for v,m in self.scaler_vals.items():
            logging.info(v)
            logging.info(f"  {v}:   mean={m[0]}, std={m[1]}")

        # filepath = os.path.sep.join([self.file_dir, 'std_scaler.npz'])
        # logging.info(f'  Save scaling parameters in: {filepath}.')
        # np.savez(filepath, **self.scaler_vals)



    def load_scaler(self, scaler_path):
        """
        For using pre-computed scaling parameters instead of fitting to training set
        """ 
        logging.info(f"Load scaler from {scaler_path} ...")
        #self.scaler_vals = dict.fromkeys(self.all_needed_vars)
        with np.load(scaler_path) as npz:
            self.scaler_vals = {k: npz[k].copy() for k in npz.files if k in self.all_needed_vars}
        logging.info(f"  loaded mean and std for variables: {self.scaler_vals}")
        

        logging.info("  mean and std per variable:")
        for v in self.all_needed_vars:
            # if v in self.scaler_vals:
            #     logging.info(v)
            #     logging.info(f"  {v}:   mean={self.scaler_vals[v][0]}, std={self.scaler_vals[v][1]}")
            if v not in self.scaler_vals:
                logging.error(f"Variable {v} not found in loaded scaler values! Cannot apply scaling to this variable.")
                raise KeyError(f"Variable {v} not found in loaded scaler values! Cannot apply scaling to this variable.")


    def apply_scaler(self, data):
        if self.scaler_vals is not None:
            logging.debug(f"   scale variables {self.all_needed_vars}")
            for var in self.all_needed_vars:
                logging.debug(f"   scale variable {var}")
                data[:,:,self.idx_map[var]] = (data[:,:,self.idx_map[var]] - self.scaler_vals[var][0]) / self.scaler_vals[var][1]
                minvar = data.min()
                maxvar = data.max()
                logging.debug(f"   resulting range after scaling: min={minvar}, max={maxvar}")
        return data


    def _get_indices(self, sub_list):
        """Get indices of sub_list in self.data_var_names."""
        return [self.data_var_names.index(x) for x in sub_list]


    def _get_variable_indices(self):
        """Get indices of all needed variables in the dataset."""
        self.daily_idx = self._get_indices(self.daily_inputs)
        self.medrange_idx = self._get_indices(self.medrange_inputs)
        self.spinup_idx = self._get_indices(self.spinup_inputs)
        self.auto_id = self._get_indices(self.auto_inputs)
        self.target_idx = self._get_indices(self.target_vars)
        self.trunoff_idx = self._get_indices(self.trunoff_var)
        self.all_needed_idx = self.daily_idx + self.medrange_idx + self.spinup_idx + self.auto_id + self.target_idx + self.trunoff_idx
        self.all_needed_vars = self.daily_inputs + self.medrange_inputs + self.spinup_inputs + self.auto_inputs + self.target_vars + self.trunoff_var
        #self.idx_map = {v: i for i, v in enumerate(self.all_needed_idx)}
        self.idx_map = {k:v for k, v in zip(self.all_needed_vars, self.all_needed_idx)}

    def _lazy_init(self):
        """Lazy initialization of the dataset to avoid issues with multiprocessing."""
        if self.data is None:
            # open the dataset (this happens in the process that calls __getitem__,
            # which is each DataLoader worker process when num_workers>0)
            self.data = xr.open_zarr(self.file_dir, consolidated=True, chunks="auto")
            # register a finalizer so that when this object is GC'd in this process
            # the underlying store will be closed even if __del__ isn't called.
            try:
                # finalize will call self._close_data in the same process
                import weakref
                self._finalizer = weakref.finalize(self, type(self)._close_data, self)
            except Exception:
                # if finalize fails for some reason, that's OK; __del__ remains as fallback
                self._finalizer = None

    def _close_data(self):
        """Internal helper to close the underlying xarray/zarr store in this process."""
        try:
            data = getattr(self, 'data', None)
            if data is not None:
                try:
                    # xarray Dataset/Group close
                    data.close()
                except Exception:
                    pass
        finally:
            # remove reference
            try:
                self.data = None
            except Exception:
                pass

    def close(self):
        """Public explicit close() to deterministically release files/handles in this process."""
        # detach the finalizer so it won't try to run twice
        try:
            if getattr(self, '_finalizer', None) is not None:
                self._finalizer.detach()
        except Exception:
            pass

        # actually close
        try:
            self._close_data()
        except Exception:
            pass

    def __del__(self):
        # keep __del__ only as last-resort safety net
        try:
            self.close()
        except Exception:
            pass



    def __len__(self):
        """Return the length of the dataset."""
        # if self.alternating_indices is not None:
        #     return len(self.alternating_indices)
        return len(self.subset_indices)

    def to_tensor(self, batch_np, var_list):
        indices = [self.idx_map[k] for k in var_list]
        return torch.from_numpy(batch_np[:, :, indices])


    def __getitem__(self, idx):
        """Get item by index."""

        try:
            #print(f"[{os.getpid()}] __getitem__ start for idx={idx}")
            self._lazy_init()
            actual_idx = self.subset_indices[idx]  # get index in full dataset

            if actual_idx+1-self.sequ_len >= 0:
                batch = self.data.isel({self.batch_dim:slice(actual_idx+1-self.sequ_len,actual_idx+1)})
            else:  
                max_len = -actual_idx-1+self.sequ_len
                logging.info(f"Index {actual_idx} is too small for sequence length {self.sequ_len}. Padding oldest time step {max_len} times.")
                batch = self.data.isel({self.batch_dim:slice(0,actual_idx+1)})
                first = batch.isel({self.batch_dim:slice(0,1)})
                pad = xr.concat([first]*(max_len), dim=self.batch_dim)#.expand_dims({self.batch_dim: np.arange(self.sequ_len-max_len-1)})
                batch = xr.concat([pad, batch], dim=self.batch_dim)

                
            doy_batch = torch.from_numpy(batch.coords['dayofyear'].values)

            batch_np = batch.data.values

            batch_np = self.apply_scaler(batch_np)

            daily_input_data = self.to_tensor(batch_np, self.daily_inputs)
            medrange_input_data = self.to_tensor(batch_np, self.medrange_inputs)
            spinup_input_data = self.to_tensor(batch_np, self.spinup_inputs)
            auto_data = self.to_tensor(batch_np, self.auto_inputs)
            target_data = self.to_tensor(batch_np, self.target_vars)
            trunoff_data = self.to_tensor(batch_np, self.trunoff_var)
            #doy_batch = doy_batch.repeat_interleave(daily_input_data.shape[0],)
            doy_batch = doy_batch.unsqueeze(1).expand(-1, daily_input_data.shape[1])
            
            return daily_input_data, medrange_input_data, spinup_input_data, target_data, auto_data, trunoff_data, doy_batch, idx
        
        except Exception as e:
            print(f"[{os.getpid()}] __getitem__ ERROR: {e}")
            raise

def collate_batch(batch_list):
    """Custom collate function to handle larger batch_sizes (multiples of sample chunk size!)."""
    Xd_list, Xm_list, Xs_list, target_list, auto_list, trunoff_list, doy_list, idx_list = zip(*batch_list)
    
    def safe_cat(tensor_list, dim=0):
        # Filter out empty tensors
        non_empty = [t for t in tensor_list if t.numel() > 0]
        if len(non_empty) == 0:
            return torch.empty((sequ_dim, batch_dim, 0))   # return empty tensor with correct shape
        elif len(non_empty) == 1:
            return non_empty[0]
        else:
            return torch.cat(non_empty, dim=dim)
        
    sequ_dim, batch_dim, feat_dim = 0, 0, 0

    
    Xd_stacked = safe_cat(Xd_list, dim=1)

    sequ_dim, batch_dim, feat_dim = Xd_stacked.shape   # daily input is never empty, so use its shape
    Xm_stacked = safe_cat(Xm_list, dim=1)
    Xs_stacked = safe_cat(Xs_list, dim=1)
    target_stacked = safe_cat(target_list, dim=1)
    auto_stacked = safe_cat(auto_list, dim=1)
    trunoff_stacked = safe_cat(trunoff_list, dim=1)
    doy_stacked = safe_cat(doy_list, dim=1)

    if not isinstance(idx_list[0], torch.Tensor):
        idx_stacked = torch.tensor(idx_list)
    else:
        idx_stacked = torch.cat(idx_list)

    return Xd_stacked, Xm_stacked, Xs_stacked, target_stacked, auto_stacked, trunoff_stacked, doy_stacked, idx_stacked

def my_collate_fn(batch):
    """Wrapper for collate_batch (necessary if num_workers>1)."""
    return collate_batch(batch)


def worker_init_fn(worker_id):
    print(f"[{os.getpid()}] Worker {worker_id} initialized")



if __name__ == "__main__":
    # testing

    mp.set_start_method('spawn', force=True)

    #logging.getLogger().setLevel(logging.CRITICAL)
    logging.getLogger().setLevel(logging.INFO)
    
    # test FirnpackCellsDataset
    # base_dir = os.path.sep.join(['..', 'data', 'processed', 'HIRHAM5-ERAInterim', 'v_00'])
    # data_dir = os.path.sep.join([base_dir, 'snmel_reg'])
    logging.info('Test FirnpackCellsDataset')

    yaml_file = f'{project_dir}/modeling/spec_files_crossval/specs_melt_modularNNEBM_mixed2500_new.yml'
    specs = read_yaml.read_yaml_file(yaml_file)

    base_dir = os.path.abspath(os.path.sep.join([project_dir, specs['directories']['base_dir']]))
    data_dir = os.path.sep.join([base_dir, specs['directories']['data_file']])

    zarrset = ZarrDataset(specs)
    
    # try:
    #     zarrset.check_file()
    # except (FileNotFoundError, KeyError) as e:
    #     zarrset.make_file()

    variable_names = zarrset.get_variable_dict()



    
    #my_dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=my_collate_fn, num_workers=0)

    file_dir = zarrset.get_file_dir()
    # temp_data_split = os.path.sep.join([base_dir, specs['directories']['temp_split_file']])
    # train_val_test_indices = json.load(open(temp_data_split))
    # train_dates = train_val_test_indices['train_sub']
    logging.info('Finished setting up dataset...')
    dataset = FirnpackCellsDataset(data_dir+'/data_sub_newchunks.zarr', dates=None, variable_names=variable_names)
    
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=my_collate_fn,
        num_workers=1,
        persistent_workers=True,
        worker_init_fn=AffinityInitializer(base_offset=0, cores_per_worker=1),
        pin_memory=True,
        drop_last=True
    )
    for Xd, Xm, Xy, y, yprev, tr, doy, idx in dataloader:
        print('****', y.size(0), y.size(1), y.size())
        print(f"Xd shape: {Xd.shape}, Xm shape: {Xm.shape}, yprev shape: {yprev.shape}, y shape: {y.shape}, doy: {doy}, idx: {idx}")


    # logging.info('Create dataloader...')
    # my_dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=my_collate_fn, num_workers=0)
    # print(f"Finished creating dataloader.")
    
    # my_iterator = iter(my_dataloader) 
    # time_list = []
    # for i in range(10):
    #     start_time = time.time()
    #     Xd, Xm, Xy, y, yprev, tr, doy = next(my_iterator)
    #     time_list.append(time.time() - start_time)
    #     # logging.info(f'Xd shape: {Xd.shape}, Xm shape: {Xm.shape}, Xy shape: {Xy.shape}, y shape: {y.shape}, yprev shape: {yprev.shape}, trunoff shape: {tr.shape}, doy shape: {doy.shape}')
    #     # logging.info(f'Finished getting one batch in {time.time()-start_time} seconds.')
    # logging.info(f'Average time for one batch: {np.mean(time_list[1:])} seconds.')
    # logging.info(f'Min time for one batch: {np.min(time_list[1:])} seconds.')
    # logging.info(f'Max time for one batch: {np.max(time_list[1:])} seconds.')
    
    