# -*- coding: utf-8 -*-
"""
Created on 15.08.2024
@author: eschlager
ModelPredictor class: Make predictions with a trained model and save them to a zarr file.
"""

import os
import sys
import torch
import logging
import numpy as np
import xarray as xr
import zarr
import dask.array as da
import gc
import shutil
from scipy.special import inv_boxcox

# import local modules
script_dir = os.path.abspath(os.path.dirname(__file__))
project_dir = os.path.sep.join([script_dir, '..'])
sys.path.append(os.path.sep.join([project_dir , 'modeling', 'models']))
import help_fcts
import read_yaml
import meltNN as NN
from my_utils import shutdown_dataloader

BUFFERSIZE = 366   # buffer size for writing predictions to zarr file

class ModelPredictor():

    def __init__(self, model_dir, var_dict, dataloader, filedir, load='best', device='cpu'):
        '''
        load 'best' or 'latest' model
        '''        
        self.model_dir = model_dir
        self.var_dict = var_dict
        self.dataloader = dataloader
        self.load = load
        self.device = device

        self.dataset_file = filedir
        base, ext = os.path.splitext(filedir)
        self.rawpredictions_file = f"{base}_raw{ext}"  # make intermediate file for raw predictions
        
        self.specs = read_yaml.read_yaml_file(os.path.sep.join([model_dir, 'specs.yml']))
        
        self.targets = var_dict['target']
        self.z_values = dataloader.dataset.z_values
        self.time_values = dataloader.dataset.dates
        self.nr_z = len(set(self.z_values))
        self.nr_time = len(set(self.time_values))
        logging.info(f'Number of time steps: {self.nr_time}')
        self.nr_samples = dataloader.dataset.nr_samples
        logging.info(f'Number of samples in dataset: {self.nr_samples}.')
        batch_size = self.dataloader.batch_size
        self.chunk_size = 100
        if self.chunk_size > self.nr_samples/batch_size:
            self.chunk_size = int(np.ceil((self.nr_samples/batch_size)))
        
        
        self.get_scaler()
        
        self.transform_inputs = False
        if 'transform_inputs' in self.specs['data']:
            if self.specs['data']['transform_inputs']:
                self.transform_inputs = True

        self.transform_targets = False
        if 'transform_targets' in self.specs['data']:
            if self.specs['data']['transform_targets']:
                self.transform_targets = True


    def close(self):
        """Explicitly clean up resources"""
        try:
            if hasattr(self, 'model'):
                del self.model
        except Exception:
            pass
        
        try:
            if hasattr(self, 'dataloader'):
                shutdown_dataloader(self.dataloader)
        except Exception:
            pass
        
        gc.collect()
        torch.cuda.empty_cache()


    def make_predictions_file(self, coords, ref_coords=None, extra_coords=None, inference_mode=False, perturb=None, append_mode=False):
        """
        Main function for making predictions.
        coords: tuple of (x_coords, y_coords) for all z locations
        ref_coords: tuple of (x_ref, y_ref) for whole grid (coords may only be subset)
        extra_coords: tuple of (lon, lat) associated to (x_ref, y_ref)
        inference_mode: if predictions are teacher-forced or autoregressive (for autoregressive model only)
        perturb: name of variable which should be perturbed; perturbation amount is 10 units (hard-coded in _make_raw_predictions)
        append_mode: boolean; whether to create new file with predictions, or append to existing file
        """
        created = False
        raw_group = None

        if os.path.exists(self.rawpredictions_file):
            logging.info(f'Read raw predictions from {self.rawpredictions_file} ...')
            raw_group = zarr.open(self.rawpredictions_file, mode='r')
        else:
            logging.info(f'Create raw predictions and write to {self.rawpredictions_file} ...')
            self._load_model()
            raw_group = self._create_raw_zarr_file(coords, perturb)  # create empty zarr storage
            created = True
            try:
                self._make_raw_predictions(perturb, inference_mode)  # make and store raw predictions
            except Exception:
                try:
                    raw_group = None
                    shutil.rmtree(self.rawpredictions_file)
                except Exception:
                    pass
                raise

        try: # build final dataset from raw predictions
            logging.info(f'Post-process raw predictions ...')
            ds_xr = self._create_dataset(raw_group, perturb)

            if perturb is not None:
                nan_mask = ~ds_xr[self.targets[0]+'_'+perturb+'_pred'].isnull().all(dim=["z"]).compute()
            else:
                nan_mask = ~ds_xr[self.targets[0]+'_pred'].isnull().all(dim=["z"]).compute()
            ds_xr = ds_xr.isel(time=nan_mask)

            ds_xr = ds_xr.set_index(z=['y', 'x']).unstack('z')
            if ref_coords is not None:
                try:
                    ds_xr = ds_xr.reindex({'x': ref_coords[0], 'y': ref_coords[1]})   # missing cells -> NaN
                except Exception as e:
                    logging.info(f'Error reindexing to reference coordinates: {e}')
                    raise ValueError(f'Error reindexing to reference coordinates: {e}')
            
            if extra_coords is not None:
                try:
                    ds_xr = ds_xr.assign_coords(lon=extra_coords[0], lat=extra_coords[1])
                except Exception as e:
                    logging.info(f'Error assigning extra coordinates: {e}')
                    raise ValueError(f'Error reindexing to extra coordinates: {e}')


            logging.info(f'Saving post-processed predictions to {self.dataset_file} ...')
            ds_xr = ds_xr.chunk({'time': self.chunk_size, 'x': -1, 'y': -1})

            if append_mode:
                existing = xr.open_zarr(self.dataset_file)
                new_vars = [
                    v for v in ds_xr.data_vars
                    if v not in existing.data_vars
                ]
                if new_vars:
                    ds_xr[new_vars].to_zarr(
                        store=self.dataset_file,
                        mode="a",
                        consolidated=True)

            else:
                ds_xr.to_zarr(
                    store=self.dataset_file,
                    mode="w",
                    compute=True,
                    consolidated=True,
                    encoding={
                        var: {"chunks": ds_xr[var].data.chunksize,
                              "compressors": [zarr.codecs.BloscCodec(cname="zstd", clevel=5)],
                              }
                        for var in ds_xr.data_vars
                    }
                )
   
        finally:  # clean up and close all open zarr files and stores
            try:
                shutdown_dataloader(self.dataloader)
            except Exception:
                logging.exception("shutdown_dataloader failed")
                
            try:
                if raw_group is not None:
                    for key in list(raw_group['data'].keys()):
                        try:
                            raw_group['data'][key] = None
                        except Exception:
                            pass
                    raw_group = None
            except Exception:
                pass

            try:
                del raw_group
            except Exception:
                pass

            try:
                del ds_xr
            except Exception:
                pass

            if created:
                try:
                    shutil.rmtree(self.rawpredictions_file)
                except Exception:
                    pass

            try:
                if hasattr(self, 'model'):
                    del self.model
            except Exception:
                pass
            
            gc.collect()
            torch.cuda.empty_cache()
      

        
    def _load_model(self):
        logging.info('Init model...')
        self.model = NN.init_model(self.var_dict, self.specs).to(self.device)
        model_name = f'{self.load}_model.pth'
        try:
            PATH = os.path.sep.join([self.model_dir, model_name])
            logging.info(f'Load model {str(PATH)}...')
            checkpoint = torch.load(PATH, weights_only=True, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
        except FileNotFoundError:
            raise FileNotFoundError(f'Cannot load model: {self.load} model not found')
        except Exception as e:
            raise ValueError(f'Cannot load model: {e}')

    
        
    def _create_raw_zarr_file(self, coords, perturb):
        '''
        Create a zarr file to store the raw predictions.
        '''
        zarr_file = zarr.open(self.rawpredictions_file, mode='w')
        zarr_file.create_group('data')
        
        self.real_values = {}
        for var in self.targets:
            self.real_values[var] = zarr_file.create_array(
                f'data/{var}_true', 
                shape=(self.nr_time, self.nr_z),  
                dtype=np.float32,
                chunks=(self.chunk_size, self.nr_z),
                overwrite=True
        )
        self.pred_values = {}
        if perturb is not None:
            for var in self.targets:
                self.pred_values[var] = zarr_file.create_array(
                    f'data/{var}_{perturb}_pred', 
                    shape=(self.nr_time, self.nr_z),  
                    dtype=np.float32,
                    chunks=(self.chunk_size, self.nr_z),
                    overwrite=True
            )        
        else:
            for var in self.targets:
                self.pred_values[var] = zarr_file.create_array(
                    f'data/{var}_pred', 
                    shape=(self.nr_time, self.nr_z),  
                    dtype=np.float32,
                    chunks=(self.chunk_size, self.nr_z),
                    overwrite=True
            )        
        self.x = zarr_file.create_array(
            'data/x', 
            shape=(self.nr_z),  
            dtype=np.int32,
            chunks=(self.nr_z),
            overwrite=True
        )
        self.x[:] = coords[0]
        self.y = zarr_file.create_array(
            'data/y', 
            shape=(self.nr_z),  
            dtype=np.int32,
            chunks=(self.nr_z),
            overwrite=True
        )
        self.y[:] = coords[1]
        
        self.z = zarr_file.create_array(
            'data/z', 
            shape=(self.nr_z),  
            dtype=np.int32,
            chunks=(self.nr_z),
            overwrite=True
        )
        self.z[:] = self.z_values

        self.time = zarr_file.create_array(
            'data/time', 
            shape=(len(self.time_values)),  
            dtype=np.float64,
            chunks=(self.chunk_size),
            overwrite=True
        )
        self.time[:] = np.array(self.time_values, dtype='datetime64[ns]')

        return zarr_file

    
    def _flush_buffer(self, start_idx):
        if self.buffer_count > 0:            
            for j,var in enumerate(self.targets):
                self.real_values[var][start_idx : start_idx + self.buffer_count,:] = self.buffer_real[:self.buffer_count,:,j]
                self.pred_values[var][start_idx : start_idx + self.buffer_count,:] = self.buffer_pred[:self.buffer_count,:,j]


    def _to_device(self, *args):
        return [x.to(self.device) for x in args]

        
    def _make_raw_predictions(self, perturb, inference_mode=False):
        '''
        Make predictions with the model and save them to the zarr file
        '''

        x_daily,_,_,_,y_prev,_,_,_ = next(iter(self.dataloader))

        
        if y_prev.numel() != 0:   # if val_yprev is not empty, i.e., if model is autoregressive
            use_prev = True
            if inference_mode:
                logging.info('Inference mode: Make autoregressive predictions using previous predictions as input...')
            else:
                logging.info('Teacher-forced mode: Make autoregressive predictions using true previous values as input...')
        else:
            use_prev = False
            inference_mode = False
            logging.info('Model is not autoregressive...')



        if perturb is not None:      
            watt_perturb = 10    # Perturbation by 10 W/m2 for energy fluxes; however any input can be perturbed!
            logging.info(f'Perturb {perturb} by {watt_perturb} W/m2.')
            perturb_idx = self.dataloader.dataset.daily_inputs.index(perturb)
            watt_perturb_scaled = watt_perturb / self.scaler[perturb][1]

        total_batches = len(self.dataloader)
        self.buffer_pred = np.empty((BUFFERSIZE, self.nr_z, len(self.targets)), dtype=np.float32)
        self.buffer_real = np.empty((BUFFERSIZE, self.nr_z, len(self.targets)), dtype=np.float32)
        self.buffer_count = 0
        start_index = 0
        
        with torch.inference_mode():
            for i, (x_daily, x_medrange, x_spinup, y, y_prev, trunoff_map, doy, idx) in enumerate(self.dataloader, start=1):
                batchsize = int(x_daily.size(1)/self.nr_z)
                # if next batch would exceed the time chunk size, flush the buffer to zarr file
                if self.buffer_count + batchsize > BUFFERSIZE:
                    logging.info(f"  Flush buffer to zarr file at index {start_index} with {self.buffer_count} entries.")
                    self._flush_buffer(start_index)
                    start_index += self.buffer_count
                    self.buffer_count = 0

                logging.info(f"  processing batch {i}/{total_batches}")           
                        
                x_daily, x_medrange, x_spinup, y, y_prev, trunoff_map, = self._to_device(
                                    x_daily, x_medrange, x_spinup, y, y_prev, trunoff_map)
                
                x_daily = x_daily[-1,:,:]
                x_medrange = x_medrange[-1,:,:]
                x_spinup = x_spinup[-1,:,:]
                y_prev = y_prev[-1,:,:]
                trunoff_map = trunoff_map[-1,:,:]
                doy = doy[-1,:]
                y = y[-1,:,:]

                if perturb is not None:
                    x_daily[:, perturb_idx] += watt_perturb_scaled
            
                if inference_mode: # full inference mode
                    if i == 1: 
                        # start with zero as initial previous values --> recommended to start predicting timeseries in winter!
                        y_prev = torch.zeros_like(y_prev)
                        y_prev_old = y_prev.clone()
                    else:
                        for var in self.var_dict['target']:
                            if 'mask' in var:   # need to take extra care of mask variable names...
                                var_base = var.split('_')[0]
                            else:
                                var_base = var
                            if self.specs['model']['auto'][var] > 1:
                                # shift previous predictions by one
                                for j in range(self.specs['model']['auto'][var], 1, -1):
                                    var_new = f'{var_base}_d-{j}'   # e.g. 'snmel_d-2'
                                    var_old = f'{var_base}_d-{j-1}'   # e.g. 'snmel_d-1'
                                    if 'mask' in var:
                                        var_new += '_mask'
                                        var_old += '_mask'
                                    logging.info(f'Shift previous prediction {var_old} by one day to {var_new}.')
                                    idx_new = self.dataloader.dataset.auto_inputs.index(var_new)
                                    idx_old = self.dataloader.dataset.auto_inputs.index(var_old)
                                    logging.info(f'  shift {idx_old} to {idx_new}...')
                                    y_prev[:, idx_new] = y_prev_old[:, idx_old]

                            # add recent output as latest history
                            var_d1 = f'{var_base}_d-1'
                            if 'mask' in var:
                                var_d1 += '_mask'
                            output_idx = self.dataloader.dataset.target_vars.index(var)
                            prev_idx = self.dataloader.dataset.auto_inputs.index(var_d1)
                            logging.info(f'Update previous {var_d1} ({output_idx}) with the latest prediction of {var} ({prev_idx}).')
                            logging.info(f'  write previous output to {prev_idx}...')
                            y_prev[:, prev_idx] = output[:, output_idx]
                            y_prev_old = y_prev.clone()  # save for next iteration


                output = self.model.inference(x_daily, x_medrange, x_spinup, y_prev, trunoff_map, doy)

                self.buffer_real[self.buffer_count:self.buffer_count+batchsize,:,:] = y.cpu().numpy().reshape(batchsize, self.nr_z, len(self.targets))
                self.buffer_pred[self.buffer_count:self.buffer_count+batchsize,:,:] = output.cpu().numpy().reshape(batchsize, self.nr_z, len(self.targets))
                self.buffer_count += batchsize
            
            if self.buffer_count > 0:
                logging.info(f"  Flush buffer to zarr file at index {start_index} with {self.buffer_count} entries.")
                self._flush_buffer(start_index)
            
        logging.info(f"Finished predicting.")


    @staticmethod
    def reverse_transform(data, transf_fct, transf_param):
        if transf_fct == 'box_cox':
            data = xr.apply_ufunc(inv_boxcox, data, transf_param, input_core_dims=[[], []], dask='parallelized')
        elif transf_fct == 'log':
            if transf_param==1:
                logging.info('  ... using expm1 as inverse of log1p transformation...')
                data = xr.apply_ufunc(np.expm1, data, input_core_dims=[[]], dask='parallelized')
            elif transf_param==0:
                logging.info('  ... using exp as inverse of log transformation...')
                data = xr.apply_ufunc(np.exp, data, input_core_dims=[[]], dask='parallelized')
            elif transf_param>0:
                logging.info(f'  ... using exp and subtracting {transf_param} as inverse of log1p transformation')
                data = xr.apply_ufunc(help_fcts.inv_log_plus_c, data, transf_param, input_core_dims=[[], []], dask='parallelized')
        return data


    def _create_dataset(self, ds, perturb):
        '''
        Create an xarray dataset from the raw predictions stored in the zarr file.
        Reverse scaling and transformation of the input and target variables, and add dataset metadata.
        
        '''

        true_vars = {}
        pred_vars = {}

        for target in self.targets:
            if perturb is not None:
                target_pred = target + '_' + perturb
            else:
                target_pred = target
            true = xr.DataArray(da.from_zarr(ds[f'data/{target}_true']))
            if target in self.scaler:
                logging.info(f'Scale target variable {target} back to original range...')
                scale_mean, scale_std = self.scaler[target]
                # Dask-backed array from Zarr
                # mean_val = true.mean().compute().item()
                # std_val = true.std().compute().item()
                # logging.info(f'Mean value of true variable {target}: {mean_val}, std value: {std_val}')
                
                logging.info(f'Reverse scaling for target variable {target} with mean {scale_mean} and std {scale_std}')
                true = true * scale_std + scale_mean
                # mean_val = true.mean().compute().item()
                # std_val = true.std().compute().item()
                # logging.info(f'Mean value of true variable {target} after reverse scaling: {mean_val}, std value: {std_val}')
            if self.transform_targets:
                if target in self.specs['data']['transform_targets']:
                    transf_fct, transf_param = self.specs['data']['transform_targets'][target]
                    logging.info(f'Inverse transform target variable {target} with inverse {transf_fct} with parameter {transf_param}...')
                    true = self.reverse_transform(true, transf_fct, transf_param)
                    true = true.fillna(0)  # if have a transformation that works on positive data only, set invalid nans to 0!
            
            true_vars[f'{target}_true'] = (['time', 'z'], da.maximum(true.data, 0))
            

            pred = xr.DataArray(da.from_zarr(ds[f'data/{target_pred}_pred']))
            # mean_val = pred.mean().compute().item()
            # std_val = pred.std().compute().item()
            # logging.info(f'Mean value of pred variable {target}: {mean_val}, std value: {std_val}')
            if target in self.scaler:
                logging.info(f'Scale predicted variable {target} back to original range...')
                pred = pred * scale_std + scale_mean
                # mean_val = pred.mean().compute().item()
                # std_val = pred.std().compute().item()
                # logging.info(f'Mean value of pred variable {target} after reverse scaling: {mean_val}, std value: {std_val}')
            if self.transform_targets:
                if target in self.specs['data']['transform_targets']:
                    transf_fct, transf_param = self.specs['data']['transform_targets'][target]
                    logging.info(f'Inverse transform target variable {target} with inverse {transf_fct} with parameter {transf_param}...')
                    pred = self.reverse_transform(pred, transf_fct, transf_param)
                    pred = pred.fillna(0)
            pred_vars[f'{target_pred}_pred'] = (['time', 'z'], da.maximum(pred.data, 0))

        ds_xr = xr.Dataset(
            {
                **true_vars,
                **pred_vars
            },
            coords={
                'time': ('time', ds['data/time'][:].astype('datetime64[ns]')),
                'x': ('z', ds['data/x'][:]),
                'y': ('z', ds['data/y'][:]),
            }
        )       


        x_values = ds_xr['x']
        y_values = ds_xr['y']
        x_coords = xr.DataArray(x_values, dims=['z'])
        y_coords = xr.DataArray(y_values, dims=['z'])

        ds_xr = ds_xr.assign_coords(x=x_coords, y=y_coords)

        return ds_xr


    def get_scaler(self):
        scaler_paths = [
            os.path.sep.join([self.model_dir, 'std_scaler.npz']),
            os.path.sep.join([self.model_dir, '..', 'std_scaler.npz']),
            os.path.sep.join([project_dir, self.specs['directories']['base_dir'], self.specs['directories']['data_file'], 'std_scaler.npz']),
            ]

        self.scaler = {}
        for self.scaler_path in scaler_paths:
            try:
                with np.load(self.scaler_path) as npz:
                    self.scaler = {k: npz[k].copy() for k in npz.files}
                    logging.info(f"Load scaler file {self.scaler_path}")
                break  # Success, stop trying
            except FileNotFoundError:
                continue

        if not self.scaler:
            logging.warning("No scaler file found.")


