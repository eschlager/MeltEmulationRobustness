# -*- coding: utf-8 -*-
"""
Created on 13.08.2025
@author: eschlager
Class for final dataset preparation for training GRAMMLET and making predictions.
Train and val data sets are prepared using temporal and spatial sub-sampling, data preprocessing as defined in specs file, and scaling.
The resulting datasets are saved as zarr files.
"""

import gc
import json
import logging
import os
import re
import sys

import dask
import dask.array as da
from dask.diagnostics import ProgressBar
import numpy as np
import pandas as pd
from scipy.special import boxcox
import xarray as xr

# import local modules
script_dir = os.path.abspath(os.path.dirname(__file__))
project_dir = os.path.sep.join([script_dir, '..'])
sys.path.append(os.path.sep.join([project_dir, 'src']))
import read_yaml
import logging_config
import help_fcts



DAYS_PER_MEDRANGE = 30  # number of days per medium-range aggregate (e.g. weekly or monthly average)

class ZarrDataset():
    def __init__(self, specs):
        """
        Prepare zarr dataset for training NNfirn based on yaml specs.
        """
        dask.config.set(scheduler='threads')
        self.specs = specs
        self.data_specs = specs['data']
        self.directory_specs = specs['directories']
        self.base_dir = os.path.abspath(os.path.sep.join([project_dir, self.directory_specs['base_dir']]))
        self.file_dir = os.path.sep.join([self.base_dir, self.directory_specs['data_file']])

        self.auto = self.specs['model'].get('auto')
        if not isinstance(self.auto, dict):
            self.auto = {}
        else:
            assert all([t in self.data_specs['target'] for t in self.auto.keys()]), "Auto-regressive variables in daily module must be output variables!"

        self.batch_size = self.specs['training']['batch_size']
        self.processing_chunk_size = 512

        self.targets = self.data_specs['target']
        if isinstance(self.targets,str):
            self.targets = [self.targets]

        if 'trunoff_file' in self.directory_specs:
            self.runoff_dir = os.path.sep.join([self.base_dir, self.directory_specs['trunoff_file']])
        else:
            self.runoff_dir = None

        self.inputs = self.data_specs['input']
        self.variables_all = self.inputs + self.targets

        # slice variable names in different categories
        self.variables_daily = [x for x in self.variables_all if 'yr' not in x and '_med-' not in x]
        self.inputs_daily = [x for x in self.inputs if 'yr' not in x and '_med-' not in x]
        self.inputs_spinup = [x for x in self.inputs if 'yr' in x]
        self.inputs_medrange = [x for x in self.inputs if '_med-' in x]

        # get all variable names based on using previous days and medium-range aggregates
        self.historic_specs = self.add_historic_names()
        
        self.variables_full = []
        variable_dict = self.get_variable_dict()
        logging.info(f"Variables specified:")
        for c,v in variable_dict.items():
            logging.info(f"  {c}: {v}")
            self.variables_full.extend(v)
        
        logging.info(f"  resulting full variable list: {self.variables_full}")
        
        self.variables_contd = [vv for vv in self.variables_all if (('mask' not in vv) and ('trunoff' not in vv))]   # only continuous variables
        self.variables_full_contd = [vv for vv in self.variables_full if (('mask' not in vv) and ('trunoff' not in vv))]   # only continuous variables
        self.ds = None


    def get_variable_dict(self):
        var_dict = {}
        var_dict['daily'] = self.inputs_daily + self.inputs_daily_new
        var_dict['medrange'] = self.inputs_medrange + self.inputs_medrange_new
        var_dict['spinup'] = self.inputs_spinup
        var_dict['auto'] = self.inputs_auto
        var_dict['target'] = self.targets
        if self.runoff_dir:
            var_dict['trunoff'] = ['trunoff']
        return var_dict
    
    def get_file_dir(self):
        return self.file_dir

    def add_historic_names(self):
        logging.info("Get all variable names...")
        historic_specs = []
        
        def get_lags(vars_list, start_day, nr_entities, days_per_entity):
            new_names = []
            for v in vars_list:
                for lag_day in range(start_day, nr_entities * days_per_entity + start_day, days_per_entity):
                    if '_mask' in v:   # if var is a mask, add '_mask'-suffix at the very end of new variable name
                        new_name = f"{v.replace('_mask','')}_d-{lag_day}_mask"
                    else:
                        new_name = f"{v}_d-{lag_day}"
                    historic_specs.append((v, lag_day, new_name))
                    logging.debug(f"  add {new_name} for {v} with lag {lag_day} days")
                    new_names.append(new_name)
            return new_names
        
        # Daily lags
        if self.data_specs['prec_days'] > 0:
            logging.info("Get daily variable names...")
            self.inputs_daily_new = get_lags(
                self.inputs_daily, 
                start_day=1,
                nr_entities=self.data_specs['prec_days'],
                days_per_entity=1
            )
        else:
            self.inputs_daily_new = []
        
        # med-range lags
        self.prec_medrange_units = self.data_specs.get('prec_medrange_units')
        if not isinstance(self.prec_medrange_units, int):
            self.prec_medrange_units = 0
        if self.prec_medrange_units > 0:
                logging.info("Get medium-range variable names...")
                start_day = 1 #+ self.data_specs['prec_days']
                self.inputs_medrange_new = get_lags(
                    self.inputs_medrange,
                    start_day=start_day,
                    nr_entities=self.prec_medrange_units,
                    days_per_entity=DAYS_PER_MEDRANGE
                )
        else:
            self.inputs_medrange_new = []
        
        self.inputs_auto = []
        if len(self.auto) > 0:
            logging.info(f"Add auto-regressive variable names in daily module {self.auto}...")
            for t,n in self.auto.items():
                logging.debug(f"   add previous days of {t}...")
                self.inputs_auto.extend(get_lags([t], start_day=1, nr_entities=n, days_per_entity=1))
  
        return historic_specs


    def add_historic_data(self, _ds):        
        for original_var, lag_day, new_name in self.historic_specs :
            if original_var in _ds:
                logging.info(f"Adding data for {new_name} from {lag_day} days before target date ...")
                _ds[new_name] = _ds[original_var].shift(time=lag_day)
            else:
                logging.error(f"Variable {original_var} not found in dataset, skipping {new_name}")
        return _ds


    def check_file(self, datafile='train_sub.zarr'):
        """Check if file exists and contains all variables."""
        logging.info(f"Check data {datafile} in {self.file_dir}")
        with xr.open_zarr(self.file_dir+'/'+datafile, consolidated=True) as ds:

            for x in self.variables_full:
                if x not in ds.variable_names:
                    logging.error(f"Variable {x} not found in dataset {self.file_dir}!")
                    logging.error(f"Available variables: {ds.variable_names}")
                    raise KeyError(f"Variable {x} not found in dataset {self.file_dir}!")
            
            for x in ds.variable_names:
                if self.runoff_dir:
                    self.variables_full = self.variables_full + ['trunoff']
                if x not in self.variables_full:
                    logging.info(f"Have extra variable {x.item()} in dataset - not used for training!")

            # train_times = np.array(self.train_val_test_indices['train_sub'], dtype='datetime64[ns]') 
            # ds_time_values = self.ds['time'].values
            # included = np.isin(train_times, ds_time_values)
            # if not included.all():
            #     logging.error(f"Not all dates in train indices are included in dataset {self.file_dir}!")
            #     raise KeyError("Not all dates in train indices are included in dataset!")
            
            # val_times = np.array(self.train_val_test_indices['val_sub'], dtype='datetime64[ns]') 
            # ds_time_values = xr.open_zarr(self.file_dir+'/val.zarr', consolidated=True)['time'].values
            # included = np.isin(val_times, ds_time_values)
            # if not included.all():
            #     logging.error(f"Not all dates in val indices are included in dataset {self.file_dir}!")
            #     raise KeyError("Not all dates in val indices are included in dataset!")
            
            # test_times = np.array(self.train_val_test_indices['test'], dtype='datetime64[ns]') 
            # ds_time_values = xr.open_zarr(self.file_dir+'/test.zarr', consolidated=True)['time'].values
            # included = np.isin(test_times, ds_time_values)
            # if not included.all():
            #     logging.error(f"Not all dates in test indices are included in dataset {self.file_dir}!")
            #     raise KeyError("Not all dates in test indices are included in dataset!")


    def get_spatial_indices(self):
        spatial_idx_file = os.path.sep.join([self.file_dir, self.directory_specs['spatial_sub_file']])
        logging.info(f"Load spatially sub-sampling for training and validation data from file {spatial_idx_file}...")
        spatial_idx = xr.open_dataset(os.path.sep.join([spatial_idx_file]))['subsampling']
        logging.info(f"  sub-sample {int(spatial_idx.sum().item())} locations.")
        return spatial_idx

    def make_file(self, use_spatial_subsampling=True):
        """Create zarr file from base dataset in case check_file was unsuccessful."""
        os.makedirs(self.file_dir, exist_ok=True)
        logging_config.define_root_logger(os.path.join(self.file_dir, f'log.txt'))
        logging.getLogger().setLevel(logging.INFO)

        temp_data_split = os.path.sep.join([self.base_dir, self.directory_specs['temp_split_file']])
        logging.info(f'Load temporal split data from file {temp_data_split}...')
        with open(temp_data_split, 'r') as f: 
            train_val_test_indices = json.load(f)
        if use_spatial_subsampling:
            spatial_idx = self.get_spatial_indices()

        logging.info("Create zarr data file...")
        self.read_data()

        # clip variable data if specified in specs
        if 'clipping' in self.data_specs:
            self.clip_vars(self.data_specs['clipping'])

        # mask continuous variables if their associated mask is an input/target too
        for var in self.variables_all:
            if var+'_mask' in self.variables_all:
                logging.info(f'Set values of {var} to nan according to mask {var}_mask:')
                mask = self.ds[var+'_mask']
                self.ds[var] = self.ds[var].where(mask)

        # transform data as specified;
        # consider if transformation is applied to masked data, since transformation on unmasked data could be invalid, 
        # or need different scaling parameters
        if 'transform_inputs' in self.data_specs:
            self.transform_vars(self.data_specs['transform_inputs'])

        if 'transform_targets' in self.data_specs:
            self.transform_vars(self.data_specs['transform_targets'])

        # add all the data for previous days and medium-range aggregates, and the auto-regressive variables
        print(self.prec_medrange_units)
        print(self.data_specs['prec_days'])
        historic_period = 1 + np.max([self.data_specs['prec_days'], self.prec_medrange_units * 7])

        logging.info("Rechunk base dataset ...")
        self.ds = self.ds.chunk({"time": self.processing_chunk_size, "z": self.processing_chunk_size})
        logging.info("  finished rechunking.")
        
        # Subsample train and val data temporally and spatially
        logging.info("Get training data...")
        train_dates = pd.to_datetime(train_val_test_indices['train'])
        train_dates_start = np.min(train_dates) - pd.Timedelta(days=historic_period)
        train_dates_end = np.max(train_dates)
        logging.info(f"  extract training data from {train_dates_start} to {train_dates_end} ...")
        train_set = self.ds.sel(time=slice(train_dates_start, train_dates_end))
        logging.info(f"  apply spatial sub-sampling and drop locations with NaNs ...")
        train_set = train_set.where(spatial_idx, drop=True)#.dropna(dim='z', how='any')
        logging.info("  finished spatial sub-sampling.")
        logging.info(f"  rechunk data for processing ...")
        train_set = train_set.chunk({"time": self.processing_chunk_size, "z": self.processing_chunk_size})
        train_set = self.add_historic_data(train_set)

        val_dates = pd.to_datetime(train_val_test_indices['val'])
        val_dates_start = np.min(val_dates) - pd.Timedelta(days=historic_period)
        val_dates_end = np.max(val_dates)
        logging.info(f"  extract val data from {val_dates_start} to {val_dates_end} ...")
        val_set = self.ds.sel(time=slice(val_dates_start,val_dates_end))
        logging.info(f"  rechunk data for processing ...")
        val_set = val_set.chunk({"time": self.processing_chunk_size, "z": self.processing_chunk_size})
        val_set = self.add_historic_data(val_set)

        logging.info(f"  use val dataset without sub-sampling for testing after training...")
        test_dates = pd.to_datetime(train_val_test_indices['test'])
        test_dates_start = np.min(test_dates) - pd.Timedelta(days=historic_period)
        test_dates_end = np.max(test_dates)
        logging.info(f"  extract test data from {test_dates_start} to {test_dates_end} ...")
        test_set = self.ds.sel(time=slice(test_dates_start,test_dates_end))#.dropna(dim='z', how='any')
        logging.info(f"  rechunk data for processing ...")
        test_set = test_set.chunk({"time": self.processing_chunk_size, "z": self.processing_chunk_size})
        test_set = self.add_historic_data(test_set)
        
        self.fill_vals = {var: None for var in self.variables_full_contd if var + '_mask' in self.variables_full}

        # fit scaler on spatially and temporally sub-sampled training data!
        train_sub_dates = pd.to_datetime(train_val_test_indices['train_sub'])
        train_sub = train_set.sel(time=train_sub_dates)
        self.fit_scaler(train_sub)
        ## alternatively load an existing scaler
        # self.load_scaler(self.file_dir+'/std_scaler.npz')

        logging.info("Process train set:")
        train_set = self.apply_scaling(train_set)
        train_set = self.fill_nans(train_set)
        train_set = train_set.sel(time=train_dates)

        filepath = os.path.sep.join([self.file_dir, 'fill_nans.npz'])
        logging.info(f'  Save filling values: {filepath}.')
        np.savez(filepath, **self.fill_vals)


        train = self.restructure_dataset(train_set)
        train_dir = os.path.join(self.file_dir, 'train_sub.zarr')
        logging.info(f"Prepare saving of train data to {train_dir} ...")
        self.save_dataset(train, train_dir, self.batch_size)
        
        logging.info("Process val set:")
        val_set = self.apply_scaling(val_set)
        val_set = self.fill_nans(val_set)
        val_set = val_set.sel(time=val_dates)
        val = self.restructure_dataset(val_set)
        val_dir = os.path.join(self.file_dir, 'val.zarr')
        logging.info(f"Prepare saving of val data to {val_dir} ...")
        self.save_dataset(val, val_dir, self.batch_size)

        val_sub = val_set.where(spatial_idx, drop=True)
        val_sub = self.restructure_dataset(val_sub)
        val_sub_dir = os.path.join(self.file_dir, 'val_sub.zarr')
        logging.info(f"Prepare saving of val data to {val_sub_dir} ...")
        self.save_dataset(val_sub, val_sub_dir, self.batch_size)

        logging.info("Process test set:")
        test_set = self.apply_scaling(test_set)
        test_set = self.fill_nans(test_set)
        test_set = test_set.sel(time=test_dates)
        test = self.restructure_dataset(test_set)
        test_dir = os.path.join(self.file_dir, 'test.zarr')
        logging.info(f"Save test data to {test_dir} ...")
        nr_locs = len(set(test.z.values))
        self.save_dataset(test, test_dir, nr_locs)
       
        logging.info(f"Successfully saved processed data!")

        self.ds.close() 
        del self.ds 
        gc.collect()



    def prepare_prediction_file(self, dates, daily_file, medrange_file=None, spinup_file=None, scaler_file=None, use_spatial_subsampling=False, savedir='./predictions.zarr'):
        """
        Create zarr from basefile for making prediction from new data
        """
        os.makedirs(self.file_dir, exist_ok=True)
        logging_config.define_root_logger(os.path.join(self.file_dir, f'log.txt'))
        logging.getLogger().setLevel(logging.INFO)

        logging.info("Create zarr data file...")
        
        # add all the data for previous days and medium-range aggregates, and the auto-regressive variables
        historic_period = 1 + np.max([self.data_specs['prec_days'], self.prec_medrange_units * 7])
        dates_all = pd.to_datetime(dates)
        dates_start = np.min(dates_all).replace(hour=0, minute=0, second=0) - pd.Timedelta(days=historic_period)
        dates_end = np.max(dates_all).replace(hour=23, minute=59, second=59)
        time_slice = slice(dates_start, dates_end)

        logging.info(f"  Open daily data from {dates_start} to {dates_end} ...")
        with xr.open_dataset(daily_file) as daily_data:
            daily_data = daily_data[self.variables_daily].sel(time=time_slice).astype(np.float32)
            if 'x' in daily_data.dims and 'y' in daily_data.dims:
                daily_data = daily_data.chunk({"time": 1, "x": -1, "y": -1})
                daily_data = (daily_data
                              .stack(z=('x', 'y')).reset_index('z')
                              .dropna(dim='z', how='all'))
            self.ds = daily_data
        if medrange_file:
            logging.info(f"Add medium-range data from {medrange_file} ...")
            with xr.open_dataset(medrange_file) as medrange_data:
                medrange_data = medrange_data[self.inputs_medrange].sel(time=time_slice)
                if 'x' in medrange_data.dims and 'y' in medrange_data.dims:
                    medrange_data = medrange_data.chunk({"time": 1, "x": -1, "y": -1})
                    medrange_data = (medrange_data.stack(z=('x', 'y')).reset_index('z')
                                     .dropna(dim='z', how='all'))
                medrange_data = medrange_data.reindex(time=self.ds.time, method='nearest', tolerance=pd.Timedelta('1D'))
                self.ds = xr.merge([self.ds, medrange_data])
        #logging.info(f'self.ds time {self.ds.time.values}')
        if spinup_file:
            logging.info(f"Add long-term data from {spinup_file} ...")
            with xr.open_dataset(spinup_file) as spinup_data:
                spinup_data = spinup_data[self.inputs_spinup].sel(time=time_slice)
                #logging.info(f'spinup_data time {spinup_data.time.values}')
                if 'x' in spinup_data.dims and 'y' in spinup_data.dims:
                    spinup_data = spinup_data.chunk({"time": 1, "x": -1, "y": -1})
                    spinup_data = (spinup_data.stack(z=('x', 'y')).reset_index('z')
                                   .dropna(dim='z', how='all'))
                spinup_data = spinup_data.reindex(time=self.ds.time, method='nearest', tolerance=pd.Timedelta('1D'))
                self.ds = xr.merge([self.ds, spinup_data])
        
        logging.info(f'Merged time coordinates: {self.ds.time.values}')
        self.ds = self.ds.assign_coords(dayofyear=("time", self.ds.time.dt.dayofyear.data))         

        # clip variable data if specified in specs
        if 'clipping' in self.data_specs:
            self.clip_vars(self.data_specs['clipping'])

        # mask continuous variables if their associated mask is an input/target too
        for var in self.variables_all:
            if var+'_mask' in self.variables_all:
                logging.info(f'Set values of {var} to nan according to mask {var}_mask:')
                mask = self.ds[var+'_mask']
                self.ds[var] = self.ds[var].where(mask)

        # transform data as specified;
        # consider if transformation is applied to masked data, since transformation on unmasked data could be invalid, 
        # or need different scaling parameters
        if 'transform_inputs' in self.data_specs:
            self.transform_vars(self.data_specs['transform_inputs'])

        if 'transform_targets' in self.data_specs:
            self.transform_vars(self.data_specs['transform_targets'])

        if use_spatial_subsampling:
            spatial_idx = self.get_spatial_indices()
            self.ds = self.ds.where(spatial_idx, drop=True)

        logging.info("Rechunk base dataset ...")
        self.ds = self.ds.chunk({"time": self.processing_chunk_size, "z": self.processing_chunk_size})
        logging.info("  finished rechunking.")
        
        self.ds = self.add_historic_data(self.ds)
        
        # select only needed variables
        self.ds = self.ds[self.variables_full]

        self.fill_vals = {var: None for var in self.variables_full_contd if var + '_mask' in self.variables_full}

        if scaler_file:
            self.load_scaler(scaler_file)
            self.ds = self.apply_scaling(self.ds)

        self.ds = self.restructure_dataset(self.ds)
        
        logging.info(f"Save data to {savedir} ...")
        self.save_dataset(self.ds, savedir, self.batch_size)
    
        daily_data.close()
        self.ds.close() 
        del self.ds 
        gc.collect()


    def read_data(self):
        base_file = os.path.abspath(os.path.sep.join([self.base_dir, 'base_dataset.zarr']))
        ds = xr.open_zarr(base_file)
        if ('sfwater' in self.variables_daily) and ('sfwater' not in ds.data_vars):
            variables_daily = [v for v in self.variables_daily if v != 'sfwater']
            self.ds = ds[variables_daily].astype(np.float32)
            logging.info(f"Select daily variables {variables_daily} from base dataset {base_file}.")
            logging.info(f"Creat sfwater from snmel+rainfall.")
            self.ds['sfwater'] = ds['snmel'].astype(np.float32) + ds['rainfall'].astype(np.float32)
        elif ('energyin' in self.variables_daily) and ('energyin' not in ds.data_vars):
            variables_daily = [v for v in self.variables_daily if v != 'energyin']
            self.ds = ds[variables_daily].astype(np.float32)
            logging.info(f"Select daily variables {variables_daily} from base dataset {base_file}.")
            logging.info(f"Creat energyin from ahfl+ahfs+dlwrad.")
            self.ds['energyin'] = ds['ahfl'].astype(np.float32) + ds['ahfs'].astype(np.float32) + ds['dlwrad'].astype(np.float32)
        else:
            self.ds  = ds[self.variables_daily].astype(np.float32)
            logging.info(f"Select daily variables {self.variables_daily} from base dataset {base_file}.")
        
        if self.inputs_medrange:
            agg_file = os.path.abspath(os.path.sep.join([self.base_dir, 'base_dataset_tempagg.zarr']))
            ds = xr.open_zarr(agg_file)#.shift(time=self.data_specs['prec_days']+1)
            if ('sfwater_med-avg' in self.inputs_medrange) and ('sfwater_med-avg' not in ds.data_vars):
                inputs_medrange = [v for v in self.inputs_medrange if v != 'sfwater_med-avg']
                ds_agg = ds[inputs_medrange].astype(np.float32)
                logging.info(f"Select medium-range aggregated variables {inputs_medrange} from base dataset {agg_file}.")
                logging.info(f"Creat sfwater_med-avg from snmel_med-avg+rainfall_med-avg.")
                ds_agg['sfwater_med-avg'] = ds['snmel_med-avg'].astype(np.float32) + ds['rainfall_med-avg'].astype(np.float32)
            else:
                logging.info(f"Select medium-range aggregated variables {self.inputs_medrange} from base dataset {agg_file}.")
                ds_agg = ds[self.inputs_medrange].astype(np.float32)
            ds_agg = ds_agg.reindex(time=self.ds.time, method='nearest', tolerance=pd.Timedelta('1D'))
            self.ds = xr.merge([self.ds, ds_agg])
        
        if self.inputs_spinup:
            spinup_file = os.path.abspath(os.path.sep.join([self.base_dir, 'base_dataset_10yr.zarr']))
            ds_spinup = xr.open_zarr(spinup_file)[self.inputs_spinup].astype(np.float32)
            ds_spinup = ds_spinup.reindex(time=self.ds.time, method='nearest', tolerance=pd.Timedelta('1D'))
            logging.info(f"Select 10-yr average variables {self.inputs_spinup} from base dataset {spinup_file}.")
            self.ds = xr.merge([self.ds, ds_spinup])


        self.ds = self.ds.assign_coords(dayofyear=("time", self.ds.time.dt.dayofyear.data))

        # add trunoff map if specified
        if self.runoff_dir:
            self.trunoff_map = xr.open_zarr(self.runoff_dir).chunk({'z': self.processing_chunk_size})
            # self.trunoff_map_subsampled = self.trunoff_map.where(spatial_idx)#.dropna(dim='z', how='all')
            logging.info("  add runoff timescale to dataset ...")
            self.ds['trunoff'] = self.trunoff_map['trunoffnorm']
        
        
    def clip_vars(self, dict_clipper):
        for var in dict_clipper:
            if len(dict_clipper[var])==2:
                logging.info(f"  clip variable {var} to range {dict_clipper[var]}.")
                self.ds[var] = self.ds[var].clip(
                    min=dict_clipper[var][0], 
                    max=dict_clipper[var][1])
            else:
                logging.warning(f"  clipping for variable {var} not specified correctly - skip clipping.")


    def transform_vars(self, dict_transf):
        """Transform variables according to specifications in dict_transf.
        Be careful on masked data, since transformation on unmasked data could be invalid
        (e.g. log(0) or log(-x) for x<0), or need different scaling parameters!
        """
        for var, (transf_fct, transf_param) in dict_transf.items():
            logging.info("Perform transformation ...")
            if transf_fct == 'sym_log':
                logging.info(f'  transform {var} using sym_log transformation with a={transf_param}')
                self.ds[var] = xr.apply_ufunc(help_fcts.sym_log, self.ds[var], transf_param, dask='parallelized')
            elif transf_fct == 'log':
                if transf_param==1:
                    logging.info(f'  transform {var} using log1p transformation')
                    self.ds[var] = xr.apply_ufunc(np.log1p, self.ds[var], dask='parallelized')
                elif transf_param==0: 
                    logging.info(f'  transform {var} using log(x)')
                    self.ds[var] = xr.apply_ufunc(np.log, self.ds[var], dask='parallelized')
                elif transf_param>0:
                    logging.info(f'  transform {var} using log(x+{transf_param})')
                    self.ds[var] = xr.apply_ufunc(help_fcts.log_plus_c, self.ds[var], transf_param, dask='parallelized') 
            elif transf_fct == 'box_cox':
                logging.info(f'  transform {var} using box-cox with lambda={transf_param}')
                #self.ds[var] = xr.apply_ufunc(self.box_cox, self.ds[var], transf_param, dask='parallelized')
                self.ds[var] = xr.apply_ufunc(boxcox, self.ds[var], transf_param, input_core_dims=[[], []], dask='parallelized')
            else:
                logging.error(f'  transformation ({transf_fct},{transf_param}) for {var} is invalid!')
    

    def fit_scaler(self, _ds):
        logging.info("Fit scaler ...")
        self.scaler_vals = dict.fromkeys(self.variables_full_contd)
        for v in self.variables_contd:
            vars_group = [var for var in self.variables_full_contd if re.fullmatch(rf'{v}(_d-\d+)?', var)]
            logging.info(f"    fit standard scaler on {v} for variable group {vars_group}...")                                
            combined = _ds[v]
            combined_mean = combined.mean().compute().item()
            combined_std = combined.std().compute().item()
            for vg in vars_group:
                self.scaler_vals[vg] = (combined_mean, combined_std)         
        
        logging.info("  resulting mean and std per variable:")
        for v,m in self.scaler_vals.items():
            logging.info(v)
            logging.info(f"  {v}:   mean={m[0]}, std={m[1]}")

        filepath = os.path.sep.join([self.file_dir, 'std_scaler.npz'])
        logging.info(f'  Save scaling parameters in: {filepath}.')
        np.savez(filepath, **self.scaler_vals)



    def load_scaler(self, scaler_path):
        """
        For using pre-computed scaling parameters instead of fitting to training set
        """ 
        logging.info(f"Load scaler from {scaler_path} ...")
        self.scaler_vals = dict.fromkeys(self.variables_full_contd)
        with np.load(scaler_path) as npz:
            for k in self.variables_contd:
                self.scaler_vals = {k: npz[k].copy() for k in npz.files}

        logging.info("  mean and std per variable:")
        for v,m in self.scaler_vals.items():
            logging.info(v)
            logging.info(f"  {v}:   mean={m[0]}, std={m[1]}")
        
            

    def apply_scaling(self, _ds):
        logging.info("  apply standard scaling on continuous data...")        
        for var in self.variables_full_contd:
            logging.info(f"    scale {var}:")
            _ds[var] = (_ds[var] - self.scaler_vals[var][0]) / self.scaler_vals[var][1]
            minvar = _ds[var].min().compute().item()
            maxvar = _ds[var].max().compute().item()
            logging.info(f"   resulting range after scaling: min={minvar}, max={maxvar}")
        logging.info('  ... finished scaling')
        return _ds


    def fill_nans(self, _ds):
        """ Fill nan values in continuous variables, if they have an associated mask,
        where I have set the masked values to nan beforehand.
        Check min value of transformed data before filling, as define fill_value as minval-0.1.
        """

        for var in self.fill_vals.keys():
            if self.fill_vals[var] is None:
                logging.info(f"  determine fill value for {var} ...")
                # compute min val; data has been masked before, so min is over valid data only.
                minval = _ds[var].min().compute().item()
                fill_value = minval-0.1   # set minval to a smaller value than any valid value.
                logging.info(f"    min value of {var} before filling: {minval}")
                logging.info(f"    use fill value: {fill_value}")
                self.fill_vals[var] = fill_value
            logging.info(f"  Fill transformed and scaled {var} with {self.fill_vals[var]} according to mask.")
            _ds[var] = _ds[var].where(_ds[var+'_mask'], other=self.fill_vals[var])
        return _ds


    def restructure_dataset(self, _ds):
        logging.info('  reshape dataset to dataarray... ')
        _ds = _ds.to_array(dim='variable_names')
        _ds.name = "data"
        _ds = _ds.chunk({"time": 1, "z": -1, "variable_names": -1})
        _ds = _ds.transpose('time', 'z', 'variable_names')
        return _ds
        

    def shuffle_dataset(self, _ds):
        n = _ds.sizes['sample']
        np.random.seed(5)
        logging.info('Shuffle samples ...')
        shuffled_indices = da.from_array(np.random.permutation(n), chunks=n)
        _ds = _ds.isel(sample=shuffled_indices)
        filepath = os.path.sep.join([self.file_dir, 'shuffled_indices.npz'])
        np.savez(filepath, indices=shuffled_indices.compute())
        return _ds

    
    def save_dataset(self, _ds, filename, batch_size):
        _ds = _ds.drop_encoding()
        logging.info(f"  number of time steps: {_ds.sizes['time']}")
        logging.info(f"  number of locations per time chunk: {_ds.sizes['z']}")
        logging.info("  saving ...")
        with ProgressBar():
            _ds.to_zarr(os.path.join(filename), mode='w', consolidated=True, align_chunks=True)





if __name__ == "__main__":

    # Make file based on yaml specs
    yaml_file = './spec_files/test_file.yml'
    specs = read_yaml.read_yaml_file(os.path.sep.join([script_dir, yaml_file]))

    base_dir = os.path.abspath(os.path.sep.join([project_dir, specs['directories']['base_dir']]))
    file_dir = os.path.sep.join([base_dir, specs['directories']['data_file']])
    os.makedirs(file_dir, exist_ok=True)
    logging_config.define_root_logger(os.path.join(file_dir, f'log.txt'))
    
    zarrset = ZarrDataset(specs)
    
    try:
        zarrset.check_file()
        print(zarrset.get_variable_dict())
    except (FileNotFoundError, KeyError) as e:
        zarrset.make_file()
    
    