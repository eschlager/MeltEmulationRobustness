# -*- coding: utf-8 -*-
"""
Created on 05.03.2024
@author: eschlager
Script for creating base dataset from netcdf files;
this base dataset is then used to create training specific datasets for train_GRAMMLET.py:
prepare_trainset.py with class ZarrDataset which is called at beginning of train_GRAMMLET.py
"""
#%%
import numpy as np
import os
import sys
import xarray as xr
import logging
import dask
from dask.diagnostics import ProgressBar

script_dir = os.path.abspath(os.path.dirname(__file__))
project_dir = os.path.sep.join([script_dir, '..', '..'])
sys.path.append(os.path.sep.join([project_dir , 'src']))
import logging_config
import help_fcts



if __name__ == "__main__":
    dask.config.set(scheduler='threads')

    out_path = os.path.sep.join([project_dir, 'data', 'processed', 'HIRHAM5-CESM2', 'v_04'])
    os.makedirs(out_path, exist_ok=True)
    logging_config.define_root_logger(os.path.join(out_path, f'log.txt'))
    
    # interim_path = os.path.sep.join([project_dir, 'data', 'interim', 'ERAI', 'HIRHAM5'])
    interim_path = os.path.sep.join(['/dmidata', 'users', 'elkesc', 'data', 'interim', 'CESM2_SSP5-85', 'HIRHAM5'])
    in_path_aux = os.path.sep.join([interim_path, 'AuxFiles'])
    in_path_h5 = os.path.sep.join([interim_path, 'firnpack'])
    
    # %% --------- Create file with original coordinates for later reconstruction ---------
    meta = xr.Dataset()
    src = xr.open_dataset(os.path.sep.join([in_path_aux, 'GRLmask.nc']))
    # include x and y dims/coords (common names)
    if "x" in src.coords:
        meta = meta.assign_coords(x=("x", src["x"].values))
    elif "x" in src.variables:
        # if x is just a variable (not a coord), include it as a variable
        meta["x"] = ("x", src["x"].values)

    if "y" in src.coords:
        meta = meta.assign_coords(y=("y", src["y"].values))
    elif "y" in src.variables:
        meta["y"] = ("y", src["y"].values)

    # lat/lon may be 1D or 2D; include them exactly as in source
    if "lat" in src:
        # ensure values are concrete (compute if dask)
        lat_vals = src["lat"].values
        meta["lat"] = (src["lat"].dims, lat_vals)

    if "lon" in src:
        lon_vals = src["lon"].values
        meta["lon"] = (src["lon"].dims, lon_vals)

    print(meta)
    meta.attrs = src.attrs
    filepath = os.path.sep.join([out_path, 'coords_only.zarr'])
    logging.info(f"GRL mask saved to {filepath}")
    meta.to_zarr(filepath, mode="w", consolidated=True)


    # # %% -------------------- Create stacked nc files for AuxFiles --------------------
    # # -------------------- Create file with runoff timescale --------------------
    # filepath = os.path.sep.join([out_path, 'trunoff.zarr'])
    # ds_basins = xr.open_dataset(os.path.sep.join([in_path_aux, 'runoff_timescale.nc']))
    # ds = ds_basins['trunoffnorm']
    # ds = ds.stack(z=('x', 'y')).reset_index('z')  # flatten spatial dimension
    # ds = ds.dropna(dim='z', how='all')  # drop all NaN values
    # ds.to_zarr(filepath, mode="w", consolidated=True)
    # logging.info(f"runoff timescale saved to {filepath}")
    # ds.close()

    # # -------------------- Create file with runoff timescale --------------------
    # filepath = os.path.sep.join([out_path, 'runoff_dailyrates.zarr'])
    # ds = xr.open_dataset(os.path.sep.join([out_path, 'runoff_weighting.nc']))
    # for lag_val in ds['lag'].values:
    #     var_name = f'runoffrate_{24-lag_val}'  # or f'data_lag_{lag_val}'
    #     ds[var_name] = ds['weights'].sel(lag=lag_val)
    # ds = ds.drop_vars('weights')
    # ds = ds.stack(z=('x', 'y')).reset_index('z')  # flatten spatial dimension
    # ds = ds.dropna(dim='z', how='all')  # drop all NaN values
    # ds.to_zarr(filepath, mode="w", consolidated=True)
    # logging.info(f"runoff timescale saved to {filepath}")
    # ds.close()

    # # -------------------- Create file with GRL mask --------------------
    # filepath = os.path.sep.join([out_path, 'GRLmask.zarr'])
    # ds_basins = xr.open_dataset(os.path.sep.join([in_path_aux, 'GRLmask.nc']))
    # ds = ds_basins[['glacGRL', 'maskbas']]
    # ds = ds.stack(z=('x', 'y')).reset_index('z')  # flatten spatial dimension
    # ds = ds.dropna(dim='z', how='all')  # drop all NaN values
    # ds.to_zarr(filepath, mode="w", consolidated=True)
    # logging.info(f"GRL mask saved to {filepath}")
    # ds.close()


    # -------------------- Create file with zones --------------------
    filepath = os.path.sep.join([out_path, 'GRLzones.zarr'])   # output file from create_spatial_subsampling.py
    ds_zones = xr.open_dataset(os.path.sep.join([in_path_aux, 'GRLzones.nc']))
    ds = ds_zones['zones']
    ds = ds.stack(z=('x', 'y')).reset_index('z')  # flatten spatial dimension
    ds = ds.dropna(dim='z', how='all')  # drop all NaN values
    ds.to_zarr(filepath, mode="w", consolidated=True)
    logging.info(f"GRL mask saved to {filepath}")
    ds_zones.close()

    # # -------------------- Create file for spatial subsampling --------------------
    # file_name = 'GRL_subsampleidx_basinSE'    # output file from create_spatial_subsampling.py
    # filepath = os.path.sep.join([out_path, f'{file_name}_flattened.zarr'])
    # ds_t = xr.open_dataset(os.path.sep.join([in_path_aux, f'{file_name}.nc']))  
    # ds = ds_t['subsampling'].astype(np.float32)
    # ds_mask = xr.open_dataset(os.path.sep.join([in_path_aux, 'GRLmask.nc']))['glacGRL']
    # ds = ds.where(ds_mask==1)
    # ds = ds.stack(z=('x', 'y')).reset_index('z')  # flatten spatial dimension
    # ds = ds.dropna(dim='z', how='all')  # drop all NaN values
    # ds = ds.chunk({"z": -1})
    # #ds.to_zarr(filepath, mode="w", consolidated=True)
    # ds.to_zarr(filepath)
    # logging.info(f"Flattened spatial sub-sampling indices saved to {filepath}")
    # ds_t.close()
    # ds.close()

    # -------------------- Create file for spatial subsampling --------------------
    file_name = 'GRL_subsampleidx_5000'    # output file from create_spatial_subsampling.py
    filepath = os.path.sep.join([out_path, f'{file_name}_flattened.zarr'])
    ds_t = xr.open_dataset(os.path.sep.join([in_path_aux, f'{file_name}.nc']))  
    ds = ds_t['subsampling'].astype(np.float32)
    ds_mask = xr.open_dataset(os.path.sep.join([in_path_aux, 'GRLmask.nc']))['glacGRL']
    ds = ds.where(ds_mask==1)
    ds = ds.stack(z=('x', 'y')).reset_index('z')  # flatten spatial dimension
    ds = ds.dropna(dim='z', how='all')  # drop all NaN values
    ds = ds.chunk({"z": -1})
    #ds.to_zarr(filepath, mode="w", consolidated=True)
    ds.to_zarr(filepath)
    logging.info(f"Flattened spatial sub-sampling indices saved to {filepath}")
    ds_t.close()
    ds.close()
    

    # %% Correct daily files: clip to valid ranges, correct runoff values using available water estimates

    # logging.info(f"Load all daily files ...")
    # years = range(1980, 2017)
    # dropvars = ['evspsbl', 'gld', 'tsl2', 'sradsm', 'tradsm', 'rogl', 'tsl', 'rfrz', 'supimp', 'grndhflx', 'sn']
    # chunks_init = {"time": 1024, "y": 64, "x": 64}
    # ds_all = xr.open_mfdataset(
    #     [f"{in_path_h5}/Daily2D_GRL_{year}.nc" for year in years],
    #     concat_dim="time",
    #     combine="nested",
    #     chunks=chunks_init
    # ).drop_vars(dropvars)
    # ds_all = ds_all.sortby('time')

    # logging.info(f"Correct runoff data ...")
    # file_runoff_weights = os.path.sep.join([out_path, f'runoff_weighting.nc'])
    # try: 
    #     runoff_decay_weights = xr.open_dataset(file_runoff_weights)['weights']
    #     logging.info(f"  loaded runoff lag weights ...")
    # except FileNotFoundError:
    #     logging.info(f"  create runoff lag weights ...")
    #     ds_tau = xr.open_dataset(os.path.sep.join([in_path_aux, 'runoff_timescale.nc']))['trunoffdays']
    #     runoff_decay_weights = help_fcts.runoff_decay_weights(ds_tau, window=25)
    #     runoff_decay_weights.to_netcdf(file_runoff_weights)
    #     logging.info(f"    created runoff lag weights.")

    # # compute available water as 110% of the estimated upper bound of the last 3 days
    # logging.info(f"  compute upper bound for available water ...")
    # available = 1.1 * help_fcts.available_water_n_days(ds_all['snmel'], ds_all['rainfall'], weights=runoff_decay_weights, window=25, n_days=3)
    # available = available.chunk(chunks_init)
    # logging.info(f"  correct runoff data using upper bound for available water ...")
    # ## for timeindex < window the available water will be nan, so no correction will be applied
    # correction_mask = (ds_all['rogl'] > available) & (ds_all['rogl'] > 100)
    # ds_all['rogl'] = ds_all['rogl'].where(~correction_mask, other=available)
    # n_corrected = correction_mask.sum().compute().item()
    # logging.info(f"  number of runoff values corrected: {n_corrected}")


    # logging.info(f"Correct data ranges ...")
    # vars_ranges = {'ahfs': (-140,None),
    #                 }

    # for var, (vmin,vmax) in vars_ranges.items():
    #     if var in ds_all.data_vars:
    #         data = ds_all[var]
    #         logging.info(f" check if {var} in [{vmin}, {vmax}]...")
    #         clipped_at_min = (data < vmin) if vmin is not None else xr.zeros_like(data, dtype=bool)
    #         clipped_at_max = (data > vmax) if vmax is not None else xr.zeros_like(data, dtype=bool)
    #         n_clipped_min = clipped_at_min.sum().compute().item()
    #         n_clipped_max = clipped_at_max.sum().compute().item()

    #         if n_clipped_min > 0:
    #             logging.info(f"   # samples clipped at min: {n_clipped_min}")
    #         if n_clipped_max > 0:
    #             logging.info(f"   # samples clipped at max: {n_clipped_max}")
    #         ds_all[var] = data.clip(min=vmin, max=vmax)

    #     else:
    #         logging.info(f"{var}: not found in dataset")

    # # ds_all['sfwater'] = ds_all['rainfall'] + ds_all['snmel']
    # ds_all['energyin'] = ds_all['dlwrad'] + ds_all['ahfs'] + ds_all['ahfl']

    # # %% -------------------- Restructure dataset, and save daily data to zarr files --------------------
    # filepath = os.path.sep.join([out_path, 'base_dataset.zarr'])
    # chunksize = {"time": 2048, "z": -1}
    # logging.info("Re-structure dataset...")
    # # ds = ds.to_dataarray(dim='variable_names') # stack all variables into an extra dimension
    # # ds.name = "data"
    # logging.info("Flatten spatial dimension...")
    # ds = ds_all.stack(z=('x', 'y')).reset_index('z')  # flatten spatial dimension
    # logging.info("Drop nans...")
    # # ds = ds.dropna(dim='z', how='all')  # drop all NaN values
    # nan_mask = ~ds['tas'].isnull().all(dim=["time"]).compute()   # compute null only from tas, because either all vars or none are nan for a specific location!
    # logging.info(f'Shape of nan-maks: {nan_mask.shape}')
    # logging.info(f'amount of all z: {len(ds.z)}')
    # ds = ds.isel(z=nan_mask)
    # logging.info(f'amount of non-nan z: {len(ds.z)}')

    # # compute binary masks based on thresholds:
    # binary_variables = {'rainfall': 1., 'snfall': 1., 'snmel': 1., 'rogl': 1.}
    # for v,t in binary_variables.items():
    #     logging.info(f'Create binary mask for {v} using threshold {t}...')
    #     mask = (ds[v] > t).astype(bool)  # 1 where condition is True, 0 otherwise
    #     ds[f'{v}_mask'] = mask

    # ds_sel = ds.sel(time=slice("1989-01-01", None))
    # logging.info("Rechunk time dimension...")
    # ds_sel = ds_sel.chunk(chunksize)

    # logging.info(f"Saving data to {filepath}...")

    # with ProgressBar():
    #     ds_sel.to_zarr(filepath, mode="w", consolidated=True)


    # #%% -------------------- Create 10yr rolling averages --------------------
    # outpath_roll = filepath.replace('.zarr', '_10yr.zarr')
    # vars_to_roll = ['tas', 'rainfall', 'snfall', 'energyin', 'dswrad']

    # ds_sel = ds[vars_to_roll]

    # logging.info(f'Create 10yrs rolling mean for {vars_to_roll}...)')
    # window_size = 3652
    # rolling_10yr = ds_sel.rolling(time=window_size, min_periods=window_size)

    # logging.info(f'  drop nans ...')
    # means_10yr = rolling_10yr.mean()
    # means_10yr = means_10yr.sel(time=slice('1990-01-01', None))

    # means_10yr = means_10yr.rename({var: f"{var}_10yr-avg" for var in means_10yr.data_vars})

    # print(f"Saving 10years rolling mean results to {outpath_roll} ...")
    # means_10yr = means_10yr.chunk(chunksize)
    # with ProgressBar():
    #     means_10yr.to_zarr(outpath_roll, mode='w', consolidated=True)

    # #range of 10yrs for tas
    # ds_sel = ds['tas'].sel(time=slice("1988-12-01", None))
    # window_size = 3652
    # rolling = ds_sel.rolling(time=window_size, min_periods=window_size)
    # logging.info(f'Create 10yr rolling range for tas...')
    # min_ = rolling.min()
    # max_ = rolling.max()
    # range_ = max_ - min_

    # logging.info(f'  drop nans ...')
    # range_ = range_.sel(time=slice("1990-01-01", None))
    # range_.name = f"tas_10yr-range"

    # print(f"Saving 10yr ranges to {outpath_roll} ...")
    # range_ = range_.chunk(chunksize)
    # with ProgressBar():
    #     range_.to_zarr(outpath_roll, mode='a', consolidated=True)
 
      

    # # %% -------------------- Create 1yr rolling averages --------------------
    # filepath = os.path.sep.join([out_path, 'base_dataset.zarr'])
    # outpath_roll = filepath.replace('.zarr', '_10yr.zarr')
    # vars_to_roll = ['tas', 'rainfall', 'snfall']
    # ds = xr.open_zarr(filepath, chunks={'time': 1024})
    # ds_sel = ds[vars_to_roll]

    # logging.info(f'Create 1yr rolling mean for {vars_to_roll}...)')
    # window_size = 365
    # rolling_10yr = ds_sel.rolling(time=window_size, min_periods=window_size)

    # logging.info(f'  drop nans ...')
    # means_10yr = rolling_10yr.mean()
    # means_10yr = means_10yr.sel(time=slice('1990-01-01', None))

    # means_10yr = means_10yr.rename({var: f"{var}_1yr-avg" for var in means_10yr.data_vars})

    # print(f"Saving 1years rolling mean results to {outpath_roll} ...")
    # means_10yr = means_10yr.chunk( {"time": 2048, "z": -1})
    # with ProgressBar():
    #     means_10yr.to_zarr(outpath_roll, mode='a', consolidated=True)


    # #%% -------------------- Create medium-range rolling averages --------------------
    # filepath = os.path.sep.join([out_path, 'base_dataset.zarr'])
    # chunksize = {"time": 2048, "z": -1}
    
    # ds = xr.open_zarr(filepath, chunks={'time': 1024})

    # outpath_roll = filepath.replace('.zarr', '_tempagg.zarr')
    # vars_to_roll = ['tas', 'rainfall', 'snfall', 'ahfs', 'ahfl', 'dlwrad', 'dswrad', 'snmel']
    # ds_sel = ds[vars_to_roll].sel(time=slice("1988-12-01", None))

    # logging.info(f'Create medium-range rolling mean for {vars_to_roll}...)')
    # window_size = 7
    # rolling_med = ds_sel.rolling(time=window_size, min_periods=window_size).mean()

    # logging.info(f'  drop nans ...')
    # rolling_med = rolling_med.sel(time=slice("1989-01-01", None))

    # rolling_med = rolling_med.rename({var: f"{var}_med{window_size}-avg" for var in rolling_med.data_vars})
    # rolling_med = rolling_med.chunk(chunksize)
    # print(f"Saving rolling mean over past {window_size} days to {outpath_roll} ...")
    # with ProgressBar():
    #     rolling_med.to_zarr(outpath_roll, mode='w', consolidated=True)
    
    # # # %% -------------------- Create medium-range rolling min, max and range for tas --------------------
    # # vars_to_roll = ['tas']
    # # for v in vars_to_roll:
    # #     ds_sel = ds[v].sel(time=slice("1988-12-01", None))
    # #     logging.info(f'Create medium-range rolling min, max, and range for {v}...')
    # #     window_size = 7
    # #     rolling = ds_sel.rolling(time=window_size, min_periods=window_size)

    # #     min_ = rolling.min()
    # #     max_ = rolling.max()
    # #     range_ = max_ - min_

    # #     logging.info(f'  drop nans ...')
    # #     min_ = min_.sel(time=slice("1989-01-01", None))
    # #     max_ = max_.sel(time=slice("1989-01-01", None))
    # #     range_ = range_.sel(time=slice("1989-01-01", None))

    # #     min_.name = f"{v}_med-min"
    # #     max_.name = f"{v}_med-max"
    # #     range_.name = f"{v}_med-range"

    # #     rolling_ds = xr.merge([min_, max_, range_])

    # #     print(f"Saving rolling min, max and range for {window_size} days of {v} to {outpath_roll} ...")
    # #     rolling_ds = rolling_ds.chunk(chunksize)
    # #     with ProgressBar():
    # #         rolling_ds.to_zarr(outpath_roll, mode='a', consolidated=True)
        



    # #%% -------------------- Create 10yr averages --------------------
    # filepath = os.path.sep.join([out_path, 'base_dataset.zarr'])
    
    # outpath_roll = filepath.replace('.zarr', '_decade.zarr')
    # vars_to_roll = ['tas', 'rainfall', 'snfall', 'sfwater', 'energyin', 'dswrad']
    # ds_sel = ds[vars_to_roll]
    # logging.info(f'Create 10yrs rolling windows for {vars_to_roll}...)')

    # logging.info('  resampling to annual means...')
    # annual = ds_sel.resample(time='YE').mean()
    
    # logging.info('  computing trailing 10-year rolling mean...')
    # window_size = 10

    # with ProgressBar():
    #     means_10yr = annual.rolling(
    #         time=window_size, 
    #         min_periods=window_size,  # Requires full 10 years
    #         center=False  # Trailing window (past + current)
    #     ).mean()

    # logging.info(f'Time stamps {means_10yr.time}')

    # # Relabel as January 1st of next year
    # import pandas as pd
    # means_10yr['time'] = means_10yr['time'] + np.timedelta64(1, 'D')

    # means_10yr = means_10yr.sel(time=slice('1990-01-01', None))
    # logging.info(f'   new time stamps {means_10yr.time}')

    # means_10yr = means_10yr.rename({var: f"{var}-avg" for var in means_10yr.data_vars})

    # logging.info(f"Saving 10years rolling mean results to {outpath_roll} ...")
    # chunksize = {"time": -1, "z": -1}
    # means_10yr = means_10yr.chunk(chunksize)
    # with ProgressBar():
    #     means_10yr.astype(np.float32).to_zarr(outpath_roll, mode='w', consolidated=True)


    # #%% -------------------- Create quartal averages of previous year  --------------------
    # filepath = os.path.sep.join([out_path, 'base_dataset.zarr'])
    # ds = xr.open_zarr(filepath, chunks={'time': 1024})
    # chunksize = {"time": -1, "quarter": -1, "z": -1}
    
    # #ds = xr.open_zarr(filepath, chunks={'time': 1024})
    # outpath_roll = filepath.replace('.zarr', '_quartals.zarr')
    # vars_to_roll = ['tas', 'rainfall', 'snfall', 'sfwater', 'energyin', 'dswrad']

    # ds_sel = ds[vars_to_roll]
    # last_year = ds_sel.time.dt.year.values[-1]

    # logging.info(f'Create quartal averages for {vars_to_roll}...)')
    # means_quartal = ds_sel.resample(time='QE').mean()
    # means_quartal = means_quartal.sel(time=slice('1989-03-31', f'{last_year-1}-12-31'))
    # logging.info(f'Have quartal means for {means_quartal.time.values}')

    # # Get january 1st time stamps
    # import pandas as pd
    # jan1_times = ds_sel.time.sel(time=(ds_sel.time.dt.month == 1) & (ds_sel.time.dt.day == 1))
    # jan1_times = jan1_times.sel(time=slice('1990-01-01', None))
    # jan1_lookup = {pd.Timestamp(t).year: t for t in jan1_times.values}

    # # Map each quarter to its reference Jan 1st timestamp of next year
    # quarter_years = means_quartal.time.dt.year.values
    # quarter_nums = means_quartal.time.dt.quarter.values

    # reference_times = []
    # valid_indices = []
    # for i, year in enumerate(quarter_years):
    #     target_year = year + 1
    #     if target_year in jan1_lookup:
    #         reference_times.append(jan1_lookup[target_year])
    #         valid_indices.append(i)

    # means_quartal = means_quartal.isel(time=valid_indices)
    # quarter_nums = quarter_nums[valid_indices]

    # logging.info(f'Reindex quartals {means_quartal.time.values} to reference Jan 1st timestamps {reference_times} ...')

    # # Assign and reshape
    # means_quartal = means_quartal.assign_coords(
    #     year_jan1=('time', reference_times),
    #     quarter=('time', quarter_nums)
    # )

    # means_quartal = means_quartal.set_index(time=['year_jan1', 'quarter']).unstack('time')

    # means_quartal = (
    #     means_quartal.rename_dims({'year_jan1': 'time'})
    #     .assign_coords(time=means_quartal.year_jan1.values)
    #     .drop_vars('year_jan1'))

    # means_quartal = means_quartal.rename({var: f"{var}-avg" for var in means_quartal.data_vars})

    # print(f"Saving quartal mean results to {outpath_roll} ...")
    # means_quartal = means_quartal.transpose('time', 'quarter', 'z')
    # means_quartal = means_quartal.chunk(chunksize)

    # print(means_quartal)

    # with ProgressBar():
    #     means_quartal.astype(np.float32).to_zarr(outpath_roll, mode='w', consolidated=True)
