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



if __name__ == "__main__":
    dask.config.set(scheduler='threads')

    out_path = os.path.sep.join([project_dir, 'data', 'processed', 'HIRHAM5-ERAInterim', 'v_02'])
    os.makedirs(out_path, exist_ok=True)
    logging_config.define_root_logger(os.path.join(out_path, f'log.txt'))
    
    interim_path = os.path.sep.join([project_dir, 'data', 'interim', 'ERAI', 'HIRHAM5'])
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

    # %% -------------------- Create stacked nc files for AuxFiles --------------------
    # -------------------- Create file with GRL mask --------------------
    filepath = os.path.sep.join([out_path, 'GRLmask.zarr'])
    ds_basins = xr.open_dataset(os.path.sep.join([in_path_aux, 'GRLmask.nc']))
    ds = ds_basins[['glacGRL', 'maskbas']]
    ds = ds.stack(z=('x', 'y')).reset_index('z')  # flatten spatial dimension
    ds = ds.dropna(dim='z', how='all')  # drop all NaN values
    ds.to_zarr(filepath, mode="w", consolidated=True)
    logging.info(f"GRL mask saved to {filepath}")
    ds.close()


    # -------------------- Create file with zones --------------------
    filepath = os.path.sep.join([out_path, 'GRLzones.zarr'])   # output file from create_spatial_subsampling.py
    ds_zones = xr.open_dataset(os.path.sep.join([in_path_aux, 'GRLzones.nc']))
    ds = ds_zones['zones']
    ds = ds.stack(z=('x', 'y')).reset_index('z')  # flatten spatial dimension
    ds = ds.dropna(dim='z', how='all')  # drop all NaN values
    ds.to_zarr(filepath, mode="w", consolidated=True)
    logging.info(f"GRL mask saved to {filepath}")
    ds_zones.close()


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
    ds.to_netcdf(filepath)
    logging.info(f"Flattened spatial sub-sampling indices saved to {filepath}")
    ds_t.close()
    ds.close()
    

    # %% Correct daily files: clip to valid ranges, correct runoff values using available water estimates

    logging.info(f"Load all daily files ...")
    years = range(1980, 2017)
    dropvars = ['tsl', 'sn', 'evspsbl', 'gld']
    chunks_init = {"time": 1024, "y": 64, "x": 64}
    ds_all = xr.open_mfdataset(
        [f"{in_path_h5}/Daily2D_GRL_{year}.nc" for year in years],
        concat_dim="time",
        combine="nested",
        chunks=chunks_init
    ).drop_vars(dropvars)
    ds_all = ds_all.sortby('time')


    logging.info(f"Correct data ranges ...")
    vars_ranges = {'ahfs': (-140,None),
                    }

    for var, (vmin,vmax) in vars_ranges.items():
        if var in ds_all.data_vars:
            data = ds_all[var]
            logging.info(f" check if {var} in [{vmin}, {vmax}]...")
            clipped_at_min = (data < vmin) if vmin is not None else xr.zeros_like(data, dtype=bool)
            clipped_at_max = (data > vmax) if vmax is not None else xr.zeros_like(data, dtype=bool)
            n_clipped_min = clipped_at_min.sum().compute().item()
            n_clipped_max = clipped_at_max.sum().compute().item()

            if n_clipped_min > 0:
                logging.info(f"   # samples clipped at min: {n_clipped_min}")
            if n_clipped_max > 0:
                logging.info(f"   # samples clipped at max: {n_clipped_max}")
            ds_all[var] = data.clip(min=vmin, max=vmax)

        else:
            logging.info(f"{var}: not found in dataset")


    # %% -------------------- Restructure dataset, and save daily data to zarr files --------------------
    filepath = os.path.sep.join([out_path, 'base_dataset.zarr'])
    chunksize = {"time": 2048, "z": -1}
    logging.info("Re-structure dataset...")
    # ds = ds.to_dataarray(dim='variable_names') # stack all variables into an extra dimension
    # ds.name = "data"
    logging.info("Flatten spatial dimension...")
    ds = ds_all.stack(z=('x', 'y')).reset_index('z')  # flatten spatial dimension
    logging.info("Drop nans...")
    # ds = ds.dropna(dim='z', how='all')  # drop all NaN values
    nan_mask = ~ds['tas'].isnull().all(dim=["time"]).compute()   # compute null only from tas, because either all vars or none are nan for a specific location!
    logging.info(f'Shape of nan-maks: {nan_mask.shape}')
    logging.info(f'amount of all z: {len(ds.z)}')
    ds = ds.isel(z=nan_mask)
    logging.info(f'amount of non-nan z: {len(ds.z)}')

    # # compute binary masks based on thresholds:
    # binary_variables = {'rainfall': 1., 'snfall': 1., 'snmel': 1., 'rogl': 1.}
    # for v,t in binary_variables.items():
    #     logging.info(f'Create binary mask for {v} using threshold {t}...')
    #     mask = (ds[v] > t).astype(bool)  # 1 where condition is True, 0 otherwise
    #     ds[f'{v}_mask'] = mask

    logging.info("Rechunk time dimension...")
    ds = ds.chunk(chunksize)

    logging.info(f"Saving data to {filepath}...")

    with ProgressBar():
        ds.to_zarr(filepath, mode="w", consolidated=True)


    #%% -------------------- Create 10yr rolling averages --------------------
    outpath_roll = filepath.replace('.zarr', '_10yr.zarr')
    vars_to_roll = ['tas', 'rainfall', 'snfall']

    ds_sel = ds[vars_to_roll]

    logging.info(f'Create 10yrs rolling mean for {vars_to_roll}...)')
    window_size = 3652
    rolling_10yr = ds_sel.rolling(time=window_size, min_periods=window_size)

    logging.info(f'  drop nans ...')
    means_10yr = rolling_10yr.mean()
    means_10yr = means_10yr.sel(time=slice('1990-01-01', None))

    means_10yr = means_10yr.rename({var: f"{var}_10yr-avg" for var in means_10yr.data_vars})

    print(f"Saving 10years rolling mean results to {outpath_roll} ...")
    means_10yr = means_10yr.chunk(chunksize)
    with ProgressBar():
        means_10yr.to_zarr(outpath_roll, mode='w', consolidated=True)

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


    # %% -------------------- Create medium-range rolling averages --------------------
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

    # rolling_med = rolling_med.rename({var: f"{var}_med-avg" for var in rolling_med.data_vars})
    # rolling_med = rolling_med.chunk(chunksize)
    # print(f"Saving rolling mean over past {window_size} days to {outpath_roll} ...")
    # with ProgressBar():
    #     rolling_med.to_zarr(outpath_roll, mode='w', consolidated=True)
    
    # # %% -------------------- Create medium-range rolling min, max and range for tas --------------------
    # vars_to_roll = ['tas']
    # for v in vars_to_roll:
    #     ds_sel = ds[v].sel(time=slice("1988-12-01", None))
    #     logging.info(f'Create medium-range rolling min, max, and range for {v}...')
    #     window_size = 7
    #     rolling = ds_sel.rolling(time=window_size, min_periods=window_size)

    #     min_ = rolling.min()
    #     max_ = rolling.max()
    #     range_ = max_ - min_

    #     logging.info(f'  drop nans ...')
    #     min_ = min_.sel(time=slice("1989-01-01", None))
    #     max_ = max_.sel(time=slice("1989-01-01", None))
    #     range_ = range_.sel(time=slice("1989-01-01", None))

    #     min_.name = f"{v}_med-min"
    #     max_.name = f"{v}_med-max"
    #     range_.name = f"{v}_med-range"

    #     rolling_ds = xr.merge([min_, max_, range_])

    #     print(f"Saving rolling min, max and range for {window_size} days of {v} to {outpath_roll} ...")
    #     rolling_ds = rolling_ds.chunk(chunksize)
    #     with ProgressBar():
    #         rolling_ds.to_zarr(outpath_roll, mode='a', consolidated=True)
        
