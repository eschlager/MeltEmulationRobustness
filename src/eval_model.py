# -*- coding: utf-8 -*-
"""
Created on 15.08.2024
@author: eschlager
ModelEvaluator class: selection of evaluation plots for a prediction dataset
"""

import os
import sys
import logging
from datetime import datetime

import cartopy.crs as ccrs
import colorcet as cc
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from matplotlib.ticker import MaxNLocator
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate
import pandas as pd

# import local modules
from GRL_plotter import plot_greenland_only

plt.figure(constrained_layout=True)

LABEL_TRUE = 'true'
LABEL_PRED = 'predicted'

class ModelEvaluator():

    def __init__(self, ds, target_name, batch_size=64, zones=None):

        self.ds = ds
        self.target_name = target_name
        dataset = XarrayZarrDataset(ds, self.target_name)
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)#collate_fn=lambda x: collate_batch(x))
        self.zones = zones

        if 'z' in self.ds.dims:
            self.ds = self.ds.set_index(z=['y', 'x']).unstack('z')

        if self.target_name == 'snmel':
            self.target_label = 'melt'
            self.target_unit = 'mm w.e. per day'
            self.reduce_time = 'sum'
        elif self.target_name == 'rogl':
            self.target_label = 'runoff'
            self.target_unit = 'mm w.e. per day'
            self.reduce_time = 'sum'
        elif self.target_name == 'albedom':
            self.target_label = 'albedo'   
            self.target_unit = ' '
            self.reduce_time = 'mean'
        else:
            self.target_label = self.target_name
            self.target_unit = ' '
            self.reduce_time = 'sum'
            logging.warning(f"No label and unit defined for target {self.target_name}. Use 'sum' as default reduce_time for map plots.")
        
        if not self.target_unit:
            self.target_full_label = self.target_label
        else:
            self.target_full_label = f'{self.target_label} ({self.target_unit})'

        self.true = self.ds[f'{self.target_name}_true'].values.astype(np.float64)
        self.pred = self.ds[f'{self.target_name}_pred'].values.astype(np.float64)

    
    def get_rmse(self):
        return self.calc_rmse(self.true, self.pred)

    @staticmethod
    def calc_rmse(true, pred):
        mask = ~np.isnan(true) & ~np.isnan(pred)
        mse = torch.nn.MSELoss()(torch.from_numpy(true[mask]), torch.from_numpy(pred[mask]))
        return torch.sqrt(mse).item()

    def get_mae(self):
        return self.calc_mae(self.true, self.pred)

    @staticmethod
    def calc_mae(true, pred):
        mask = ~np.isnan(true) & ~np.isnan(pred)
        mae = torch.nn.L1Loss()(torch.from_numpy(true[mask]), torch.from_numpy(pred[mask]))
        return mae.item()

    def get_mbe(self):
        return self.calc_mbe(self.true, self.pred)

    @staticmethod
    def calc_mbe(true, pred):
        mask = ~np.isnan(true) & ~np.isnan(pred)
        mbe = torch.mean(torch.from_numpy(pred[mask])-torch.from_numpy(true[mask]))
        return mbe.item()
    
    def get_r2(self):
        return self.calc_r2(self.true, self.pred)

    @staticmethod
    def calc_r2(true, pred):
        from sklearn.metrics import r2_score
        mask = ~np.isnan(true) & ~np.isnan(pred)
        return r2_score(true[mask], pred[mask])

    @staticmethod
    def smart_truncate(x):
        import math
        """Truncate x toward zero using your magnitude rules."""
        if x is None:
            return None

        ax = abs(x)
        if 100 <= ax < 1000:
            # truncate towards zero to nearest 10
            return math.trunc(x / 10) * 10
        elif 10 <= ax < 100:
            # truncate towards zero to integer
            return math.trunc(x)
        else:
            # one-digit: leave unchanged
            return x


    def plot_pred_vs_target_density(self, x, y, ref_line=None, zone_cat=None, year=None, month=None, color_threshold=1.e5, x_lims=None, y_lims=None):
        '''
        ref_line: 'equal' for plotting line x=y or 'zero' for plotting line at y=0
        zone_cat: use only data from this zone category (int or float)
        year: int, e.g. 2020 (can be used together with month)
        month: str, e.g. 'January' (can be used together with year)
        color_threshold: threshold for bin counts to be masked in the colormap
        x_lims, y_lims: tuple (min, max) to set x and y axis limits; if None, automatic limits are used
        '''
     
        fig, ax = plt.subplots(figsize=(4.2,4.2))
        ax.set_xlabel(x)
        ax.set_ylabel(y)

        ds_dates = self.ds["time"].values

        if year is not None:
            dates_per_year = [d for d in pd.to_datetime(ds_dates) if d.year==year]
            if month is not None:   # specific month of specific year
                month_idx = datetime.strptime(month, "%B").month
                dates_per_month = [d for d in dates_per_year if d.month==month_idx]
                time_mask = np.isin(pd.to_datetime(ds_dates), pd.to_datetime(dates_per_month))
                time_idx = np.where(time_mask)[0]
                title = f'{month} {year}'
            else:  # one specific year
                time_mask = np.isin(pd.to_datetime(ds_dates), pd.to_datetime(dates_per_year))
                time_idx = np.where(time_mask)[0]
                title = year
            n_years = 1
        elif month is not None:   # specific month (aggregated over all years in dataset)
            month_idx = datetime.strptime(month, "%B").month
            dates_per_month = [d for d in pd.to_datetime(ds_dates) if d.month==month_idx]
            time_mask = np.isin(pd.to_datetime(ds_dates), pd.to_datetime(dates_per_month))
            time_idx = np.where(time_mask)[0]
            years = np.unique(pd.to_datetime(dates_per_month).year)
            n_years = len(years)
            if n_years >1:
                title = f'{month} {str(np.min(years))}-{str(np.max(years))}'
                color_threshold = color_threshold * n_years   # adapt color_threshold if plotting muliple years
            else: 
                title = f'{month} {years[0]}'
        else: # all dates in dataset
            time_idx = None
            years = np.unique(pd.to_datetime(ds_dates).year)
            n_years = len(years)
            if n_years >1:
                title = f'{str(np.min(years))}-{str(np.max(years))}'
            else: 
                title = f'{years[0]}'
            color_threshold = color_threshold * n_years   # adapt color_threshold if plotting muliple years

        if zone_cat is not None: # for spatial masking
            loc_mask = self.zones == zone_cat
            loc_idx = np.where(loc_mask)[0]
            title = f'{title} zone {int(zone_cat)}'
        else: 
            loc_idx = None

        if (time_idx is not None) and (loc_idx is not None):
            xx = self.ds[x].isel(time=time_idx).where(loc_mask).values
            yy = self.ds[y].isel(time=time_idx).where(loc_mask).values
        elif time_idx is not None:
            xx = self.ds[x].isel(time=time_idx).values
            yy = self.ds[y].isel(time=time_idx).values
        elif loc_idx is not None:
            xx = self.ds[x].where(loc_mask).values
            yy = self.ds[y].where(loc_mask).values
        else:
            xx = self.ds[x].values
            yy = self.ds[y].values

        xx = xx.flatten()
        yy = yy.flatten()
        valid_mask = ~np.isnan(xx) & ~np.isnan(yy)
        xx = xx[valid_mask]
        yy = yy[valid_mask]
        
        logging.info("  plotting ...")
        gridsize = 50
        hb = ax.hexbin(xx, yy, gridsize=gridsize, bins='log', cmap='RdYlBu_r', linewidths=(0,), mincnt=1, vmin=1)
        offsets = hb.get_offsets()        # bin centers
        counts = hb.get_array()           # bin values (counts)

        
        # add text of true and predicted amount in Gt
        if self.target_unit == 'mm w.e. per day':
            scaling_factor = 1.796553703424 / 58391   # adjust for dataset
            if (x.endswith('true') and y.endswith('pred')):
                xx_total = xx.sum()*scaling_factor / n_years
                yy_total = yy.sum()*scaling_factor / n_years
                if n_years >1:
                    ax.text(0.6, 0.9, f'Avg. total {LABEL_TRUE}: {xx_total:.0f}Gt', transform=ax.transAxes, ha='right')
                    ax.text(0.6, 0.85, f'Avg. total {LABEL_PRED}: {yy_total:.0f}Gt', transform=ax.transAxes, ha='right')
                else:
                    ax.text(0.5, 0.9, f'{LABEL_TRUE}: {xx_total:.0f}Gt', transform=ax.transAxes, ha='right')
                    ax.text(0.5, 0.85, f'{LABEL_PRED}: {yy_total:.0f}Gt', transform=ax.transAxes, ha='right')
                plt.xlabel(f'{LABEL_TRUE} {self.target_full_label}')
                plt.ylabel(f'{LABEL_PRED} {self.target_full_label}')
        # add rmse and mae as text in figures
        rmse = self.calc_rmse(xx, yy)
        mae = self.calc_mae(xx,yy)
        ax.text(.9, 0.15, f'RMSE: {rmse:.2f}', transform=ax.transAxes, ha='right')
        ax.text(.9, 0.10, f'MAE: {mae:.2f}', transform=ax.transAxes, ha='right')

        # Mask high-count bins from colormap and colorbar
        masked_counts = np.ma.array(counts, mask=counts > color_threshold)
        hb.set_array(masked_counts)
        vmin = np.nanmin(masked_counts)
        vmax = np.nanmax(masked_counts)

        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = None, None
        hb.set_clim(vmin=vmin, vmax=vmax)

        # compute flat-to-flat width in data coords
        from matplotlib.patches import RegularPolygon
        from matplotlib.collections import PatchCollection
        x0, x1 = ax.get_xlim()
        flat_width = (x1 - x0) / gridsize

        # hexagon circumradius R (distance center -> vertex) given flat-to-flat width w:
        # flat-to-flat = R * sqrt(3)  =>  R = w / sqrt(3)
        R = flat_width / np.sqrt(3)

        patches = []
        high_bins = offsets[counts > color_threshold]
        for x_center, y_center in high_bins:
            hex_patch = RegularPolygon((x_center, y_center),
                                    numVertices=6,
                                    radius=R,
                                    orientation=0.,
                                    edgecolor='none', facecolor='black')
            patches.append(hex_patch)

        pc = PatchCollection(patches, match_original=True)  # match_original keeps edge/face colors
        ax.add_collection(pc)

        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.2)  # size is 4% of axes height
        cbar = fig.colorbar(hb, cax=cax)
        cbar.set_label(f"bin count (≤ {int(color_threshold)})")
        
        # plot reference line
        if ref_line == 'equal':
            ax_xmin, ax_xmax = ax.get_xlim()
            ax_ymin, ax_ymax = ax.get_ylim()
            if x_lims is not None and y_lims is not None:
                min_val = min(x_lims[0], y_lims[0])
                max_val = max(x_lims[1], y_lims[1])
            elif x_lims is not None:    
                min_val = x_lims[0]
                max_val = x_lims[1]
            elif y_lims is not None:
                min_val = y_lims[0]
                max_val = y_lims[1]
            else:
                min_val = min(ax_xmin, ax_ymin)
                max_val = max(ax_xmax, ax_ymax)
            if ax_xmin < min_val:
                logging.warning(f'Data x minimum {ax_xmin} is smaller than defined x-axis minimum {min_val}!')
            if ax_ymin < min_val:
                logging.warning(f'Data y minimum {ax_ymin} is smaller than defined y-axis minimum {min_val}!')
            if ax_xmax > max_val:
                logging.warning(f'Data x maximum {ax_xmax} is larger than defined x-axis maximum {max_val}!')
            if ax_ymax > max_val:
                logging.warning(f'Data y maximum {ax_ymax} is larger than defined y-axis maximum {max_val}!')

            ax.plot([min_val,max_val], [min_val,max_val], 'k-', alpha=0.4, zorder=0)
            ax.set_xlim(xmin=min_val, xmax=max_val)
            ax.set_ylim(ymin=min_val, ymax=max_val)
            ax.set_aspect('equal')
            nticks = 4
            ticks = np.linspace(0, self.smart_truncate(max_val), nticks)
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
        else:
            if x_lims is not None:
                xmin, xmax = x_lims
            else:
                xmin, xmax = ax.get_xlim()

            if y_lims is not None:
                ymin, ymax = y_lims
            else:
                ymin, ymax = ax.get_ylim()
            ax.set_xlim(xmin=xmin, xmax=xmax)
            ax.set_ylim(ymin=ymin, ymax=ymax)
        
        ax.tick_params(axis='both', labelsize=8)
        
        ax.set_title(title)
        plt.tight_layout()
        return ax

    def _get_value_limits(self, minval, maxval, value_lim):
        """
        Determine value limits and colorbar extension based on provided value limits.
        """
        if value_lim is not None:
            value_min, value_max = value_lim
            if value_max is None:
                value_max = maxval
            if value_min is None:
                value_min = minval

            if maxval > value_max:
                if minval < value_min:
                    extend_colorbar = 'both'
                else:
                    extend_colorbar = 'max'
            elif minval < value_min:
                extend_colorbar = 'min'
            else:
                extend_colorbar = 'neither'
            maxval = value_max
            minval = value_min
        else:
            extend_colorbar = 'neither'
            
        return minval, maxval, extend_colorbar


    def plot_map(self, year=None, month=None, date=None, residual_max=None, value_lim=None, join_colorbar=False):
        """
        plot maps of predicted, true, and pred-true target values 
        year: int, e.g. 2020 (can be used together with month)
        month: str, e.g. 'January' (can be used together with year)
        date: list of dates to select from the dataset, e.g. ['2020-01-01', '2020-02-01'] (cannot be used together with year or month)
        residual_max: float, maximum absolute value for residual colorbar (symmetric around zero); 
            if values exceed limit, arrows at the ends of the colorbar indicate that
        value_lim: tuple (min, max) to set value limits for true and predicted plots; if None, automatic limits are used
            if values exceed limit, arrows at the ends of the colorbar indicate that
        """
        import matplotlib.gridspec as gridspec
        fig = plt.figure(figsize=(8,4))            
        axs = [None]*3
        display_crs = ccrs.NorthPolarStereo(central_longitude=-40)
        
        if join_colorbar:
            # no gap between first two subplots (true and predicted)
            gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 0.12, 1], wspace=0)
            axs[0] = fig.add_subplot(gs[0, 0], projection=display_crs)
            axs[1] = fig.add_subplot(gs[0, 1], projection=display_crs, sharey=axs[0])
            axs[2] = fig.add_subplot(gs[0, 3], projection=display_crs)
        else:
            # gap between each subplot
            gs = gridspec.GridSpec(1, 5, width_ratios=[1, 0.15, 1, 0.15, 1], wspace=0)
            axs[0] = fig.add_subplot(gs[0, 0], projection=display_crs)
            axs[1] = fig.add_subplot(gs[0, 2], projection=display_crs)
            axs[2] = fig.add_subplot(gs[0, 4], projection=display_crs)

        data = self.ds[[f'{self.target_name}_true', f'{self.target_name}_pred']]
        data = data.rename({f'{self.target_name}_true': 'true', f'{self.target_name}_pred': 'pred'})
        
        if date is not None and ((year is not None) or (month is not None)):
            logging.error('Both date and year/month specified. Specifiy only one of them.')
            raise ValueError('Both date and year/month specified. Specifiy only one of them.')
        
        if date is not None:   # plot for one specific day
            try:
                date_str = pd.to_datetime(date.astype('datetime64[D]').item()).strftime('%Y-%m-%d')
            except: 
                date_str = date.strftime('%Y-%m-%d')
            xx = data['true'].sel(time=date, method='nearest')
            yy = data['pred'].sel(time=date, method='nearest')
            nr_days = 1
            xx_daily_values = xx.values.flatten()
            yy_daily_values = yy.values.flatten()
            logging.info(f'Get data from datetime {xx.time.values}')
            title = f"{self.target_label} {date_str}"
            cbar_label = self.target_unit
        elif year is not None:
            ds_dates = self.ds["time"].values
            dates_per_year = [d for d in pd.to_datetime(ds_dates) if d.year==year]
            if month is not None:   # specific month of specific year
                if isinstance(month, list):
                    # Multiple months
                    month_indices = [datetime.strptime(m, "%B").month for m in month]
                    dates_per_months = [d for d in dates_per_year if d.month in month_indices]
                    time_mask = np.isin(pd.to_datetime(ds_dates), pd.to_datetime(dates_per_months))
                    title = f'{self.target_label} {" & ".join(month)} {year} ({self.reduce_time})'
                    cbar_label = self.target_unit.replace('day', 'season')
                else:
                    # Single month (original logic)
                    month_idx = datetime.strptime(month, "%B").month
                    dates_per_month = [d for d in dates_per_year if d.month==month_idx]
                    time_mask = np.isin(pd.to_datetime(ds_dates), pd.to_datetime(dates_per_month))
                    title = f'{self.target_label} {month} {year} ({self.reduce_time})'
                    cbar_label = self.target_unit.replace('day', 'month')
            else:  # one specific year
                time_mask = np.isin(pd.to_datetime(ds_dates), pd.to_datetime(dates_per_year))
                title = f'{self.target_label} {year} ({self.reduce_time})'
                cbar_label = self.target_unit.replace('day', 'year')
            time_idx = np.where(time_mask)[0]
            xx = data['true'].isel(time=time_idx)
            yy = data['pred'].isel(time=time_idx)
            nr_days = len(time_idx)
            xx_daily_values = xx.values.flatten()
            yy_daily_values = yy.values.flatten()
            xx = xx.reduce(getattr(np, self.reduce_time), dim="time")
            yy = yy.reduce(getattr(np, self.reduce_time), dim="time")
        elif month is not None:   # specific month (aggregated over all years in dataset)
            ds_dates = self.ds["time"].values
            if isinstance(month, list):
                # Multiple months
                month_indices = [datetime.strptime(m, "%B").month for m in month]
                dates_per_month = [d for d in pd.to_datetime(ds_dates) if d.month in month_indices]
                time_mask = np.isin(pd.to_datetime(ds_dates), pd.to_datetime(dates_per_month))
                time_idx = np.where(time_mask)[0]
                title = f'{self.target_label} {" & ".join(month)} '
                cbar_label = self.target_unit.replace('day', 'season')
            else:
                # Single month (original logic)
                month_idx = datetime.strptime(month, "%B").month
                dates_per_month = [d for d in pd.to_datetime(ds_dates) if d.month==month_idx]
                time_mask = np.isin(pd.to_datetime(ds_dates), pd.to_datetime(dates_per_month))
                time_idx = np.where(time_mask)[0]
                title = f'{self.target_label} {month} '
                cbar_label = self.target_unit.replace('day', 'month')
            years = np.unique(pd.to_datetime(dates_per_month).year)
            n_years = len(years)
            xx = data['true'].isel(time=time_idx)
            yy = data['pred'].isel(time=time_idx)
            nr_days = len(time_idx)
            xx_daily_values = xx.values.flatten()
            yy_daily_values = yy.values.flatten()
            xx = xx.reduce(getattr(np, self.reduce_time), dim="time")
            yy = yy.reduce(getattr(np, self.reduce_time), dim="time")
            if self.reduce_time == 'sum':  # calculated monthly sums, so now we divide by nr years to get average monthly total
                xx = xx / n_years
                yy = yy / n_years
            if n_years == 1:  # In case whole dataset is only one year
                title = title+f'({self.reduce_time})'
            else:
                title = title+f'({self.reduce_time}; avg. across {str(np.min(years))}-{str(np.max(years))})'
        else:
            xx = data['true'].sum(dim='time', skipna=False)
            yy = data['pred'].sum(dim='time', skipna=False)
            nr_days = len(data['time'])
            xx_daily_values = data['true'].values.flatten()
            yy_daily_values = data['pred'].values.flatten()
            years = np.unique(pd.to_datetime(self.ds["time"].values).year)
            if len(years)==1:
                cbar_label = self.target_unit.replace('day', 'year')
                title = f'{self.target_label} {years[0]} ({self.reduce_time})'
            elif len(years) >1:
                cbar_label = f'mm w.e.'
                title = f'{self.target_label} {years.min()}-{years.max()} ({self.reduce_time})'

        # select colormap
        if self.target_name == 'albedom':
            cmap = mcolors.ListedColormap(cc.linear_blue_5_95_c73)
        else:  # good colormap for melt, runoff, etc.:
            cmap = mcolors.ListedColormap(cc.linear_kryw_5_100_c67[::-1])
        cmap.set_bad((0, 0, 0, 0.0)) 

        # get value limits
        min_x = np.nanmin(xx.values.flatten())
        max_x =  np.nanmax(xx.values.flatten())
        min_y = np.nanmin(yy.values.flatten())
        max_y =  np.nanmax(yy.values.flatten())

        if join_colorbar:
            # value limits
            minval = min(min_x, min_y)
            maxval = max(max_x, max_y)
            # minval = min(minval,0)
            # maxval = max(maxval,1)
            minval_x, maxval_x, extend_colorbar_x = self._get_value_limits(minval, maxval, value_lim)
            minval_y, maxval_y, extend_colorbar_y = minval_x, maxval_x, extend_colorbar_x
        else:
            minval_x, maxval_x, extend_colorbar_x = self._get_value_limits(min_x, max_x, value_lim)
            minval_y, maxval_y, extend_colorbar_y = self._get_value_limits(min_y, max_y, value_lim)

        logging.info(f"Max limits (x): {maxval_x}, {maxval_y}")

        # plot true
        try:
            axs[0], p0 = plot_greenland_only(xx, ax=axs[0], pcolormesh_kwargs={'cmap': cmap, 'vmin': minval_x, 'vmax': maxval_x})
            axs[1], p1 = plot_greenland_only(yy, ax=axs[1], pcolormesh_kwargs={'cmap': cmap, 'vmin': minval_y, 'vmax': maxval_y})
        except ValueError as e:
            if "Couldn't find lon/lat variables!" in str(e):
                logging.info("Plot map on regular grid.")
                p0 = axs[0].pcolormesh(xx, cmap=cmap, vmin=minval_x, vmax=maxval_x)
                p1 = axs[1].pcolormesh(yy, cmap=cmap, vmin=minval_y, vmax=maxval_y)

        axs[0].set_title(LABEL_TRUE)
        axs[0].set_axis_off()        
        axs[1].set_title(LABEL_PRED)
        axs[1].set_axis_off()

        # plot residuals
        residual = yy - xx
        mindiff = np.nanmin(residual.values.flatten())
        maxdiff = np.nanmax(residual.values.flatten())
        absdiff = max(abs(mindiff), abs(maxdiff), 0.1)
        if residual_max is not None:
            extend_colorbarmin = True if residual_max < abs(mindiff) else False
            extend_colorbarmax = True if residual_max < abs(maxdiff) else False
            absdiff = residual_max
        else:
            extend_colorbarmin = None
            extend_colorbarmax = None
        margin = 0.02 * 2 * absdiff
        norm = mcolors.TwoSlopeNorm(vmin=-absdiff-margin, vcenter=0, vmax=absdiff+margin)

        cmap = mpl.cm.get_cmap('seismic')
        cmap.set_bad((0, 0, 0, 0.0)) 
        try:
            axs[2], p2 = plot_greenland_only(residual, ax=axs[2], pcolormesh_kwargs={'cmap': cmap, 'norm': norm})
        except ValueError as e:
            if "Couldn't find lon/lat variables!" in str(e):
                p2 = axs[2].pcolormesh(residual, cmap=cmap, norm=norm)
        axs[2].set_title('residual')
        axs[2].set_axis_off()
    

        # make colorbars
        extra_drop = 0.02   # vertical gap (figure fraction) between axes bottom and colorbar top
        cbar_height = 0.03  # thickness of both colorbars (figure fraction)
        bbox0 = axs[0].get_position()
        bbox1 = axs[1].get_position()
        
        if join_colorbar:
            # joined colorbar true and pred
            x0 = min(bbox0.x0, bbox1.x0)
            x1 = max(bbox0.x1, bbox1.x1)
            width01 = x1 - x0
            y01 = min(bbox0.y0, bbox1.y0) - extra_drop - cbar_height
            cax01 = fig.add_axes([x0, y01, width01, cbar_height])
            cb01 = fig.colorbar(p0, cax=cax01, orientation='horizontal', extend=extend_colorbar_x)
            cb01.set_label(cbar_label)
        else:
            x0 = bbox0.x0
            x1 = bbox0.x1
            width01 = x1 - x0
            y0 = bbox0.y0 - extra_drop - cbar_height
            cax0  = fig.add_axes([x0, y0, width01, cbar_height])
            cbar = fig.colorbar(p0, cax=cax0, orientation='horizontal', extend=extend_colorbar_x)
            cbar.set_label(cbar_label)

            x0 = bbox1.x0
            x1 = bbox1.x1
            width01 = x1 - x0
            y0 = bbox1.y0 - extra_drop - cbar_height
            cax1  = fig.add_axes([x0, y0, width01, cbar_height])
            cbar = fig.colorbar(p1, cax=cax1, orientation='horizontal', extend=extend_colorbar_y)
            cbar.set_label(cbar_label)


        # create residual colorbar
        bbox2 = axs[2].get_position()
        x3 = bbox2.x0
        width3 = bbox2.width
        y3 = bbox2.y0 - extra_drop - cbar_height
        cax3  = fig.add_axes([x3, y3, width3, cbar_height])
        if extend_colorbarmin and extend_colorbarmax:
            extend_res_colorbar = 'both'
        elif extend_colorbarmin:
            extend_res_colorbar = 'min'
        elif extend_colorbarmax:
            extend_res_colorbar = 'max'
        else:
            extend_res_colorbar = 'neither'
        cb3 = fig.colorbar(p2, cax=cax3, orientation='horizontal', extend=extend_res_colorbar)
        cb3.set_label(cbar_label)


        # indicate min and max at the residual colorbar:
        cbar_ax = cb3.ax
        trans = cbar_ax.transData
        inv = cbar_ax.transData.inverted()
        triangle_size = 8
        _, y_top_disp = cbar_ax.transAxes.transform((0, 1)) + triangle_size
        # triangle for minimum of difference (over-prediction)
        x_disp, _ = trans.transform((mindiff, 0))
        triangle_disp = np.array([
            [x_disp, y_top_disp],
            [x_disp-triangle_size / 3, y_top_disp+triangle_size],
            [x_disp+triangle_size / 3, y_top_disp+triangle_size]])
        triangle_data = inv.transform(triangle_disp)
        triangle = patches.Polygon(triangle_data, color='blue', clip_on=False)
        cbar_ax.add_patch(triangle)
        # triangle for maximum of difference (under-prediction)
        x_disp, _ = trans.transform((maxdiff, 0))
        triangle_disp = np.array([
            [x_disp, y_top_disp],
            [x_disp-triangle_size / 3, y_top_disp+triangle_size],
            [x_disp+triangle_size / 3, y_top_disp+triangle_size]])
        triangle_data = inv.transform(triangle_disp)
        triangle = patches.Polygon(triangle_data, color='red', clip_on=False)
        cbar_ax.add_patch(triangle)
                
        tick_locator = MaxNLocator(nbins=5)
        cb3.locator = tick_locator
        cb3.update_ticks()  

        # add text
        if self.target_name == 'snmel':
            threshold = 1.
            area_scaling_factor = 1796553.703424 / 58391 / nr_days   # transform number of pixels into km²
            logging.info(f'Calculate mean melt extent for {nr_days} days...')
            xx_pos = xx_daily_values[xx_daily_values>threshold]
            xx_melt_extent = len(xx_pos)
            axs[0].text(1., 0.0, rf'ME: {int(xx_melt_extent*area_scaling_factor)}km$\mathrm{{^2}}$', transform=axs[0].transAxes, ha='right')
            if xx_melt_extent > 100:
                xx_median = np.median(xx_pos)
                axs[0].text(1., 0.12, f'med: {xx_median:.2f}', transform=axs[0].transAxes, ha='right')
                from scipy.stats import iqr
                xx_iqr = iqr(xx_pos)
                axs[0].text(1., 0.06, f'IQR: {xx_iqr:.2f}', transform=axs[0].transAxes, ha='right')

            yy_pos = yy_daily_values[yy_daily_values>threshold]
            yy_melt_extent = len(yy_pos)
            axs[1].text(1., 0.0, rf'ME: {int(yy_melt_extent*area_scaling_factor)}km$\mathrm{{^2}}$', transform=axs[1].transAxes, ha='right')
            if xx_melt_extent > 100:
                yy_median = np.median(yy_pos)
                axs[1].text(1., 0.12, f'med: {yy_median:.2f}', transform=axs[1].transAxes, ha='right')
                yy_iqr = iqr(yy_pos)
                axs[1].text(1., 0.06, f'IQR: {yy_iqr:.2f}', transform=axs[1].transAxes, ha='right')

        rmse = self.calc_rmse(xx_daily_values, yy_daily_values)
        mae = self.calc_mae(xx_daily_values, yy_daily_values)
        mbe = self.calc_mbe(xx_daily_values, yy_daily_values)
        axs[2].text(1., 0.18, f'RMSE: {rmse:.2f}', transform=axs[2].transAxes, ha='right')
        axs[2].text(1., 0.12, f'MAE: {mae:.2f}', transform=axs[2].transAxes, ha='right')
        axs[2].text(1., 0.06, f'MBE: {mbe:.2f}', transform=axs[2].transAxes, ha='right')

        plt.suptitle(title)
        return axs[0]
                
                
def plot_loss(out_dir, log_scale=False):
    '''
    Plot the training and validation loss from the loss.csv file in the output directory.
    log_scale: if True, use logarithmic scale for the y-axis
    '''
    loss_history = pd.read_csv(os.path.sep.join([out_dir, 'loss.csv']), delimiter=';')
    train_loss = ['train_loss'] + loss_history['train_loss'].tolist()
    last_index = len(train_loss) - 1 - train_loss[::-1].index('train_loss')
    train_loss = train_loss[last_index + 3:]  # skip first two epochs
    train_loss = [float(x) for x in train_loss]

    val_loss = ['val_loss'] + loss_history['val_loss'].tolist()
    last_index = len(val_loss) - 1 - val_loss[::-1].index('val_loss')
    val_loss = val_loss[last_index + 3:]
    val_loss = [float(x) for x in val_loss]
    
    try: 
        lr = ['lr'] + loss_history['lr'].tolist()
        last_index = len(lr) - 1 - lr[::-1].index('lr')
        lr = lr[last_index + 3:]
        lr = [float(x) for x in lr]
    except:
        lr = None
    fig, ax = plt.subplots()
    if lr is not None:
        ax_lr = ax.twinx()
        ax_lr.plot(range(3,len(lr)+3), lr, color='grey')
        ax_lr.set_ylabel('learning rate')
        ax_lr.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax.plot(range(3,len(train_loss)+3), train_loss, label='training')
    ax.plot(range(3,len(val_loss)+3), val_loss, label='validation')
    ax.legend(loc='upper right')
    ax.set_ylabel('loss')
    if log_scale:
        ax.set_yscale('log')
        fig.savefig(os.path.sep.join([out_dir, 'loss_log.png']), bbox_inches='tight')
    else:
        fig.savefig(os.path.sep.join([out_dir, 'loss.png']), bbox_inches='tight')
    plt.close(fig)
    
    
    

class XarrayZarrDataset(Dataset):
    def __init__(self, ds, target_name):
        """
        ds: xarray Dataset containing the real values and predictions
        batch_size: Batch size for DataLoader
        """
        self.ds = ds
        self.dates = ds["time"]
        self.target_name = target_name

    def __len__(self):
        return len(self.dates)

    def __getitem__(self, idx):
        date = self.dates.isel(time=idx)
        real_values_batch = self.ds[f"{self.target_name}_true"].isel(time=idx)
        predictions_batch = self.ds[f"{self.target_name}_pred"].isel(time=idx)
        logging.info(real_values_batch.values)
        logging.info(predictions_batch.values)
        logging.info(date.values)
        return real_values_batch.values, predictions_batch.values, date.values

   
def collate_batch(batch):
    y_true = np.concatenate([item[0] for item in batch])

    y_pred = np.concatenate([item[1] for item in batch])

    # Repeat datetime indices to match the number of samples (n_locations * n_zones)
    dates = np.repeat(np.array([item[2] for item in batch]), y_true.shape[0]/len(batch))
    
    return y_true, y_pred, dates

def custom_collate(batch):
    true, pred, date = zip(*batch)  # unzip list of (tensor, string)
    batched_true = default_collate(true)  # stack tensors as usual
    batched_pred = default_collate(pred)
    # keep strings as list (no tensor conversion)
    batched_date = list(date)
    return batched_true, batched_pred, batched_date
    