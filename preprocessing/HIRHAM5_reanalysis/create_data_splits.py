# -*- coding: utf-8 -*-
"""
Created on 24.07.2025   
@author: eschlager
Script for splitting dates into training and validation sets
with seasonal subsampling of the training and validation sets used during training.
"""
#%% 
import os
import logging
import json
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

script_dir = os.path.abspath(os.path.dirname(__file__))
project_dir = os.path.sep.join([script_dir, '..', '..'])
sys.path.append(os.path.sep.join([project_dir, 'src']))
import logging_config
logging.getLogger().setLevel(logging.INFO)


def sample_dates_by_year_chunks(weights, start_date, end_date, n_samples_per_year, figpath=None):
    """
    Sample dates within 12-month chunks starting from `start_date`, using the given weights.

    Parameters:
        dates (array-like): Array or Series of datetime-like objects.
        weights (array-like): Array of weights corresponding to each date.
        start_date (datetime-like): The start of the overall date range.
        end_date (datetime-like): The end of the overall date range.
        n_samples_per_year (int): Number of dates to sample per 12-month period.

    Returns:
        List of sampled dates.
    """
    sampled_dates = []

    # Ensure dates and weights are pandas Series
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    weights = weights[dates.dayofyear - 1] 

    current_start = pd.Timestamp(start_date)
    while current_start < pd.Timestamp(end_date):
        # Define the end of the 12-month period
        next_year = current_start + pd.DateOffset(years=1)
        current_end = min(next_year, pd.Timestamp(end_date))

        # Select dates within this period
        mask = (dates >= current_start) & (dates < current_end)
        period_dates = dates[mask]
        period_weights = weights[mask]

        if len(period_dates) < n_samples_per_year:
            current_start = next_year
            continue  # Skip periods with too few dates

        # Normalize weights
        period_weights = period_weights / period_weights.sum()

        # Sample without replacement
        sampled = np.random.choice(
            period_dates,
            size=n_samples_per_year,
            replace=False,
            p=period_weights
        )

        sampled_dates.extend(sampled)
        current_start = next_year

        if figpath is not None:
            plot_sampled_dates(period_dates, sampled, fig_path)

    return sampled_dates

def plot_sampled_dates(all_dates, sampled_dates, fig_path):
    plt.figure(figsize=(10, 1.5))
    plt.scatter(sampled_dates, np.zeros_like(sampled_dates), marker='|', color='darkblue')
    plt.title(f"Sampled Time Points for {all_dates.min()}-{all_dates.max()}")
    plt.yticks([])
    plt.xlabel("DOY")
    plt.grid(True, axis='x')
    plt.gca().set_xlim([all_dates.min(), all_dates.max()])
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, f'subsampling_from_{all_dates.min()}'))
    plt.show()


if __name__ == "__main__":
    
    data_path = os.path.sep.join([project_dir, 'data', 'processed', 'HIRHAM5-ERAI', 'v_04'])
    os.makedirs(data_path, exist_ok=True)
    logging_config.define_root_logger(os.path.join(data_path, f'log_split.txt'))
    
    #%% Create partition in train/val/test sets with temporal subsampling
    logging.info(f'----------- Create partition in train/val/test sets --------------')    

    train_start_date = pd.to_datetime('1990-01-01T12:00:00.000000000')
    train_end_date = pd.to_datetime('2015-12-31T12:00:00.000000000')
    val_start_date = pd.to_datetime('2016-01-01T12:00:00.000000000')
    val_end_date = pd.to_datetime('2016-12-31T12:00:00.000000000')

    ## temporal sub-sampling of training data
    doy = np.arange(1, 367)   # days 1 to 366
    mu = 205    # 24th of July ~ peak of melt season
    sigma = 60
    min_weight_doy = 0.0
    logging.info(f'Subsample training dates using seasonal weighting following gaussian distribution with mu {mu} and sigma {sigma}') 
    gauss_weights = np.exp(-0.5 * ((doy - mu) / sigma) ** 2)
    weights = min_weight_doy + (1 - min_weight_doy) * gauss_weights
    weights_doy = gauss_weights / gauss_weights.sum()
    np.savez(os.path.sep.join([data_path, 'weights_doy']), weights_doy=weights_doy)

    n_samples_per_year = 100
    logging.info(f'Sample {n_samples_per_year} dates per year')
    
    fig_path = os.path.sep.join([data_path, f'figures_{n_samples_per_year}'])
    os.makedirs(fig_path, exist_ok=True)

    np.random.seed(42)
    train_sampled_dates = sample_dates_by_year_chunks(weights_doy, train_start_date, train_end_date, n_samples_per_year, fig_path)
    np.random.seed(42)
    val_sampled_dates = sample_dates_by_year_chunks(weights_doy, val_start_date, val_end_date, n_samples_per_year, fig_path)

    sampled_train_dates = pd.DatetimeIndex(np.sort(train_sampled_dates))
    sampled_val_dates = pd.DatetimeIndex(np.sort(val_sampled_dates))

    train_dates = pd.date_range(start=train_start_date, end=train_end_date, freq='D')
    val_dates = pd.date_range(start=val_start_date, end=val_end_date, freq='D')


    logging.info(f'Number of all training dates: {len(train_dates)}')
    logging.info(f'Number of sub-sampled training dates: {len(sampled_train_dates)}')
    logging.info(f'Number of all validation dates: {len(val_dates)}')
    logging.info(f'Number of sub-sampled validation dates: {len(sampled_val_dates)}')



    logging.info(f'Save dates to json')
    json.dump({'train_sub':sampled_train_dates.strftime('%Y-%m-%dT%H:%M:%S.%f000').tolist(),
        'train':train_dates.strftime('%Y-%m-%dT%H:%M:%S.%f000').tolist(),
        'val_sub':sampled_val_dates.strftime('%Y-%m-%dT%H:%M:%S.%f000').tolist(),
        'val':val_dates.strftime('%Y-%m-%dT%H:%M:%S.%f000').tolist(),},
        open(os.path.sep.join([data_path, f'train_val_split_{n_samples_per_year}.json']),'w'),)
    
