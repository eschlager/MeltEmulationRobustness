# -*- coding: utf-8 -*-
"""
Created on 05.09.2024
@author: eschlager
Helper functions
"""

import numpy as np
import logging


# scoring functions
def get_rmse(residual, ndigits=None):
    residual = residual.astype('float64')
    score = np.sqrt(np.mean((residual)**2)).item()
    if isinstance(ndigits, int):
        score = round(score, ndigits=ndigits)
    return score

def get_mae(residual, ndigits=None):
    residual = residual.astype('float64')
    score = np.mean(np.abs(residual)).item()
    if isinstance(ndigits, int):
        score = round(score, ndigits=ndigits)
    return score

def get_mbe(residual, ndigits=None):
    residual = residual.astype('float64')
    score = np.mean(residual).item()
    if isinstance(ndigits, int):
        score = round(score, ndigits=ndigits)
    return score

def calc_anomaly(ds, clim):
    md_clim = clim['time'].dt.strftime('%m-%d')
    clim = clim.assign_coords(md=md_clim)

    # group by the md coordinate and average
    clim = clim.groupby('md').mean(dim='time')

    md_data = ds['time'].dt.strftime('%m-%d')
    ds = ds.assign_coords(md=('time', md_data.values)) 

    anomaly = (ds.groupby('md') - clim).compute()

    return anomaly, clim, ds

def get_r2(true, pred, ndigits=None):
    from sklearn.metrics import r2_score
    true = true.astype('float64').flatten()
    pred = pred.astype('float64').flatten()
    true = true[~np.isnan(true)]
    pred = pred[~np.isnan(pred)]
    score = r2_score(true, pred)
    if isinstance(ndigits, int):
        score = round(score, ndigits=ndigits)
    return score

# transformation functions
def sym_log(x, a):
    """ 
    Symmetric logarithm transformation.
    """
    return np.sign(x) * np.log1p(np.abs(x) / a)

def inv_sym_log(x, a):
    return np.sign(x) * a * np.exp(np.absolute(x) - 1)


def log_plus_c(x, c):
    return np.log(x + c)


def inv_log_plus_c(x, c):
    return np.exp(x) - c





















class MinMaxScaler():
    '''
    Data scaler for min-max scaling.    
    '''
    
    def __init__(self, axis=(-1,-2)):
        self.axis = axis
        
    def fit(self, data):
        self.min_ = np.nanmin(data, axis=self.axis).squeeze()
        self.max_ = np.nanmax(data, axis=self.axis).squeeze()
        if self.min_.ndim == 0:
            self.min_ = self.min_.item()
            self.max_ = self.max_.item()
        return self.min_, self.max_
    
    def transform(self, data):
        min = np.expand_dims(self.min_, axis=self.axis)
        max = np.expand_dims(self.max_, axis=self.axis)
        # data_scaled = (data - self.min_) / (self.max_ - self.min_)
        data_scaled = (data - min) / (max - min)
        return data_scaled
    
    
    
    
    
    
def minmaxscaling(data, min=None, max=None, axis=(-1,-2)):
    '''
    Scale data using min and max values. If min and max are not provided, they are calculated from the data over the given axis.
    
    '''
    if min is None:
        min = np.nanmin(data, axis=axis)
        min = np.expand_dims(min, axis=axis)
    if max is None:
        max = np.nanmax(data, axis=axis)
        max = np.expand_dims(max, axis=axis)
    data_scaled = ((data - min) / (max - min))
    min = min.squeeze()
    max = max.squeeze()
    if min.ndim == 0:
        min = np.array([min])
        max = np.array([max])
    return data_scaled, min, max


def inv_minmaxscaling(data, minmax):
    '''
    Inverse minmax scaling of data using min and max values.
    Works for one variable only for now!
    '''
    if isinstance(minmax, tuple) or isinstance(minmax, np.ndarray):
        if len(minmax) == 2:
            min, max = minmax
            logging.debug(f'Inverse scaling with {min=} and {max=}')
        else:
            raise ValueError('minmax has to be a tuple of length 2')
    else:
        raise ValueError('minmax has to be a tuple of length 2')

    return data * (max - min) + min