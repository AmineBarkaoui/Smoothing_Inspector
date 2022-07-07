#import calendar
#from datetime import date
#import datetime
from math import log10
#import os
#import re
#import time

#import glob
#import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio

import rioxarray as rx
import pandas as pd
#from tqdm import tqdm
import xarray as xr

# V_curve and WCV smoothing module
import seasmon_xr

# Garcia smoothing functions
from garcia_fns import *


def smooth(da, vcurve, garcia, wcv, srange_bool, robust, p_bool, p_v=0.9, p_wcv=0.8, srange=None):
    
    ds = da.to_dataset(name='band')
    nodata = -3000.
    
    
    
    if vcurve:
        if p_bool:
            p = p_v
        else:
            p = 0.9
            
        if srange_bool:
            ds_smoothed_v = da.hdc.whit.whitsvc(nodata = nodata, srange = srange, p = p)
        else:
            ds_smoothed_v = da.hdc.whit.whitsvc(nodata = nodata, srange = np.arange(-2.,4.2,0.2, dtype=np.float64), p = p)
        ds['smoothed_v'] = ds_smoothed_v.band
        ds['Sopts_v'] = ds_smoothed_v.sgrid
        ds['fits_v'] = ds_smoothed_v.fits
        ds['pens_v'] = ds_smoothed_v.pens
    
    if wcv:
        if p_bool:
            p = p_wcv
        else:
            p = 0.8
            
        if srange_bool:
            ds_smoothed_wcv = da.hdc.whit.whitsgcv(nodata = nodata, srange = srange, p = p, robust = robust)
        else:
            ds_smoothed_wcv = da.hdc.whit.whitsgcv(nodata = nodata, p = p, robust = robust)
        ds['smoothed_wcv'] = ds_smoothed_wcv.band
        ds['Sopts_wcv'] = ds_smoothed_wcv.sgrid
        
    if garcia:
        # Reformat the Data
        z = da.transpose('latitude', 'longitude', 'time')
        z = z.to_dataset(name='band')
        
        # Clean the dataset
        z['band_n'] = replace_bad(z.band)
        
        # Masking out pixels
        z['nan_mask'] = nan_mask(z.band_n)   
        
        if not srange_bool:
            srange = np.arange(-2.,4.2,0.2, dtype=np.float64) 
            
        def Garcia_wrapper(x, mask):
            if not mask:
                # If we want to skip, return -3000 for all timeseries values
                # and the S_opt
                return ((-3000) * np.ones(x.shape[0]), -3000)
            else:
                # Doing the actual smoothing
                g_smooth_rens = Garcia_smoothing_complete(x,
                                                          fit_robust=robust,
                                                          fit_envelope=True,
                                                          neg_residuals_only=True,
                                                          Sopt_Rog=True,
                                                          Sopt_range=srange)
                smoothed_ndvi = g_smooth_rens[0]
                Sopt = g_smooth_rens[1][-1,1]
                return (smoothed_ndvi, Sopt)
  
        Sopts_g = xr.apply_ufunc(Garcia_wrapper,
                                     z.band_n[:, :, :].astype(np.float32),
                                     z.nan_mask[:, :],
                                     input_core_dims=[['time'],[]],
                                     output_core_dims=[['time'], []],
                                     vectorize = True,
                                     dask = 'parallelized',
                                     output_dtypes=[np.float32, np.float32])
        
        ds['smoothed_g'] = Sopts_g[0]
        ds['Sopts_g'] = log10(Sopts_g[1])
        
    return ds