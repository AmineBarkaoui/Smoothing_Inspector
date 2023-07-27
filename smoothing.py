import numpy as np
import xarray as xr
from math import log10

# V_curve and WCV smoothing module
import seasmon_xr

# Garcia smoothing functions
from garcia_fns import *


def smooth(da, vcurve, wcv, robust, p_v, p_wcv, srange=None, ac=None, nodata = -3000., choose='NDVI'):
    
    ds = da.to_dataset(name='band')

    if ac: 
        lc = da.expand_dims(dim={"x": 1, "y": 1}).hdc.algo.autocorr().squeeze(['x', 'y'])
    else: 
        lc = None

    if vcurve:
        if lc == None:
            if p_v != None:
                ds_smoothed_v = da.hdc.whit.whitsvc(nodata = nodata, srange = srange, p = p_v)
            else:
                ds_smoothed_v = da.hdc.whit.whitsvc(nodata = nodata, srange = srange)            
        else:
            if p_v != None:
                ds_smoothed_v = da.hdc.whit.whitsvc(nodata = nodata, srange = srange, p = p_v)
                #ds_smoothed_v = da.hdc.whit.whitsvc(nodata = nodata, lc = lc, p = p_v)
            else:
                ds_smoothed_v = da.hdc.whit.whitsvc(nodata = nodata, srange = srange)
                #ds_smoothed_v = da.hdc.whit.whitsvc(nodata = nodata, lc = lc) 

        ds['smoothed_v'] = ds_smoothed_v.band
        ds['Sopts_v'] = ds_smoothed_v.sgrid
        ds['curv'] = ds_smoothed_v.curv
    
    if wcv:
        if p_wcv != None:
            ds_smoothed_wcv = da.hdc.whit.whitsgcv(nodata = nodata, srange = srange, p = p_wcv, robust = robust)
        else:
            ds_smoothed_wcv = da.hdc.whit.whitsgcv(nodata = nodata, srange = srange, robust = robust)

        ds['smoothed_wcv'] = ds_smoothed_wcv.band
        ds['Sopts_wcv'] = ds_smoothed_wcv.sgrid
        
    #if garcia:
    #    # Reformat the Data
    #    z = da.to_dataset(name='band')
    #    
    #    # Clean the dataset
    #    z['band_n'] = replace_bad(z.band, choose)
    #
    #    g_smooth_rens = Garcia_smoothing_complete(z.band_n.values.astype(np.float32),
    #                                              fit_robust=robust,
    #                                              fit_envelope=(p_wcv!=None),
    #                                              neg_residuals_only=True,
    #                                              Sopt_Rog=True,
    #                                              Sopt_range=srange)
    #    ds['smoothed_g'] = np.int16(g_smooth_rens[0])
    #    ds['Sopts_g'] = np.int16(g_smooth_rens[1][-1,1])
        
        
    return ds