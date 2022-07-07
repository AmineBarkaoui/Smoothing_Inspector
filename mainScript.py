import datetime
import calendar
import numpy as np
import xarray as xr
import pandas as pd
import altair as alt
import streamlit as st
import time

from read_data import read_data
from smoothing import smooth

    

def main_plot(df, methods, sr, robust, pval_wcv, pval_vc, col2):

    da = xr.DataArray(np.asarray(df['NDVI']), coords = dict(time = df['Date']))

    start_time = time.time()
    if sr != None:
    	srange = np.arange(sr[0], sr[1], 0.2)
    	srange_bool = True
    	smoothed = smooth(da, methods['raw'], methods['garcia'], methods['vcurve'], methods['wcv'], 
    					  srange_bool, robust, pval_wcv, pval_vc, srange)
    else:
    	srange_bool = False 
    	smoothed = smooth(da, methods['raw'], methods['garcia'], methods['vcurve'], methods['wcv'], 
    					  srange_bool, robust, pval_wcv, pval_vc)
    print("--- %s seconds SMOOTH---" % (time.time() - start_time))

    # # FOR NOW
    # smoothed = da.to_dataset(name = 'raw')
    # smoothed['vcurve'] = da
    # smoothed['garcia'] = da
    # smoothed['wcv'] = da
    # ######################

    # ORDER: Raw, vcurve, garcia, wcv
    names = ['band', 'smoothed_v', 'smoothed_g', 'smoothed_wcv']
    dfp = pd.DataFrame(index = df['Date'].values)      
    for i,sm in enumerate(methods.keys()):   
        if methods[sm]:
            dfp[sm] = smoothed[names[i]]


    col2.line_chart(dfp)



    
def main():

    st.set_page_config(layout='wide')

    st.title("Smoothing Inspector") 

    col1, col, col2 = st.columns([10, 1, 40])  


    
    ## Data
    
    start_time = time.time()
    ndvi_MOD, ndvi_MYD, ndvi_MXD = read_data()
    print("--- %s seconds READ DATA---" % (time.time() - start_time))


        
    ## Inputs
    
    loc_list = list(set(ndvi_MOD.index.values))
    loc_list.sort()



    # Widgets
    
    col1.subheader("Inputs")
    
    loc = col1.selectbox('Location', loc_list)
    
    col1.markdown('------------')
    
    raw = col1.checkbox('Raw',value=True)
    vcurve = col1.checkbox('V-curve',value=True)
    garcia = col1.checkbox('Garcia',value=True)
    wcv = col1.checkbox('WCV',value=True)
    methods = dict(raw = raw, vcurve = vcurve, garcia = garcia, wcv = wcv)
        
    col1.markdown('------------')
    
    bound = col1.checkbox('Set bounds to Sopt',value=False)
    if bound:
        sr = col1.slider('S range',-1.8, 4.2,(-1.8, 4.2))
    else: 
        sr = None
    
    col1.markdown('------------')
    
    robust = col1.checkbox('Use robust weights',value=True)
    
    col1.markdown('------------')
    
    expec_wcv = col1.checkbox('Set a p value - WCV',value=True)
    if expec_wcv:
        pval_wcv = col1.select_slider('Select a p value for the WCV',
                                options=[0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.],
                                value = 0.8)
    else:
        pval_wcv = None
    
    expec_vc = col1.checkbox('Set a p value - V curve',value=True)
    if expec_vc:
        pval_vc = col1.select_slider('Select a p value for the V-curve',
                                options=[0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.],
                                value = 0.9)
    else:
        pval_vc = None
        
    col1.markdown('------------')
    
    
    df = ndvi_MOD
    main_plot(df.loc[loc], methods, sr, robust, pval_wcv, pval_vc, col2)


if __name__ == "__main__":
    main()
     