import time
#import datetime
#import calendar
import numpy as np
import xarray as xr
import pandas as pd
#import altair as alt
import streamlit as st

from read_data import read_data
from smoothing import smooth

    

def main_plot(smoothed, methods, col2):

    
#    if sr != None:
#    	srange = np.arange(sr[0], sr[1], 0.2)
#    	srange_bool = True
#    	smoothed = smooth(da, methods['raw'], methods['garcia'], methods['vcurve'], methods['wcv'], 
#    					  srange_bool, robust, pval_wcv, pval_vc, srange)
#    else:
#    	srange_bool = False 
#    	smoothed = smooth(da, methods['raw'], methods['garcia'], methods['vcurve'], methods['wcv'], 
#    					  srange_bool, robust, pval_wcv, pval_vc)

    # # FOR NOW
    # smoothed = da.to_dataset(name = 'raw')
    # smoothed['vcurve'] = da
    # smoothed['garcia'] = da
    # smoothed['wcv'] = da
    # ######################

    # ORDER: Raw, vcurve, garcia, wcv
    names = ['band', 'smoothed_v', 'smoothed_g', 'smoothed_wcv']
    dfp = pd.DataFrame(index = smoothed.Date.values)      
    for i,sm in enumerate(methods.keys()):   
        if methods[sm]:
            dfp[sm] = smoothed[names[i]]


    col2.line_chart(dfp)




@st.cache  # No need for TTL this time. It's static data :)
def get_data_by_state():
    ndvi_MOD = read_data()
    return ndvi_MOD
    
def main():

# =============================================================================
#   Layout
# =============================================================================
    
    st.set_page_config(layout='wide')


    st.title("Smoothing Inspector") 

    col1, col, col2 = st.columns([10, 1, 40])  

# =============================================================================
#    Data 
# =============================================================================
    
    
    start_time = time.time()
    ndvi_MOD = get_data_by_state()
    print("--- %s seconds READ DATA---" % (time.time() - start_time))

    loc_list = list(set(ndvi_MOD.index.values))
    loc_list.sort()
    

# =============================================================================
#   Widgets inputs  
# =============================================================================
    
    
    col1.subheader("Inputs")
    
    loc = col1.selectbox('Location', loc_list)
    
    col1.markdown('------------')
    
    raw = True #col1.checkbox('Raw',value=True)
    vcurve = col1.checkbox('V-curve',value=True)
    garcia = col1.checkbox('Garcia',value=False)
    wcv = col1.checkbox('WCV',value=True)
    methods = dict(raw = raw, vcurve = vcurve, garcia = garcia, wcv = wcv)
        
    col1.markdown('------------')
    
    bound = col1.checkbox('Set bounds to Sopt',value=False)
    if bound:
        sr = col1.slider('S range',-2., 4.2,(-2., 4.2))
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
        pval_vc = col1.select_slider('Select a p value for the V-curve',
                                options=[0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.],
                                value = 0.9)
    else:
        pval_wcv = None
        pval_vc = None
            
# =============================================================================
#   Smoothing
# =============================================================================
    
    start_time = time.time()
    
    df = ndvi_MOD.loc[loc]
    
    da = xr.DataArray(np.array(df['NDVI']), dims = ['time'], coords = dict(time = df['Date']))
        
    smoothed = smooth(da, vcurve, garcia, wcv, robust, pval_wcv, pval_vc, sr)

    print(smoothed)
    print("--- %s seconds SMOOTH---" % (time.time() - start_time))
    
# =============================================================================
#   Main plot
# =============================================================================
    
    main_plot(smoothed, methods, col2)



if __name__ == "__main__":
    main()
     