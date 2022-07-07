import time
#import datetime
#import calendar
import numpy as np
import xarray as xr
import pandas as pd
import altair as alt
import streamlit as st

from read_data import read_data
from smoothing import smooth

    

def plot_main(smoothed, methods, col2):
    
    # Create DataFrame
    names = ['smoothed_v', 'smoothed_g', 'smoothed_wcv']
    df = pd.DataFrame(index = smoothed.time.values) 
    df.index.name = 'time'     
    for i,sm in enumerate(methods.keys()):   
        if methods[sm]:
            df[sm] = smoothed[names[i]]
    df = df.reset_index()
    df = df.melt('time', var_name='name', value_name='value')
    
    dfraw = pd.DataFrame(index = smoothed.time.values) 
    dfraw['raw'] = smoothed['band']
    dfraw.index.name = 'time'
    dfraw = dfraw.reset_index()
    dfraw = dfraw.melt('time', var_name='name', value_name='value')
    
    valid = list(methods.values())
    names = np.array(['vcurve', 'garcia', 'wcv'])[valid].tolist()
    colors = np.array(['red', 'blue', 'green'])[valid].tolist()
    
    chart1 = alt.Chart(df).mark_line(opacity=0.6, size = 1).encode(
      x=alt.X('time'),
      y=alt.Y('value'),
      color=alt.Color('name', scale=alt.Scale(
            domain=names,
            range=colors))
      ).properties(title="Smoothing")
      
    chart2 = alt.Chart(dfraw).mark_line(color = 'grey', opacity=0.6, size = 1, point=alt.OverlayMarkDef(opacity = 0.6, size = 10)).encode(
      x=alt.X('time'),
      y=alt.Y('value')
      ).properties(title="Smoothing")
    
    layers = alt.layer(chart1, chart2).configure_area(tooltip = True).interactive()
    
    col2.altair_chart(layers, use_container_width=True)
    

def plot_vcurve(smoothed, methods, col2):
    
    # Create DataFrame
    print(smoothed['curv'][0])
    df = pd.DataFrame(index = smoothed['curv'][0]) 
    df.index.name = 'x'     

    df['curv'] = smoothed['curv'][1]
    df = df.reset_index()
    df = df.melt('x', var_name='name', value_name='value')
    
    # Utils

    chart = alt.Chart(df).mark_line(opacity=0.6).encode(
      x=alt.X('x'),
      y=alt.Y('value')
      ).properties(title="V curve")
    
    col2.altair_chart(chart, use_container_width=True)


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
    
    vcurve = col1.checkbox('V-curve',value=True)
    garcia = col1.checkbox('Garcia',value=True)
    wcv = col1.checkbox('WCV',value=True)
    methods = dict(vcurve = vcurve, garcia = garcia, wcv = wcv)       
    
    col1.markdown('------------')
    
    bound = col1.checkbox('Set bounds to Sopt',value=False)
    if bound:
        sr = col1.slider('S range',-2., 4.2,(-2., 4.2))
    else: 
        sr = None
    
    col1.markdown('------------')
    
    robust = col1.checkbox('Use robust weights',value=True)
    
    col1.markdown('------------')
    
    expec = col1.checkbox('Set a p value',value=True)
    if expec:
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
    
    da = xr.DataArray(np.array(df['NDVI'])*10000, dims = ['time'], coords = dict(time = df['Date']))

    smoothed = smooth(da, vcurve, garcia, wcv, robust, pval_wcv, pval_vc, sr)

    print("--- %s seconds SMOOTH---" % (time.time() - start_time))
    
# =============================================================================
#   Main plot
# =============================================================================
    
    plot_main(smoothed, methods, col2)
    
# =============================================================================
#   V-curve plot
# =============================================================================
    
    if vcurve:
        plot_vcurve(smoothed, methods, col2)



if __name__ == "__main__":
    main()
     