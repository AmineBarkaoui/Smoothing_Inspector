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

    

def plot_main(smoothed, methods):
    
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
    
    chart1 = alt.Chart(df, height=400).mark_line(opacity=0.8, size = 1).encode(
      x=alt.X('time'),
      y=alt.Y('value'),
      color=alt.Color('name', scale=alt.Scale(
            domain=names,
            range=colors))
      ).properties(title="Smoothing")
      
    chart2 = alt.Chart(dfraw, height=400).mark_point(color = 'grey', opacity=0.6, size = 7, point=alt.OverlayMarkDef(opacity = 0.6, size = 10)).encode(
      x=alt.X('time'),
      y=alt.Y('value')
      ).properties(title="Smoothing")
    
    layers = alt.layer(chart1, chart2).configure_area(tooltip = True).interactive()
    
    st.altair_chart(layers, use_container_width=True)
    

def plot_vcurve(smoothed, methods, srange, col):
    
    # Create DataFrame
    df = pd.DataFrame(index = srange[:-1]) 
    df.index.name = 'Sopt'     

    df['curv'] = smoothed['curv'].values[:-1]
    df = df.reset_index()
    df = df.melt('Sopt', var_name='name', value_name='value')
    
    # Utils

    chart = alt.Chart(df).mark_line(opacity=0.6).encode(
      x=alt.X('Sopt'),
      y=alt.Y('value')
      ).properties(title="V curve")
    

    #sopt = alt.Chart(df.loc[smoothed.Sopts_v.values]).mark_rule().encode(
    #    x = alt.X('Sopt')
    #)
    
    
    col.altair_chart(chart, use_container_width=False)


def print_sopt(smoothed, methods, col):

    with col:
    	if methods['vcurve']:
    		st.write('Sopt Vcurve: ', str(smoothed['Sopts_v'].values))
    	if methods['garcia']:
    		st.write('Sopt Garcia: ', str(smoothed['Sopts_g'].values))
    	if methods['wcv']:
    	    st.write('Sopt WCV: ', str(smoothed['Sopts_wcv'].values))


@st.cache  # No need for TTL this time. It's static data :)
def get_data_by_state():
    ndvi_MXD = read_data()
    return ndvi_MXD
    
def main():

# =============================================================================
#   Layout
# =============================================================================
    
    st.set_page_config(layout='wide')


    st.title("Smoothing Inspector") 

    # st, col, st = st.columns([10, 1, 40])  

# =============================================================================
#    Data 
# =============================================================================
    
    
    ndvi_MXD = get_data_by_state()


    loc_list = list(set(ndvi_MXD.index.values))
    loc_list.sort()
    

# =============================================================================
#   Widgets inputs  
# =============================================================================
    
    with st.sidebar:
    
	    
	    loc = st.selectbox('Location', loc_list)
	    
	    st.markdown('------------')
	    
	    vcurve = st.checkbox('V-curve',value=True)
	    garcia = st.checkbox('Garcia',value=True)
	    wcv = st.checkbox('WCV',value=True)
	    methods = dict(vcurve = vcurve, garcia = garcia, wcv = wcv)       
	    
	    st.markdown('------------')
	    
	    bound = st.checkbox('Set bounds to Sopt',value=False)
	    if bound:
	        sr = st.slider('S range',-2., 4.2,(-2., 4.2))
	    else: 
	        sr = None
	    
	    st.markdown('------------')
	    
	    robust = st.checkbox('Use robust weights',value=True)
	    
	    st.markdown('------------')
	    
	    expec = st.checkbox('Set a p value',value=True)
	    if expec:
	        pval_wcv = st.select_slider('Select a p value for the WCV',
	                                options=[0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.],
	                                value = 0.8)
	        pval_vc = st.select_slider('Select a p value for the V-curve',
	                                options=[0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.],
	                                value = 0.9)
	    else:
	        pval_wcv = None
	        pval_vc = None
            
# =============================================================================
#   Smoothing
# =============================================================================
    
    start_time = time.time()
    
    df = ndvi_MXD.loc[loc]
    
    da = xr.DataArray(np.array(df['NDVI'])*10000, dims = ['time'], coords = dict(time = df['Date']))
    
    if sr==None:
        srange = np.arange(-2, 4.2, 0.2, dtype=np.float64)
    else:
        srange = np.arange(sr[0], sr[1], 0.2, dtype=np.float64)
        
    smoothed = smooth(da, vcurve, garcia, wcv, robust, pval_wcv, pval_vc, srange)
    
    print("--- %s seconds SMOOTH---" % (time.time() - start_time))
    
# =============================================================================
#   Main plot
# =============================================================================
    
    plot_main(smoothed, methods)
    
# =============================================================================
#   Print Sopts
# =============================================================================
    
    col1, col2 = st.columns([40, 40])  

    print_sopt(smoothed, methods, col1)
    
# =============================================================================
#   V-curve plot
# =============================================================================
    
    if vcurve:
        plot_vcurve(smoothed, methods, srange, col2)



if __name__ == "__main__":
    main()
     