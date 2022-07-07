import datetime
import calendar
import numpy as np
import xarray as xr
import pandas as pd
import altair as alt
import streamlit as st

from read_data import read_data
from smoothing import smooth



def spaghetti_plot2(df, i, c, y1, y2, col2):

    dts = [datetime.datetime.strptime(str(x), '%j').date() for x in df.index.values]
    xticks = []
    for i,dt in enumerate(dts):
        xticks.append(calendar.month_name[dt.month][0:3] + ' ' + str(dt.day))
     
    #df.index = xticks   
    cols = []       
    for y in range(y1, y2+1):
        cols.append(str(y))


    col2.line_chart(df[cols])
    

def spaghetti_plot(df, methods, sr, robust, pval_wcv, pval_vc, col2):

    da = xr.DataArray(np.asarray(df['NDVI'], coords = dict(time = df['Date'])))
    
    smoothed = smooth(da, methods['vcurve'], methods['garcia'], methods['wcv'], robust,
                      sr, pval_wcv, pval_vc)

    cols = []       
    for sm in methods.keys():   
        if methods[sm]:
            df[sm] = smoothed[sm]
            cols.append(sm)

    col2.line_chart(df[cols])


@st.cache  # No need for TTL this time. It's static data :)
def get_data_by_state():
    ndvi_MOD = read_data()
    return ndvi_MOD
    
def main():

    # Layout
    
    st.set_page_config(layout='wide')

    st.title("Smoothing") 

    col1, col, col2 = st.columns([10, 1, 50])     
    
    ## Data
    
    ndvi_MOD = get_data_by_state()
        
    ## Inputs
    
#    loc_list = list(set(ndvi_MOD.index.values))
#    loc_list.sort()
#
#    ct = pd.read_csv('Data/crop_type.csv', index_col = 0, header = 0)
#    ct_dict = ct.to_dict()
#    ct_dict = ct_dict[list(ct_dict.keys())[0]]
#    inv_ct_dict = {v: k for k, v in ct_dict.items()}
#
#    r = pd.read_csv('Data/regions.csv', index_col = 0, header = 0)
#    r_dict = r.to_dict()
#    r_dict = r_dict[list(r_dict.keys())[0]]
#    inv_r_dict = {v: k for k, v in r_dict.items()}



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
            
        
#    col1.markdown('------------')
#    
#    y = col1.slider('Year',2002, 2021,(2002, 2021))
#    c = col1.selectbox('Land Cover', ct['Legend'].values)
#    i = col1.selectbox('Region', r['Adm1Name'].values)
#
#    df_path = 'Data/df/' + inv_ct_dict[c] + '/df_' + str(inv_r_dict[i]) + '.csv'
#    df = pd.read_csv(df_path, index_col = 0)
#    df = df/10000


    #spaghetti_plot2(df, i, c, y[0], y[1], col2)
    
    spaghetti_plot(ndvi_MOD.loc[loc], methods, sr, robust, pval_wcv, pval_vc, col2)


if __name__ == "__main__":
    main()
     