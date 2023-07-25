import time
#import datetime
#import calendar
import numpy as np
import xarray as xr
import pandas as pd
import altair as alt
import streamlit as st
import geopandas as gpd
import contextily as cx
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu

from read_data import read_data
from smoothing import smooth
from shapely.geometry import Point
    

def plot_main(smoothed, methods, choose, nodata):
    
    # Set scalling factors
    if choose == 'NDVI':
        coeff = 0.0001 ; offset = 0.
    else:
        coeff = 0.02 ; offset = -273.15
        
    # Create DataFrame
    names = ['smoothed_v', 'smoothed_wcv'] # 'smoothed_g'
    df = pd.DataFrame(index = smoothed.time.values) 
    df.index.name = 'time'     
    for i,sm in enumerate(methods.keys()):   
        if methods[sm]:
            df[sm] = smoothed[names[i]]*coeff + offset
    df = df.reset_index()
    df = df.melt('time', var_name='name', value_name='value')
    
    dfraw = pd.DataFrame(index = smoothed.time.values) 
    raw = smoothed['band']*coeff + offset
    if choose == 'LST': raw = np.maximum(raw,0) 
    dfraw['raw'] = raw
    dfraw.index.name = 'time'
    dfraw = dfraw.reset_index()
    dfraw = dfraw.melt('time', var_name='name', value_name='value')
    dfraw[dfraw.value == nodata] = np.nan
    
    valid = list(methods.values())
    names = np.array(['vcurve', 'wcv'])[valid].tolist() # 'garcia', 
    colors = np.array(['red', 'green'])[valid].tolist() # 'blue',
    
    brush = alt.selection_interval(encodings=['x'])

    chart1 = alt.Chart(df).mark_line(opacity=0.8, size = 1).encode(
        x=alt.X('time'),
        y=alt.Y('value'),
        color=alt.Color('name', scale=alt.Scale(
            domain=names,
            range=colors))
        )
      
    chart2 = alt.Chart(dfraw).mark_line(color = 'grey', opacity=0.6, size = 1, point=alt.OverlayMarkDef(opacity = 0.6, size = 10)).encode(
      x=alt.X('time'),
      y=alt.Y('value')
      )
        
    main_layers = alt.layer(chart1, chart2).add_selection(brush).properties(width=750, height=20, title='Select an interval to zoom in')
    #.configure_area(tooltip = True)#.interactive()

    subset_layers = alt.layer(chart1, chart2).encode(x=alt.X(scale={'domain':brush.ref()})).properties(width=750)
    
    layers = alt.vconcat(main_layers, subset_layers).configure_title(anchor='start')
    
    st.altair_chart(layers, use_container_width=True)
    
    
def plot_lta(smoothed, methods, col, choose):
    
    # Set scalling factors
    if choose == 'NDVI':
        coeff = 0.0001 ; offset = 0.
    else:
        coeff = 0.02 ; offset = -273.15
    
    # Create DataFrame
    names = ['smoothed_v', 'smoothed_wcv']
    df = pd.DataFrame(index = np.arange(1,13,1))
    df.index.name = 'month'     
    for i,sm in enumerate(methods.keys()):  
        if methods[sm]:
            da = xr.DataArray(smoothed[names[i]], dims = ['time'], 
              coords = dict(time = pd.to_datetime(smoothed.time.values).month.values))
            lta = da.groupby('time').mean(dim='time')
            df[sm] = lta.values*coeff + offset
    df = df.reset_index()
    df = df.melt('month', var_name='name', value_name='value')
    
    raw = xr.DataArray(smoothed['band'], dims = ['time'], 
                      coords = dict(time = pd.to_datetime(smoothed.time.values).month.values))
    if choose == 'NDVI':
        nodata = -3000.
    else:
        nodata = 0.
    raw = raw.where(raw!=nodata)
    
    lta = raw.groupby('time').mean(dim='time', skipna = True)
    
    dfraw = pd.DataFrame(index = lta.time.values) 
    lta_raw = lta.values*coeff + offset
    if choose == 'LST': lta_raw = np.maximum(lta_raw,0) 
    dfraw['raw'] = lta_raw
    dfraw.index.name = 'month'
    dfraw = dfraw.reset_index()
    dfraw = dfraw.melt('month', var_name='name', value_name='value')
    
    std_lta = raw*coeff - offset
    std_lta = std_lta.groupby('time').std(dim='time', skipna = True)
    
    dfstd = pd.DataFrame(index = std_lta.time.values) 
    dfstd['std'] = std_lta.values
    dfstd.index.name = 'month'
    dfstd = dfstd.reset_index()
    dfstd = dfstd.melt('month', var_name='name', value_name='value')
    
    dfstd['lower'] = dfraw['value']-dfstd['value']
    dfstd['upper'] = dfraw['value']+dfstd['value']

    valid = list(methods.values())
    names = np.array(['vcurve', 'wcv'])[valid].tolist()
    colors = np.array(['red', 'green'])[valid].tolist()
    
    chart1 = alt.Chart(df).mark_line(opacity=0.8, size = 1).encode(
      x=alt.X('month'),
      y=alt.Y('value'),
      color=alt.Color('name', scale=alt.Scale(
            domain=names,
            range=colors))
      ).properties(title="Long Term Average")
      
    chart2 = alt.Chart(dfraw).mark_line(color = 'grey', opacity=0.6, size = 1, point=alt.OverlayMarkDef(opacity = 0.6, size = 10)).encode(
      x=alt.X('month'),
      y=alt.Y('value')
      ).properties(title="Long Term Average")
    
    chart3 = alt.Chart(dfstd).mark_area(color='#17becf',opacity=0.2).encode(
      x=alt.X('month'),
      y='lower',
      y2='upper'
      ).properties()
    
    layers = alt.layer(chart3, chart1, chart2).configure_area(tooltip = True).interactive()
    layers.layer[0].encoding.y.title = f'{choose} average'

    col.altair_chart(layers, use_container_width=True)
    

def plot_year(smoothed, methods, col, choose, year, start_month):
    
    leng = len(smoothed.sel(time=pd.to_datetime(smoothed.time.values).year == year).time)

    # Set scalling factors
    if choose == 'NDVI':
        coeff = 0.0001 ; offset = 0.
    else:
        coeff = 0.02 ; offset = -273.15
    
    # Create DataFrame
    names = ['smoothed_v', 'smoothed_wcv']
    df = pd.DataFrame() #(index = np.arange(1,leng+1,1))
    df.index.name = 'month'     
    for i,sm in enumerate(methods.keys()):  
        if methods[sm]:
            da = xr.DataArray(smoothed[names[i]], dims = ['time'], 
              coords = dict(time = pd.to_datetime(smoothed.time.values)))
            day = da.where(da.time.dt.year >= year, drop=True)
            day = day.where(day.time.dt.month >= start_month, drop=True)
            day = day.isel(time=slice(0, leng))
            df[sm] = day.values * coeff + offset
    df = df.reset_index()
    df.month = day.time.values
    df = df.melt('month', var_name='name', value_name='value')
       
    raw = xr.DataArray(smoothed['band'], dims = ['time'], 
                      coords = dict(time = pd.to_datetime(smoothed.time.values)))
    if choose == 'NDVI':
        nodata = -3000.
    else:
        nodata = 0.
    raw = raw.where(raw!=nodata)
    rawy = raw.where(raw.time.dt.year >= year, drop=True)
    rawy = rawy.where(rawy.time.dt.month >= start_month, drop=True)
    rawy = rawy.isel(time=slice(0, leng))
    
    dfraw = pd.DataFrame(index = rawy.time.values) 
    rawy = rawy.values*coeff + offset
    if choose == 'LST': rawy = np.maximum(rawy,0) 
    dfraw['raw'] = rawy
    dfraw.index.name = 'month'
    dfraw = dfraw.reset_index()
    dfraw = dfraw.melt('month', var_name='name', value_name='value')
  
    valid = list(methods.values())
    names = np.array(['vcurve', 'wcv'])[valid].tolist()
    colors = np.array(['red', 'green'])[valid].tolist()

    chart1 = alt.Chart(df).mark_line(opacity=0.8, size = 1).encode(
      x=alt.X('month'),
      y=alt.Y('value'),
      color=alt.Color('name', scale=alt.Scale(
            domain=names,
            range=colors))
      ).properties(title=f"{choose} - {start_month}/{year} to {start_month}/{year+1}")
      
    chart2 = alt.Chart(dfraw).mark_line(color = 'grey', opacity=0.6, size = 1, point=alt.OverlayMarkDef(opacity = 0.6, size = 10)).encode(
      x=alt.X('month'),
      y=alt.Y('value')
      ).properties(title=f"{choose} - {start_month}/{year} to {start_month}/{year+1}")
    
    layers = alt.layer(chart1, chart2).configure_area(tooltip = True).interactive()
    layers.layer[0].encoding.y.title = f'{choose} year'

    col.altair_chart(layers, use_container_width=True)


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
      y=alt.Y('value', axis=alt.Axis(title='V'))
      ).properties(title="V curve")
    

    #sopt = alt.Chart(df.loc[smoothed.Sopts_v.values]).mark_rule().encode(
    #    x = alt.X('Sopt')
    #)
    
    
    col.altair_chart(chart, use_container_width=False)


def plot_location(da):
    
    # Create DataFrame
    df = da.to_dataframe(name='raw').reset_index().loc[:2]
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")

    margin = 20
    extent = (gdf.longitude[0] - margin, gdf.longitude[0] + margin, gdf.latitude[0] - margin, gdf.latitude[0] + margin) 
    chart = gdf.plot()
    chart.axis(extent)
    plt.axis('off')
    cx.add_basemap(chart, source=cx.providers.OpenStreetMap.Mapnik, crs=gdf.crs)

    st.pyplot(chart.figure)


def print_sopt(smoothed, methods, col):

    df = pd.DataFrame()
    indexes = []

    with col:
        if methods['vcurve']:
            data = pd.DataFrame({"Selected Sopt": [str(smoothed['Sopts_v'].values)]})
            df = pd.concat([df, data])
            indexes.append('Vcurve')
        if methods['wcv']:
            data = pd.DataFrame({"Selected Sopt": [str(smoothed['Sopts_wcv'].values)]})
            df = pd.concat([df, data])
            indexes.append('WCV')
    df.index = indexes
    col.table(df)


def print_rmse(smoothed, choose, methods, col):

    def rmse(da1, da2):
        rmse = (((da1 - da2) ** 2).sum('time') / da1.time.size) ** (1/2)
        rmse_masked = rmse.where(~da2.isel(time=0).isnull(), np.nan)
        return float(rmse_masked.values)
    
    if choose == 'NDVI':
        coeff = 0.0001 ; offset = 0.
    else:
        coeff = 0.02 ; offset = -273.15

    df = pd.DataFrame()
    indexes = []

    with col:
        if methods['vcurve']:
            data = pd.DataFrame({"RMSE": [str(round(rmse(
                smoothed['smoothed_v'] * coeff + offset, 
                smoothed['band'] * coeff + offset,
            ), 5))]})
            df = pd.concat([df, data])
            indexes.append('Vcurve')
        if methods['wcv']:
            data = pd.DataFrame({"RMSE": [str(round(rmse(
                smoothed['smoothed_wcv'] * coeff + offset, 
                smoothed['band'] * coeff + offset,
            ), 5))]})
            df = pd.concat([df, data])
            indexes.append('WCV')

    df.index = indexes
    col.table(df)
    


@st.cache  # No need for TTL this time. It's static data :)
def get_data_by_state(choose):
    product_MXD, names_grid_sample = read_data(choose)    
    return product_MXD, names_grid_sample
    

def main():

# =============================================================================
#   Layout
# =============================================================================
    
    st.set_page_config(layout='wide')

    choose = option_menu(None, ["NDVI", "LST"],
                         icons=['tree','thermometer-half'],
                         #menu_icon="app-indicator", default_index=0,
                         orientation='horizontal',
                         styles={
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": "#FFBE4D", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#89EE9E"},})

    st.title("Smoothing Inspector") 

# =============================================================================
#    Data 
# =============================================================================
        
    product_MXD, names_grid_sample = get_data_by_state(choose)
        
    loc_list = [*names_grid_sample, *np.unique(product_MXD.index.values)]
    loc_list.sort()
    
# =============================================================================
#   Widgets inputs  
# =============================================================================
    
    with st.sidebar:
        
        loc = st.selectbox('Location', loc_list)

        map = wcv = st.checkbox('Show point on map',value=False)

        st.markdown('------------')
        
        vcurve = st.checkbox('V-curve',value=True)
        #garcia = st.checkbox('Garcia',value=True)
        wcv = st.checkbox('WCV',value=True)
        methods = dict(vcurve = vcurve, wcv = wcv) # garcia = garcia,
        
        st.markdown('------------')

        year = st.slider('Year', 2002, 2022, 2015)
        
        start_month = st.slider('Starting month', 1, 12, 1)

        st.markdown('------------')
        
        
        bound = st.checkbox('Set bounds to Sopt',value=False)
        if bound:
            sr = st.slider('S range',-2., 4.2,(-2., 4.2))
        else: 
            sr = None
        
        st.markdown('------------')
        
        robust = st.checkbox('Use robust weights',value=False)
        
        st.markdown('------------')
        
        expec = st.checkbox('Set a p value',value=True)
        if expec:
            pval_wcv = st.select_slider('Select a p value for the WCV',
                                    options=[0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.],
                                    value = 0.8)
            if choose == 'NDVI':
                    # p at 0.9 by default for the V-curve on NDVI
                    pval_vc = st.select_slider('Select a p value for the V-curve',
                                    options=[0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.],
                                    value = 0.9)
            else:
                    # p at 0.8 by default for the V-curve on LST
                    pval_vc = st.select_slider('Select a p value for the V-curve',
                                    options=[0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.],
                                    value = 0.8)
                
        else:
            pval_wcv = None
            pval_vc = None
            
# =============================================================================
#   Smoothing
# =============================================================================
    
    start_time = time.time()
    
    if loc in names_grid_sample:

        translate_product = dict(NDVI = 'vim', LST = 'tda')
        da = xr.open_zarr(f'data/{translate_product[choose]}_zarr').band.load()

        lat = int(loc.split(',')[0].split('(')[-1])
        lon = int(loc.split(',')[-1].split(')')[0])

        da = da.sel(latitude = lat, longitude = lon, method = 'nearest')
        nodata = da.nodata

    else:

        df = product_MXD.loc[loc]
        
        if choose == 'NDVI':
            nodata = -3000.
            da = xr.DataArray(np.array(df['NDVI'])*10000, dims = ['time'], coords = dict(time=df['Date']))
            da = da.assign_coords(longitude=('time', df['Longitude']))
            da = da.assign_coords(latitude=('time', df['Latitude']))
        else:
            nodata = 0.
            da = xr.DataArray(np.array(df['LST'])*50, dims = ['time'], coords = dict(time = df['Date']))
            da = da.assign_coords(longitude=('time', df['Longitude']))
            da = da.assign_coords(latitude=('time', df['Latitude']))
        
    if sr==None:
        srange = np.arange(-2, 4.2, 0.2, dtype=np.float64)
    else:
        srange = np.arange(sr[0], sr[1], 0.2, dtype=np.float64)

    if not(map): 
        smoothed = smooth(da, vcurve, wcv, robust, pval_vc, pval_wcv, srange, nodata, choose)

        print("--- %s seconds SMOOTH---" % (time.time() - start_time))
    
# =============================================================================
#   Main plot
# =============================================================================
    
        plot_main(smoothed, methods, choose, nodata)
    
# =============================================================================
#   Long Term Average
# =============================================================================
        col1, col2 = st.columns([40, 40]) 
    
        plot_lta(smoothed, methods, col1, choose)
    
# =============================================================================
#   Print Sopts
# =============================================================================
    
        print_sopt(smoothed, methods, col1) 
    
# =============================================================================
#   V-curve plot
# =============================================================================
    
        #if vcurve:
            #plot_vcurve(smoothed, methods, srange, col2)
        plot_year(smoothed, methods, col2, choose, year, start_month)

# =============================================================================
#   Print RMSE
# =============================================================================
    
        print_rmse(smoothed, choose, methods, col2) 

# =============================================================================
#   Location plot
# =============================================================================
    
    if map: 
        plot_location(da)


if __name__ == "__main__":
    main()
     