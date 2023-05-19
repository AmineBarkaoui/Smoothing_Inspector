import glob
import xarray as xr
import altair as alt
import pandas as pd
import streamlit as st

from streamlit_option_menu import option_menu

METHODS = ['pvc_unconstrained','pvc_constrained','pwcv_basic','pwcv_robust']

TITLES = dict(zip(METHODS, ['Unconstrained V-curve', 'Constrained V-curve', 'Basic WCV', 'Robust WCV']))

def plot_main(df):
    
    df = df.drop('spatial_ref', axis=1).reset_index()
    df = df.melt(id_vars=['longitude','latitude'])

    df.columns = ['longitude','latitude','method','sopt']

    df['method'] = [TITLES[m] for m in df.method.values]

    chart = alt.Chart(df).mark_point().encode(
        x='longitude:Q',
        y='latitude:Q',
        color='sopt:Q'
    ).properties(
        width=180,
        height=180
    ).facet(
        facet='method:N',
        columns=2
    )

    st.altair_chart(chart, use_container_width=True)
    

def read_data(tile, ind):
    try:
        ds =  xr.open_mfdataset(f'data/sgrids_{ind}/{METHODS[0]}/{tile}', engine='zarr').rename({'sg':METHODS[0]})
        for m in METHODS[1:]:
            ds[m] = xr.open_mfdataset(f'data/sgrids_{ind}/{m}/{tile}', engine='zarr').rename({'sg':m})[m]
        return ds
    except: 
        return None
    
    
@st.cache_resource  # No need for TTL this time. It's static data :)
def get_data_by_state():
    tiles = glob.glob('data/sgrids_ndvi/pwcv_basic/*')
    tiles = sorted([p.split('\\')[-1] for p in tiles])

    ndvi_tiles = dict(zip(tiles, [read_data(tile, 'ndvi') for tile in tiles]))
    tda_tiles = dict(zip(tiles, [read_data(tile, 'tda') for tile in tiles]))
    tna_tiles = dict(zip(tiles, [read_data(tile, 'tna') for tile in tiles]))
    return ndvi_tiles, tda_tiles, tna_tiles


def main():

# =============================================================================
#   Layout
# =============================================================================
    
    st.set_page_config(layout='wide')

    st.title("S grids inspector") 

# =============================================================================
#    Data 
# =============================================================================
        
    ndvi_tiles, tda_tiles, tna_tiles = get_data_by_state()
        
    tiles = list(ndvi_tiles.keys())
    
# =============================================================================
#   Widgets inputs  
# =============================================================================
    
    with st.sidebar:
        
        tile = st.selectbox('Tile', tiles)
	    
        st.markdown('------------')
	    
        index = option_menu("Choose index", ["NDVI", "TDA", "TNA"],
             icons=['tree','thermometer-half','thermometer-half'],
             menu_icon="app-indicator", default_index=0,
             orientation='vertical')
            
# =============================================================================
#   Selection of dataset
# =============================================================================
    
    all_ds_indexes = dict(NDVI = ndvi_tiles, TDA = tda_tiles, TNA = tna_tiles)
    
    ds_to_plot = all_ds_indexes[index][tile]
    
# =============================================================================
#   Main plot
# =============================================================================
    
    plot_main(ds_to_plot.to_dataframe())
    

if __name__ == "__main__":
    main()