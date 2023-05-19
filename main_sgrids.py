import glob
import xarray as xr
import altair as alt
import pandas as pd

METHODS = ['pvc_unconstrained','pvc_constrained','pwcv_basic','pwcv_robust']


def plot_main(ds):
            
    display(ds.to_dataframe())
    
    
def read_data(tile, ind):
    ds =  xr.open_mfdataset(f'data/sgrids_{ind}/{METHODS[0]}/{tile}', engine='zarr').rename({'sg':METHODS[0]})
    for m in METHODS[1:]:
        ds[m] = xr.open_mfdataset(f'data/sgrids_{ind}/{m}/{tile}', engine='zarr').rename({'sg':m})[m]
    return ds
    
    
@st.cache  # No need for TTL this time. It's static data :)
def get_data_by_state():
    tiles = glob.glob('data/sgrids_ndvi/pwcv_basic/*')
    ndvi_tiles = zip(tiles, [read_data(tile, 'ndvi') for tile in tiles])
    tda_tiles = zip(tiles, [read_data(tile, 'tda') for tile in tiles])
    tna_tiles = zip(tiles, [read_data(tile, 'tna') for tile in tiles])
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
        
    ndvi_tiles, tda_tiles, tna_tiles = get_data_by_state(choose)
        
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
    
    plot_main(ds_to_plot)
    

if __name__ == "__main__":
    main()