import pandas as pd
import xarray as xr

from itertools import product

from helpers import *

def read_ndvi():
    # loading ndvi_MOD data from csv
    ndvi_MOD = pd.read_csv(
        'data/MOD13A2-MOD13A2-006-results.csv', 
        index_col=0, 
        usecols = ['ID', 
                   'Latitude',
                   'Longitude',
                   'Date', 
                   'MOD13A2_006__1_km_16_days_NDVI', 
                   'MOD13A2_006__1_km_16_days_composite_day_of_the_year'],
        dtype = {'MOD13A2_006__1_km_16_days_composite_day_of_the_year': int})
    
    
    #renaming the columns
    ndvi_MOD = ndvi_MOD.rename(columns={"MOD13A2_006__1_km_16_days_NDVI": "NDVI", 
                            "MOD13A2_006__1_km_16_days_composite_day_of_the_year": "Composite_date"})
    
    
    # Convert string Date to datetime.date
    ndvi_MOD['Date'] = ndvi_MOD['Date'].apply(fromstring)
    
    
    # Convert composite_date from julian to datetime
    # Add a True/False column to keep track of the values with the same composite_date
    
    compo = []
    same_compo = []
    
    for i in range(len(ndvi_MOD)):
        
        d = ndvi_MOD['Date'][i]
        c = ndvi_MOD['Composite_date'][i]
        
        if c != -1:    
            if (d.month == 12) and (c<20):
                compo.append(fromjulian(str(d.year+1)+str(c)))
            else:
                compo.append(fromjulian(str(d.year)+str(c)))
        else:
            # nodata so we don't care about the date
            compo.append(d)
        
        if (i==0):
            same_compo.append(False)
        elif (compo[i]==compo[i-1]):
            same_compo.append(True)
        else:
            same_compo.append(False)
                
    ndvi_MOD['Composite_date'] = compo   
    ndvi_MOD['Same_composite'] = same_compo 
    
#=============================================================================#
    
    # loading MYD data from csv
    ndvi_MYD = pd.read_csv(
        'data/MYD13A2-MYD13A2-006-results.csv', 
        index_col=0, 
        usecols = ['ID', 
                   'Latitude',
                   'Longitude',
                   'Date', 
                   'MYD13A2_006__1_km_16_days_NDVI', 
                   'MYD13A2_006__1_km_16_days_composite_day_of_the_year'],
        dtype = {'MYD13A2_006__1_km_16_days_composite_day_of_the_year': int})
    
    
    #renaming the columns
    ndvi_MYD = ndvi_MYD.rename(columns={"MYD13A2_006__1_km_16_days_NDVI": "NDVI", 
                            "MYD13A2_006__1_km_16_days_composite_day_of_the_year": "Composite_date"})
    
    
    # Convert string Date to datetime.date
    ndvi_MYD['Date'] = ndvi_MYD['Date'].apply(fromstring)
    
    
    # Convert composite_date from julian to datetime
    # Add a True/False column to keep track of the values with the same composite_date
    
    compo = []
    same_compo = []
    
    for i in range(len(ndvi_MYD)):
        
        d = ndvi_MYD['Date'][i]
        c = ndvi_MYD['Composite_date'][i]
        
        if c != -1:    
            if (d.month == 12) and (c<20):
                compo.append(fromjulian(str(d.year+1)+str(c)))
            else:
                compo.append(fromjulian(str(d.year)+str(c)))
        else:
            # nodata so we don't care about the date
            compo.append(d)
        
        if (i==0):
            same_compo.append(False)
        elif (compo[i]==compo[i-1]):
            same_compo.append(True)
        else:
            same_compo.append(False)
                
    ndvi_MYD['Composite_date'] = compo   
    ndvi_MYD['Same_composite'] = same_compo 
    
#=============================================================================#

    #creating MXD

    #changing the index to numbers
    number_index1 = pd.Index(range(0,2*len(ndvi_MYD),2))
    number_index2 = pd.Index(range(1,2*len(ndvi_MYD)+1,2))
    
    ndvi_MYD_nb = ndvi_MYD.set_index(number_index1)
    ndvi_MYD_nb['ID'] = ndvi_MYD.index
    ndvi_MOD_nb = ndvi_MOD.set_index(number_index2)
    ndvi_MOD_nb['ID'] = ndvi_MOD.index
    

    #concatenating and resetting ID as index
    ndvi_MXD = pd.concat([ndvi_MYD_nb, ndvi_MOD_nb]).sort_index()
    ndvi_MXD = ndvi_MXD.set_index('ID')
    
    #re-run same_composite
    same_compo = []
    for i in range(len(ndvi_MXD)):
        if (i==0 or i==1):
            same_compo.append(False)
        elif (ndvi_MXD['Composite_date'][i]==ndvi_MXD['Composite_date'][i-1] or ndvi_MXD['Composite_date'][i]==ndvi_MXD['Composite_date'][i-2]):
            same_compo.append(True)
        else:
            same_compo.append(False)
                  
    ndvi_MXD['Same_composite'] = same_compo 
    
    return ndvi_MXD


def read_lst():
    
    # loading ndvi_MOD data from csv
    lst_MYD = pd.read_csv(
        'data/MYD11A2-MYD11A2-006-results.csv', 
        index_col=0, 
        usecols = ['ID', 
                   'Latitude',
                   'Longitude',
                   'Date', 
                   'MYD11A2_006_LST_Day_1km'])
    
    
    #renaming the columns
    lst_MYD = lst_MYD.rename(columns={"MYD11A2_006_LST_Day_1km": "LST"})
    
    
    # Convert string Date to datetime.date
    lst_MYD['Date'] = lst_MYD['Date'].apply(fromstring)
    
    return lst_MYD


def read_data(product_type):
    
    if product_type == 'NDVI':
        product_MXD = read_ndvi()
        grid_sample = xr.open_zarr('data/vim_zarr')
        names_grid_sample = [
            f"grid({str(round(lat)).zfill(2)},{str(round(lon)).zfill(2)})" 
            for lat, lon in product(grid_sample.latitude.values, grid_sample.longitude.values)
        ]
    else:
        product_MXD = read_lst()
        grid_sample = xr.open_zarr('data/tda_zarr')
        names_grid_sample = [
            f"grid({str(round(lat)).zfill(2)},{str(round(lon)).zfill(2)})" 
            for lat, lon in product(grid_sample.latitude.values, grid_sample.longitude.values)
        ]
        
    return product_MXD, names_grid_sample