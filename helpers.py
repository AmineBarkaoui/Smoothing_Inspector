import datetime
import numpy as np
import pandas as pd

def fromstring(x):
    
    '''Converts string to datetime object'''
    
    try:
        d = datetime.datetime.strptime(x, '%d/%m/%Y').date()
    except:
        d = datetime.datetime.strptime(x, '%Y-%m-%d').date()
        
    return d

def fromjulian(x):
    
    '''Converts julian to datetime object'''

    return datetime.datetime.strptime(x, '%Y%j').date()

def ndvi_extract_ts(df: pd.DataFrame, location: str, date_begin: int, date_end: int):
    
    y = df['NDVI'].loc[location].values

    dts = df['Date'].loc[location].values

    c_dts = df['Composite_date'].loc[location].values

    same_c = df['Same_composite'].loc[location].values
    
    #Crop to date range
    date_range = np.all([dts>=datetime.date(date_begin,1,1), dts<=datetime.date(date_end,12,31)], axis=0)
    y = y[date_range]
    dts = dts[date_range]
    c_dts = c_dts[date_range]
    same_c = same_c[date_range]
    
    return (y, dts, c_dts, same_c)