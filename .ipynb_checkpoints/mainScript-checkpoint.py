import datetime
import calendar
import pandas as pd
import streamlit as st
import altair as alt
import numpy as np

#from smoothing import smooth

def spaghetti_plot(df, i, c, y1, y2, col2):

    dts = [datetime.datetime.strptime(str(x), '%j').date() for x in df.index.values]
    xticks = []
    for i,dt in enumerate(dts):
        xticks.append(calendar.month_name[dt.month][0:3] + ' ' + str(dt.day))
     
    #df.index = xticks   
    cols = []       
    for y in range(y1, y2+1):
        cols.append(str(y))


    col2.line_chart(df[cols])

    
def main():

    st.set_page_config(layout='wide')

    st.title("AFGHANISTAN \n") 
    st.markdown('### NDVI - time series plots')
    st.markdown('#')

    col1, col, col2 = st.columns([15, 1, 30]) 

    ct = pd.read_csv('Data/crop_type.csv', index_col = 0, header = 0)
    ct_dict = ct.to_dict()
    ct_dict = ct_dict[list(ct_dict.keys())[0]]
    inv_ct_dict = {v: k for k, v in ct_dict.items()}

    r = pd.read_csv('Data/regions.csv', index_col = 0, header = 0)
    r_dict = r.to_dict()
    r_dict = r_dict[list(r_dict.keys())[0]]
    inv_r_dict = {v: k for k, v in r_dict.items()}

    # Widgets
    y = col1.slider('Year',2002, 2010,(2002, 2021))
    c = col1.selectbox('Land Cover', ct['Legend'].values)
    i = col1.selectbox('Region', r['Adm1Name'].values)

    df_path = 'Data/df/' + inv_ct_dict[c] + '/df_' + str(inv_r_dict[i]) + '.csv'
    df = pd.read_csv(df_path, index_col = 0)
    df = df/10000



    spaghetti_plot(df, i, c, y[0], y[1], col2)





if __name__ == "__main__":
    main()
     