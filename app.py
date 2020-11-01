# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 13:48:58 2020

@author: Felipe
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px


st.title('Customer Experience metrics')

######################
## Initialize data ###
######################

#Load data
@st.cache
def load_data(file_name):
    
    file = './data/' + file_name
    data = pd.read_csv(file)
    return data

data_load_state = st.text('Loading data...')
data = load_data('data.csv')

#Select market
market = st.selectbox(
    'Select market',
    tuple(set(data.Market))
    )
st.write('You selected market:', market)

#Select product
product= st.selectbox(
    'Select product',
    tuple(set(data.loc[data.Market == market,'Product']))
    )
st.write('You selected product:', product)

#Subset data
product_data = data.loc[data.Product == product]

#Show original (raw) dataset if requested by user
if st.checkbox('Show product data'):
    raw_data = load_data('raw_data.csv')
    raw_data.loc[raw_data.Product == product]



######################
## Initialize model ###
######################

intercept = 0.022649095842913088
w = np.array([0.01927402, -0.00090217, -0.02436434,  0.00587104, -0.00253383,
       -0.01159318, -0.00852396, -0.01359019,  0.0016901 ,  0.03455106,
       -0.00401627, -0.00583471, -0.0121614 , -0.00535108, -0.00212911,
       -0.0023867 , -0.00404204,  0.00191445,  0.02242074,  0.00179926,
       -0.04573396, -0.01224608, -0.00255499,  0.10515191, -0.00619868,
        0.06148885, -0.00286004,  0.00623183, -0.03669058, -0.00660404,
       -0.00419707, -0.04966515,  0.00366485])


###########################
# Predict sr_esc_cnt_pf ##
##########################

most_recent_date = max(product_data.Date)
features = ['sr_bug_cnt',
            'sr_hwr_cnt_pf',
            'sr_init_sev_1_2_cnt_pf', 
            'sr_esc_cnt_pf', 
            'Market_Computing Systems',
            'Market_Data Center Networking',
            'Market_Enterprise Routing',
            'Market_Enterprise Switching', 
            'Market_Security',
            'Market_Service Provider Routing',
            'Product_Anastasia',
            'Product_Belle',
            'Product_Brownbear',
            'Product_Burrito',
            'Product_Centauri',
            'Product_Diana',
            'Product_Fajita',
            'Product_Fiona',
            'Product_Grizzlybear',
            'Product_Jasmine',
            'Product_Jupiter',
            'Product_Littlebear',
            'Product_Mamabear',
            'Product_Mars',
            'Product_Mr_Clean',
            'Product_Neptune',
            'Product_Orion',
            'Product_Papabear',
            'Product_Pluto',
            'Product_Rigel',
            'Product_Taco',
            'Product_Venus',
            'Product_Windex']

#########################
#Process Control Chat ##
########################

st.subheader('Statistical Quality Control chart')

#Select metric
metric = st.selectbox(
    'Select metric to display',
    ('sr_bug_cnt','sr_hwr_cnt_pf','sr_init_sev_1_2_cnt_pf','sr_esc_cnt_pf')
    )

mean = product_data.loc[:,metric].mean()
std= product_data.loc[:,metric].std()
z = 2.66
UCL= mean + z * std
LCL= mean - z * std
CL_style = dict(color = 'Red', dash='dash')

#metric evolution
pcc = px.line(product_data,
       x = 'Date',
       y = product_data.loc[:,metric],
       title= f'{metric} ({product})',
       range_y=(LCL - 1 * std, UCL + 1 * std))

#mean
pcc.add_shape(type='line',
                x0=min(product_data.Date),
                y0=mean,
                x1=max(product_data.Date),
                y1=mean,
                line=dict(color='Gray',dash='dash'),
                xref='x',
                yref='y'
                )

#UCL
pcc.add_shape(type='line',
                x0=min(product_data.Date),
                y0=UCL,
                x1=max(product_data.Date),
                y1=UCL,
                line=CL_style,
                xref='x',
                yref='y'
                )

#LCL
pcc.add_shape(type='line',
                x0=min(product_data.Date),
                y0=LCL,
                x1=max(product_data.Date),
                y1=LCL,
                line=CL_style,
                xref='x',
                yref='y'
                )

if metric == 'sr_esc_cnt_pf':
    #Show horizontal line on prediction for next month
    if st.checkbox('Show prediction for next month'):
        #calculate prediction
        X = product_data.loc[product_data.Date == most_recent_date, features].to_numpy()
        next_month_escalation_prediction = intercept + X.dot(w)
        st.write('Next month escalation prediction',next_month_escalation_prediction)
        #plot prediction
        pcc.add_shape(type='line',
                x0=min(product_data.Date),
                y0=next_month_escalation_prediction[0],
                x1=max(product_data.Date),
                y1=next_month_escalation_prediction[0],
                line=dict(color='Green', dash='dash'),
                xref='x',
                yref='y'
                )


st.plotly_chart(pcc, use_container_width=True)

################################
## Examine overall algorithm ##
###############################

data

