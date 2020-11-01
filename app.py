# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 13:48:58 2020

@author: Felipe
"""

import streamlit as st
import pandas as pd
import plotly.express as px

st.title('Escalation warning')

####################
## Initialization ##
####################

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

#########################
#Process Control Chat ##
########################

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

st.plotly_chart(pcc, use_container_width=True)
