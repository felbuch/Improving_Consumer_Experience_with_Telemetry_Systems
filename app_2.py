# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 13:48:58 2020

@author: Felipe
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression


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

model = LinearRegression()

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

X = product_data.loc[:,features].to_numpy()
y = product_data.sr_esc_cnt_pf_lag.to_numpy()

model.fit(X,y)

yhat = model.predict(X)
product_data['prediction'] = yhat
prediction = product_data.loc[product_data.Date == most_recent_date, 'prediction']
prediction = prediction.iloc[0]

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
        st.write('Next month escalation prediction', round(prediction,2))
        #plot prediction
        pcc.add_shape(type='line',
                x0=min(product_data.Date),
                y0=prediction,
                x1=max(product_data.Date),
                y1=prediction,
                line=dict(color='Green', dash='dash'),
                xref='x',
                yref='y'
                )


st.plotly_chart(pcc, use_container_width=True)

#####################
##  List Warnings ##
####################

#Show original (raw) dataset if requested by user
if st.checkbox('Show warnings signs, if any'):
    
    flagged = []
    for prod in set(data.Product):
        try:
            temp = data.loc[data.Product == prod]
            X = temp.loc[:,features].to_numpy()
            y = temp.sr_esc_cnt_pf_lag.to_numpy()
            model.fit(X,y)
            yhat = model.predict(X)
            temp['prediction'] = yhat
            prediction = temp.loc[temp.Date == most_recent_date, 'prediction']
            yhat = prediction.iloc[0]   
            mean = np.mean(y)
            sd = np.std(y)
            UCL = mean + z * sd
            LCL = mean - z * sd
            warn = (yhat > UCL) | (yhat < LCL)
            if warn:
                flagged.append(prod)
        except:
            pass
    
    if flagged == []:
        st.write('No products were flagged')
    else:
        for prod in flagged:
            st.write(prod)



################################
## Examine overall algorithm ##
###############################

st.subheader('Appendix')


if st.checkbox('Evaluate predictive quality of algorithm'):
    error_plot = px.scatter(data, 
                            x = 'sr_esc_cnt_pf_lag',
                            y = 'Label',
                            trendline='ols',
                            range_x=(1e-3,0.25),
                            range_y=(1e-3, 0.40),
                            labels={'sr_esc_cnt_pf_lag': 'True value',
                                    'Label': 'Predicted value'},
                            title='Comparison between true and predicted value (R2 = 93%)'
                            )

    error_plot.add_shape(type='line',
                         x0 = 0,
                         y0 = 0,
                         x1 = 1,
                         y1 = 1
                         )
    
    st.plotly_chart(error_plot, use_container_width=True)

