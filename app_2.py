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

#Define function to load data
@st.cache
def load_data(file_name):
    
    file = './data/' + file_name
    data = pd.read_csv(file)
    return data

#Load data
data_original = load_data('data_2.csv') # for control charts
st.text('Data loaded succesfully')

#Load coefficients
darg = load_data('data_reg.csv') # for regression
coef = load_data('ICE_coefficients_2.csv')
st.text('Machine Learning model loaded succesfully')

#drop unnamed 0 column
#data = data.drop(columns=['Unnamed: 0'])
darg = darg.drop(columns=['Unnamed: 0'])
coef = coef.drop(columns=['Unnamed: 0'])

#Normalize metrics
data = data_original.copy()
data['sr_bug_cnt'] = data.sr_bug_cnt / data.sr_cnt_pf
data['sr_hwr_cnt_pf'] = data.sr_hwr_cnt_pf / data.sr_cnt_pf
data['sr_init_sev_1_2_cnt_pf'] = data.sr_init_sev_1_2_cnt_pf / data.sr_cnt_pf
data['sr_esc_cnt_pf'] = data.sr_esc_cnt_pf/ data.sr_cnt_pf
data = data.drop(columns='sr_cnt_pf')

##############################
## User initial selections ###
##############################

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


#######################
## Data processing ###
######################

#Subset data
product_data = data.loc[data.Product == product]
product_darg = darg.loc[darg.Product == product]
product_coef = coef.loc[coef.Product == product]

#Sort data
product_data = product_data.sort_values('Date')


#Prepare feature vector for prediction

def make_X(df):
    '''
    Parameters
    ----------
    df : DataFrame
        product_darg dataframe

    Returns
    -------
    A pandas Series with the features to be used for prediction

    '''
    t = df.shape[0] + 1
    
    #Which columns do we have coefficients for
    cols = ['sr_bug_cnt',
            'sr_hwr_cnt_pf',
            'sr_init_sev_1_2_cnt_pf',
            'sr_esc_cnt_pf',
            'sr_bug_cnt_mean',
            'sr_bug_cnt_std',
            'sr_hwr_cnt_pf_mean',
            'sr_hwr_cnt_pf_std',
            'sr_init_sev_1_2_cnt_pf_mean',
            'sr_init_sev_1_2_cnt_pf_std',
            'sr_esc_cnt_pf_mean',
            'sr_esc_cnt_pf_std']
    
    X = (df.
         loc[:, cols]. #select only relevant columns from X
         apply(pd.to_numeric).
         assign(t = t). #create time variable
         assign(t2 = t**2) #create its square
         )
    X.insert(0,'intercept',1) #insert 1 for intercept column
    X = X.iloc[-1,:] #get last row
    return(X)
    
def make_beta(df):
    '''
    

    Parameters
    ----------
    df : Pandas DataFrame
        product_coef
    Returns
    -------
    A Pandas Series of coefficients to be used for prediction

    '''
    #Prepare coefficient vector for prediction
    beta = (product_coef.
            drop(columns='Product').
            apply(pd.to_numeric).
            T
            )
    
    return(beta)

X = make_X(product_darg)
beta = make_beta(product_coef)
  
#Make prediction
prediction = X @ beta
prediction = prediction.iloc[0]

#Show original (raw) dataset if requested by user
if st.checkbox('Show product data'):
    raw_data = load_data('raw_data.csv')
    raw_data.loc[raw_data.Product == product]


#########################
#Process Control Chat ##
########################

st.subheader('Statistical Quality Control chart')

user_z = st.slider('Spread of control limits', 1., 6., 2.66, 0.01)

#Select metric
metric = st.selectbox(
    'Select metric to display',
    ('sr_bug_cnt','sr_hwr_cnt_pf','sr_init_sev_1_2_cnt_pf','sr_esc_cnt_pf')
    )

def pcc_params(df, m, z):
    '''
    Calculates parameters for pcc, such as mean and control limits

    Parameters
    ----------
    df : dataframe
        product_data dataframe.
    m : str
        metric being observed in the process control chart.
    z : float, optional
        Number of standard deviations between mean and CLs.

    Returns
    -------
    A tuple containing mean, standard deviation, UCL and LCL

    '''
    
    mean = df.loc[:,m].mean()
    std= df.loc[:,m].std()
    UCL= mean + z * std
    LCL= mean - z * std
    
    return((mean, std, UCL, LCL))
    
mean, std, UCL, LCL = pcc_params(df = product_data, m = metric, z = user_z)    

#Define control limits style in plot
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

#######################################
##  Option button to see prediction ##
######################################


if metric == 'sr_esc_cnt_pf':
    #Show horizontal line on prediction for next month
    show_prediction = st.checkbox('Show prediction for next month')
    if show_prediction:
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
        
#Show Process Control Chart
st.plotly_chart(pcc, use_container_width=True)

if "show_prediction" in globals():        
    if show_prediction:
        #Show contributions
        if st.checkbox('Show factors affecting escalation prediction'):
            df = product_coef.drop(columns=['Product','intercept']).T.reset_index()
            df.columns = ['feature','contribution']
            df['importance'] =  round(df.contribution / np.mean(df.contribution), 4) #express in %
            pie = px.pie(data_frame = df,
                         names = 'feature', 
                         values = 'importance'
                         )
                
            st.plotly_chart(pie)
    

#####################
##  List Warnings ##
####################

#Show original (raw) dataset if requested by user
if st.checkbox('Show warnings signs, if any'):
    
    flagged = []
    for prod in set(data.Product):
        try:
            temp_darg = darg.loc[darg.Product == prod]
            temp_coef = coef.loc[coef.Product == prod]
            temp_X = make_X(temp_darg)
            temp_beta = make_beta(temp_coef)
            temp_pred = temp_X @ temp_beta
            
            temp_y = temp_darg.sr_esc_cnt_pf_lag
            
            mean = np.mean(temp_y)
            sd = np.std(temp_y)
            UCL = mean + user_z * sd
            LCL = mean - user_z * sd
            warn = (temp_pred > UCL) | (temp_pred < LCL)
            if warn:
                flagged.append(prod)
        except:
            pass
    
    if flagged == []:
        st.write('No products were flagged')
    else:
        for prod in flagged:
            st.write(prod)


