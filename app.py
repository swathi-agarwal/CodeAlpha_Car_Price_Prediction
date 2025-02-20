import pandas as pd 
import numpy as np 
import pickle as pk 
import streamlit as st

model = pk.load(open('model.pkl','rb'))

st.header('Car Price Prediction ML Model')

cars_data = pd.read_csv('Car data.csv')

def get_brand_name(car_name):
    car_name = car_name.split(' ')[0]
    return car_name.strip()
cars_data['Car_Name'] = cars_data['Car_Name'].apply(get_brand_name)

Car_Name = st.selectbox('Select Car Brand', cars_data['Car_Name'].unique())
Year = st.slider('Car Manufactured Year', 1994,2024)
Present_Price=st.slider('Present Price',0.0,50.0,step=0.1)
Driven_kms = st.slider('No of kms Driven', 11,200000)
Fuel_Type = st.selectbox('Fuel type', cars_data['Fuel_Type'].unique())
Selling_type = st.selectbox('Selling type', cars_data['Selling_type'].unique())
Transmission = st.selectbox('Transmission type', cars_data['Transmission'].unique())
Owner = st.selectbox('Owner', cars_data['Owner'].unique())


if st.button("Predict"):
    input_data_model = pd.DataFrame(
    [[Car_Name,Year,Present_Price,Driven_kms,Fuel_Type,Selling_type,Transmission,Owner]],
    columns=['Car_Name','Year','Present_Price','Driven_kms','Fuel_Type','Selling_type','Transmission','Owner'])
    
    input_data_model['Owner'].replace(['First Owner', 'Second Owner', 'Third Owner'],
                           [0,1,3], inplace=True)
    input_data_model['Fuel_Type'].replace(['Petrol','Diesel', 'CNG'],[1,2,3], inplace=True)
    input_data_model['Selling_type'].replace(['Individual', 'Dealer'],[1,2], inplace=True)
    input_data_model['Transmission'].replace(['Manual', 'Automatic'],[1,2], inplace=True)
    input_data_model['Car_Name'].replace(['ritz', 'sx4', 'ciaz', 'wagon', 'swift', 'vitara', 's', 'alto', 'ertiga', 'dzire',
     'ignis', '800', 'baleno', 'omni', 'fortuner', 'innova','corolla', 'etios', 'camry', 'land', 'Royal', 'UM',
     'KTM', 'Bajaj', 'Hyosung', 'Mahindra', 'Honda', 'Yamaha', 'TVS', 'Hero', 'Activa', 
     'Suzuki','i20', 'grand', 'i10', 'eon', 'xcent', 'elantra', 'creta', 'verna', 'city', 'brio', 'amaze', 
     'jazz'],
                          [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,29,40,41,42,43,44]
                          ,inplace=True)

    car_price = model.predict(input_data_model)

    st.markdown('Car Price is going to be '+ str(car_price[0]))