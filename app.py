import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import datetime
import pickle

df = pd.read_csv('Turbine_Data.csv', parse_dates=['Unnamed: 0'], index_col='Unnamed: 0')
df.index = pd.to_datetime(df.index)
df = df[df['ActivePower'] >= 0]
df = df[['WindSpeed', 'ActivePower']]
df['WindSpeed'] = df['WindSpeed'].interpolate(method='time')
df['Month'] = df.index.month

# Load the trained model
with open("model1.pkl", "rb") as file:
    model = pickle.load(file)

st.title('Wind Energy Generation Prediction')
st.subheader('Please enter a timeframe you want to predict')
st.text('Note: Last recorded date is on April 1st 2020. The further into the future we predict, the less accurate our predictions become.')

start_date = datetime.date(2018, 1, 1)
from_date = st.date_input("From Date:", min_value = start_date)
to_date = st.date_input("To Date:", min_value = from_date)

submit_button = st.button("Submit")

if from_date and to_date and submit_button:
    date_range = pd.date_range(start=from_date, end=to_date, freq='10T')
    df_future = pd.DataFrame({'Date': date_range})
    
    # Extracting relevant months from the selected date range
    selected_months = list(set(pd.to_datetime([from_date, to_date]).month))
    
    # Filter original dataset for those months
    historical_ws = df[df['Month'].isin(selected_months)]['WindSpeed'].values
    
    if len(historical_ws) == 0:
        st.error("No windspeed data found for selected months in historical dataset.")
    else:
        # Repeat or trim windspeed to match the length of date_range
        repeats = len(df_future) // len(historical_ws) + 1
        windspeed_series = (historical_ws.tolist() * repeats)[:len(df_future)]
        df_future['WindSpeed'] = windspeed_series
        
        # Calculate standard deviation of selected months
        std_dev = np.std(historical_ws)
        mean_val = np.mean(historical_ws)
        
        # Add noise based on actual std deviation (normalized)
        noise = np.random.normal(loc=0.0, scale=std_dev, size=len(windspeed_series))
        windspeed_series = np.array(windspeed_series) + noise

        # Ensure no negative windspeed
        windspeed_series = np.clip(windspeed_series, a_min=0, a_max=None)

        # Predict ActivePower
        df_future['WindSpeed'] = windspeed_series
        df_future['Predicted ActivePower'] = model.predict(df_future[['WindSpeed']])

        # Display results
        st.subheader("Predicted Active Power (10-min resolution)")
        st.line_chart(df_future.set_index('Date')['Predicted ActivePower'])

        with st.expander("Show forecast data"):
            st.dataframe(df_future)