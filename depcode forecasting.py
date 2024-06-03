import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Function to load and preprocess data
def load_data(filename):
    data = pd.read_csv(filename)
    data['Date'] = pd.to_datetime(data['Date'], format='%b %d, %Y')
    data.set_index('Date', inplace=True)
    return data

# Function to train ARIMA model and make predictions
def train_arima(data):
    train_size = int(len(data) * 0.8)
    train, test = data[:train_size], data[train_size:]
    model = ARIMA(train, order=(5,1,0))
    model_fit = model.fit()
    forecast, _, _ = model_fit.forecast(steps=len(test))
    return test, forecast

# Function to evaluate model performance
def evaluate_model(test, forecast):
    mse = mean_squared_error(test, forecast)
    rmse = np.sqrt(mse)
    return rmse

# Main function to run the Streamlit app
def main():
    st.title('European Brent Crude Oil Price Forecasting')

    # Upload data file
    EB = st.file_uploader("EuropeanBrent.csv", type=['csv'])
    if EB is not None:
        data = load_data(EB)

        st.subheader('Data Preview')
        st.write(data.head())

        st.subheader('Data Visualization')
        plt.figure(figsize=(10, 6))
        plt.plot(data)
        plt.title('European Brent Crude Oil Prices')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        st.pyplot()

        test, forecast = train_arima(data)

        rmse = evaluate_model(test, forecast)

        st.subheader('Model Evaluation')
        st.write('Test RMSE:', rmse)

        st.subheader('Forecast Visualization')
        plt.figure(figsize=(10, 6))
        plt.plot(test.index, test, label='Actual')
        plt.plot(test.index, forecast, color='red', label='Forecast')
        plt.title('European Brent Crude Oil Prices - Test vs Forecast')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        st.pyplot()

if __name__ == "__main__":
    main()
