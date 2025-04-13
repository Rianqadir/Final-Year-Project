import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

def run_tab():
    stock = "GOOG"  # Default stock

    # 📌 Get date range
    end = datetime.now()
    start = datetime(end.year - 10, end.month, end.day)

    # User input for stock symbol
    stock = st.text_input("Enter the stock symbol:", stock)

    # Fetch stock data
    bit_coin_data = yf.download(stock, start, end)

    # 🚀 Handle Multi-Level Index (Fix Column Issue)
    if isinstance(bit_coin_data.columns, pd.MultiIndex):
        bit_coin_data.columns = bit_coin_data.columns.droplevel(1)  # ✅ Drop extra index level

    # 🌟 Standardize Column Names
    bit_coin_data.rename(columns={'Adj Close': 'Close'}, inplace=True)

    # 🚨 Debug: Show Available Columns
    st.write("✅ Available columns in stock data:", bit_coin_data.columns.tolist())

    # 🚨 Stop if 'Close' Column is Missing
    if 'Close' not in bit_coin_data.columns:
        st.error("⚠️ Data issue: 'Close' column missing! Try another stock symbol.")
        st.stop()  # ✅ Stop execution if the Close column is not found

    # Load the pre-trained future prediction model
    model = load_model("Latest_bit_coin_model.keras")

    # 📊 Display stock data
    st.subheader(f"Stock Data for {stock}")
    st.dataframe(bit_coin_data.tail(10))

    # 🔄 Prepare Data for Prediction
    splitting_len = int(len(bit_coin_data) * 0.9)
    x_test = pd.DataFrame(bit_coin_data["Close"][splitting_len:])

    # 🔮 FUTURE PRICE PREDICTION
    st.subheader("🔮 Future Price Prediction")

    # Get last 100 days' data for prediction
    last_100 = bit_coin_data[['Close']].tail(100)
    scaler = MinMaxScaler(feature_range=(0, 1))
    last_100 = scaler.fit_transform(last_100['Close'].values.reshape(-1, 1)).reshape(1, -1, 1)

    prev_100 = np.copy(last_100).tolist()

    # Function to predict future stock prices
    def predict_future(no_of_days, prev_100):
        future_predictions = []
        for i in range(int(no_of_days)):
            next_day = model.predict(prev_100).tolist()
            prev_100[0].append(next_day[0])
            prev_100 = [prev_100[0][1:]]
            future_predictions.append(scaler.inverse_transform(next_day))
        return future_predictions

    no_of_days = int(st.text_input("Enter the number of days to predict:", "10"))
    future_results = predict_future(no_of_days, prev_100)
    future_results = np.array(future_results).reshape(-1, 1)

