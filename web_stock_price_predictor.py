import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler

# 🎨 Modern UI Styling
st.set_page_config(page_title="Stock Predictor", page_icon="📈", layout="wide")

# 🔄 **Tab Navigation**
tab1, tab2 = st.tabs(["📊 Stock Analysis", "🔮 Future Prediction"])

# ✅ Ensure session state keys exist before assignment
if "stock_data" not in st.session_state:
    st.session_state["stock_data"] = None  # Initialize

if "stock_symbol" not in st.session_state:
    st.session_state["stock_symbol"] = None  # Initialize

# 📌 ---- TAB 1: STOCK ANALYSIS ----
with tab1:
    st.title("📈 Stock Price Predictor")

    # 🎯 Sidebar for Stock Selection & Date Range
    st.sidebar.title("📊 Stock Predictor Menu")
    stock = st.sidebar.text_input("Enter Stock Symbol (e.g., AAPL, TSLA, MSFT)", "GOOG")

    # 📅 Date Selector
    end = datetime.now()
    start = st.sidebar.date_input("Start Date", datetime(end.year - 10, end.month, end.day))
    end = st.sidebar.date_input("End Date", end)

    # 📌 Fetch Data
    st.sidebar.write("🔄 **Fetching Data...**")
    stock_data = yf.download(stock, start=start, end=end)

    # 🚨 Stop if No Data
    if stock_data.empty:
        st.sidebar.error("🚨 No data found! Try another ticker.")
        st.stop()

    # 🌟 Fix MultiIndex Columns
    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data.columns = stock_data.columns.droplevel(1)

    # 🔄 Ensure 'Close' Column Exists
    stock_data.rename(columns={'Adj Close': 'Close'}, inplace=True)
    if 'Close' not in stock_data.columns:
        st.sidebar.error("🚨 'Close' column missing!")
        st.stop()

    # ✅ Store Stock Data & Symbol in Session State (So Tab 2 Can Use It)
    st.session_state["stock_symbol"] = stock
    st.session_state["stock_data"] = stock_data

    # 📊 **Display Stock Data Table**
    st.subheader(f"📊 Stock Data for {stock}")
    st.dataframe(stock_data.tail(10))  # Show last 10 rows

    # 📉 Compute Moving Averages
    stock_data['MA_250'] = stock_data['Close'].rolling(250).mean()
    stock_data['MA_100'] = stock_data['Close'].rolling(100).mean()

    # 📈 **Stock Price with Moving Averages**
    st.subheader(f"📊 {stock} Close Price with Moving Averages")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Close Price', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MA_250'], mode='lines', name='250-Day MA', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MA_100'], mode='lines', name='100-Day MA', line=dict(color='green')))
    fig.update_layout(title=f"{stock} Stock Price Analysis", xaxis_title="Date", yaxis_title="Price (USD)")
    st.plotly_chart(fig, use_container_width=True)

    # 🔄 **Data Splitting for Prediction**
    splitting_len = int(len(stock_data) * 0.7)
    x_test = stock_data[['Close']][splitting_len:].copy()

    # 📉 Normalize Data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(x_test)

    # 🧠 **Prepare Data for Model Prediction**
    x_data, y_data = [], []
    for i in range(100, len(scaled_data)):
        x_data.append(scaled_data[i - 100:i])
        y_data.append(scaled_data[i])

    x_data, y_data = np.array(x_data), np.array(y_data)

    # 🚀 **Load Pre-trained Model**
    model = load_model("Latest_stock_price_model.keras")

    # 🔮 **Make Predictions**
    predictions = model.predict(x_data)

    # 🔄 Inverse Transform
    inv_predictions = scaler.inverse_transform(predictions)
    inv_y_test = scaler.inverse_transform(y_data)

    # 📌 **Store Predictions in DataFrame**
    ploting_data = pd.DataFrame({
        'Actual Close Price': inv_y_test.reshape(-1),
        'Predicted Close Price': inv_predictions.reshape(-1)
    }, index=stock_data.index[splitting_len + 100:])

    # 📊 **Display Prediction Data Table**
    st.subheader("📊 Prediction Data Table")
    st.dataframe(ploting_data)

    # 📈 **Predicted vs Actual Close Price Chart**
    st.subheader("🔍 Actual vs Predicted Close Prices")
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(x=ploting_data.index, y=ploting_data['Actual Close Price'], mode='lines', name="Actual", line=dict(color='blue')))
    fig_pred.add_trace(go.Scatter(x=ploting_data.index, y=ploting_data['Predicted Close Price'], mode='lines', name="Predicted", line=dict(color='red')))
    fig_pred.update_layout(title="Stock Price Prediction", xaxis_title="Date", yaxis_title="Price (USD)")
    st.plotly_chart(fig_pred, use_container_width=True)

# 📌 ---- TAB 2: FUTURE PREDICTION ----
with tab2:
    st.title("🔮 Future Price Prediction")

    # 🏦 **Retrieve Stock Data from Tab 1**
    if st.session_state["stock_data"] is not None:
        bit_coin_data = st.session_state["stock_data"]
        stock = st.session_state["stock_symbol"]
    else:
        st.error("⚠️ No stock data found! Please select a stock in Tab 1.")
        st.stop()

    # 🚀 **Load Pre-trained Model**
    model = load_model("Latest_bit_coin_model.keras")

    # 📉 **Split Data for Training & Testing**
    splitting_len = int(len(bit_coin_data) * 0.9)
    x_test = pd.DataFrame(bit_coin_data["Close"][splitting_len:])

    # 🔮 **Future Price Prediction**
    st.subheader("🔮 Future Price Values")

    # **Extract Last 100 Days for Future Prediction**
    last_100 = bit_coin_data[['Close']].tail(100)
    last_100 = scaler.transform(last_100.values.reshape(-1, 1)).reshape(1, -1, 1)

    # **Function to Predict Future Prices**
    def predict_future(no_of_days, prev_100, last_known_date):
        future_predictions = []
        future_dates = []
        
        for i in range(int(no_of_days)):
            next_day = model.predict(prev_100)
            prev_100 = np.append(prev_100[:, 1:, :], [[next_day[0]]], axis=1)
            future_predictions.append(scaler.inverse_transform(next_day)[0][0])
            future_dates.append(last_known_date + timedelta(days=i+1))  # Generate future dates

        return np.array(future_predictions), future_dates

    # **User Input for Future Prediction**
    no_of_days = int(st.text_input("Enter the number of days to predict:", "10"))

    # **Get Last Date for Future Dates**
    last_known_date = bit_coin_data.index[-1]

    # **Predict Future Prices**
    future_results, future_dates = predict_future(no_of_days, last_100, last_known_date)

    # 📊 **Display Future Predictions Table with Date**
    future_table = pd.DataFrame({
        'Date': future_dates,
        'Predicted Close Price': future_results
    })
    st.subheader("📋 Future Predictions Data Table")
    st.dataframe(future_table)

    # 📈 **Future Predictions Graph**
    st.subheader(f"📈 Predicted Prices for Next {no_of_days} Days")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=future_table['Date'], y=future_table['Predicted Close Price'], mode='lines+markers', name="Future Prediction", line=dict(color='purple')))
    st.plotly_chart(fig, use_container_width=True)

