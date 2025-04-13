import streamlit as st
from web_stock_price_predictor import run_stock_price_predictor  # Import Tab 1
from web_bit_coin_price_predictor import run_future_prediction  # Import Tab 2

# 🎨 ---- Modern Styling ----
st.set_page_config(page_title="Stock Predictor", page_icon="📈", layout="wide")

# 🎯 Tabs for Navigation
tab1, tab2 = st.tabs(["📊 Stock Analysis", "🔮 Future Prediction"])

# 📌 ---- TAB 1: STOCK ANALYSIS ----
with tab1:
    run_stock_price_predictor()  # Call function from `web_stock_price_predictor.py`

# 📌 ---- TAB 2: FUTURE PREDICTION ----
with tab2:
    run_future_prediction()  # Call function from `web_bit_coin_price_predictor.py`

# 🎨 Custom CSS for Styling
st.markdown("""
    <style>
    body { font-family: 'Arial', sans-serif; }
    .main-title { font-size:36px; text-align:center; font-weight: bold; color: #ff914d; }
    .sub-title { font-size:22px; font-weight: bold; color: #444; }
    .info-box { padding:15px; background:#f7f7f7; border-radius:10px; }
    .stButton>button { border-radius: 10px; background-color: #ff914d; color: white; font-size: 16px; }
    .stTextInput>div>div>input { border-radius: 10px; }
    .styled-table { 
        background-color: #222222 !important;  /* Dark Background */
        color: #ffffff !important;  /* White Text */
        border-radius: 10px;
        padding: 10px;
        overflow-x: auto;
    }
    </style>
""", unsafe_allow_html=True)
