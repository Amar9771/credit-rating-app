import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load model and label encoder
model_path = "F:/Amareesh K/DataClean/Feb'25/27th/Predictive Model/rf_credit_rating_model.pkl"
encoder_path = "F:/Amareesh K/DataClean/Feb'25/27th/Predictive Model/label_encoder.pkl"
historical_data_path = "F:/Amareesh K/DataClean/Feb'25/27th/Predictive Model/historical_credit_data.csv"

model = joblib.load(model_path)
label_encoder = joblib.load(encoder_path)

# App layout and title with tooltip
st.set_page_config(page_title="Credit Rating Predictor", layout="centered")
st.markdown("""
    <style>
        body {
            font-family: 'Segoe UI', sans-serif;
            background-color: #f9f9f9;
        }
        h1 {
            font-weight: 600;
        }
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
            border-radius: 10px;
            padding: 10px 24px;
        }
        .stDataFrame {
            box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
            border-radius: 10px;
            padding: 10px;
        }
        select {
            padding: 8px;
            border-radius: 10px;
            border: 1px solid #ccc;
            box-shadow: 2px 2px 6px rgba(0,0,0,0.05);
            font-size: 1rem;
            margin-bottom: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div style="text-align: center; margin-bottom: 0rem;" title="Predict the credit rating of issuers using financial ratios and machine learning.">
        <img src="https://cdn-icons-png.flaticon.com/512/2331/2331970.png"
             width="40" style="margin-bottom: 0px;" />
        <h1 style="color: #4CAF50; margin-bottom: 0.0rem;">
            Credit Rating Predictor
        </h1>
        <p style="color: #666; font-size: 1.0rem; margin-top: 0;">
            Predict issuer ratings based on key financial indicators
        </p>
    </div>
""", unsafe_allow_html=True)

# Input fields
st.markdown("### 📊 Enter Financial Ratios")

col1, col2, col3 = st.columns(3)
with col1:
    current_ratio = st.number_input("Current Ratio", min_value=0.0, format="%.2f")
with col2:
    quick_ratio = st.number_input("Quick Ratio", min_value=0.0, format="%.2f")
with col3:
    debt_equity_ratio = st.number_input("Debt-to-Equity Ratio", min_value=0.0, format="%.2f")

col4, col5, col6 = st.columns(3)
with col4:
    interest_coverage_ratio = st.number_input("Interest Coverage Ratio", min_value=0.0, format="%.2f")
with col5:
    net_profit_margin = st.number_input("Net Profit Margin (%)", format="%.2f")
with col6:
    return_on_assets = st.number_input("Return on Assets (%)", format="%.2f")

col7, col8 = st.columns(2)
with col7:
    return_on_equity = st.number_input("Return on Equity (%)", format="%.2f")
with col8:
    operating_margin = st.number_input("Operating Margin (%)", format="%.2f")

input_data = np.array([[current_ratio, quick_ratio, debt_equity_ratio,
                        interest_coverage_ratio, net_profit_margin,
                        return_on_assets, return_on_equity, operating_margin]])

# Predict
if st.button("Predict Credit Rating"):
    prediction_encoded = model.predict(input_data)[0]
    prediction_label = label_encoder.inverse_transform([prediction_encoded])[0]
    st.success(f"Predicted Credit Rating: **{prediction_label}**")

# Show historical data with filters
st.markdown('<div class="historical-data">', unsafe_allow_html=True)
with st.expander("📜 Show Historical Data"):
    hist_df = pd.read_csv(historical_data_path)

    st.markdown("### 🔍 Filter Data")
    filter_col1, filter_col2 = st.columns(2)

    with filter_col1:
        selected_industry = st.selectbox("🏭 Filter by Industry", ["All"] + sorted(hist_df["Industry"].dropna().unique().tolist()))
    with filter_col2:
        selected_issuer = st.selectbox("🏢 Filter by Issuer", ["All"] + sorted(hist_df["Issuer Name"].dropna().unique().tolist()))

    # Apply filters
    filtered_df = hist_df.copy()
    if selected_industry != "All":
        filtered_df = filtered_df[filtered_df["Industry"] == selected_industry]
    if selected_issuer != "All":
        filtered_df = filtered_df[filtered_df["Issuer Name"] == selected_issuer]

    st.dataframe(filtered_df, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)
