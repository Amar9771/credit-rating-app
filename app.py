import os
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# 1) Page config
st.set_page_config(page_title="Credit Rating Predictor", layout="centered")

# 2) Custom CSS (top bar + styling)
st.markdown("""
    <style>
    .top-bar {
        position: fixed; top: 0; left: 0;
        width: 100%; height: 6px;
        background-color: #4CAF50; z-index: 1000;
    }
    body {
        background: linear-gradient(to right, #e0f2f1, #ffffff);
        padding-top: 0;
    }
    .block-container {
        max-width: 700px;
        margin: 20px auto 1rem !important;
        border: 2px solid #4CAF50 !important;
        border-radius: 0 0 15px 15px !important;
        background-color: white;
        padding: 2rem !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    .block-container h1 {
        text-align: center; color: #4CAF50; margin-bottom: 0.5rem;
    }
    input, select {
        max-width: 300px !important;
    }
    button[kind="primary"] {
        border-radius: 12px !important;
        padding: 10px 20px !important;
        background-color: #4CAF50; color: white; border: none;
        font-weight: bold; transition: background-color 0.3s ease;
    }
    button[kind="primary"]:hover { background-color: #45a049; }
    .footer { text-align: center; margin-top: 2rem; font-size: 0.9rem; color: #777; }
    .historical-data { margin-top: 4rem; }
    </style>
    <div class="top-bar"></div>
""", unsafe_allow_html=True)

# 3) Header
st.markdown("""
    <div style="text-align:center; margin-bottom:0;">
        <img src="https://cdn-icons-png.flaticon.com/512/2331/2331970.png"
             width="40" style="margin-bottom:0;" />
        <h1 style="color:#4CAF50; margin-bottom:0;">Credit Rating Predictor</h1>
        <p style="color:#666; font-size:1rem; margin-top:0;">
            Predict issuer ratings based on key financial indicators
        </p>
    </div>
""", unsafe_allow_html=True)

# 4) Load model & encoders
model            = joblib.load('credit_rating_model.pkl')
rating_encoder   = joblib.load('rating_encoder.pkl')
issuer_encoder   = joblib.load('issuer_encoder.pkl')
industry_encoder = joblib.load('industry_encoder.pkl')

# 5) CSV setup (with DefaultFlag + Predicted Rating)
data_dir = os.path.join(os.getcwd(), 'data')
os.makedirs(data_dir, exist_ok=True)

historical_data_path = os.path.join(data_dir, 'Simulated_CreditRating_Data.csv')
columns = [
    'Issuer Name','Industry','Debt to Equity','EBITDA Margin',
    'Interest Coverage','Issue Size (₹Cr)','DefaultFlag','Predicted Rating'
]
if not os.path.exists(historical_data_path):
    pd.DataFrame(columns=columns).to_csv(historical_data_path, index=False)

# 6) Input form (with DefaultFlag)
col1, col2 = st.columns([1, 2])
with col1:
    issuer_name  = st.text_input("🏢 Issuer Name")
    industry     = st.selectbox("🏭 Industry", sorted(industry_encoder.classes_))
    default_flag = int(st.checkbox("⚠️ Default Flag?", value=False))
with col2:
    debt_to_equity    = st.number_input("📉 Debt to Equity Ratio", step=0.01)
    ebitda_margin     = st.number_input("💰 EBITDA Margin (%)", step=0.01)
    interest_coverage = st.number_input("🧾 Interest Coverage Ratio", step=0.01)
    issue_size        = st.number_input("📦 Issue Size (₹ Crores)", step=1.0)

# 7) Prediction logic
st.markdown('<div style="text-align:center; margin-top:2rem;">', unsafe_allow_html=True)
if st.button("🔍 Predict Credit Rating"):
    try:
        # encode issuer & industry
        issuer_idx   = issuer_encoder.transform([issuer_name])[0] if issuer_name in issuer_encoder.classes_ else -1
        industry_idx = industry_encoder.transform([industry])[0]

        # assemble 7 features
        X_new = np.array([[debt_to_equity, ebitda_margin,
                           interest_coverage, issue_size,
                           issuer_idx, industry_idx,
                           default_flag]])
        y_pred = model.predict(X_new)
        rating = rating_encoder.inverse_transform(y_pred)[0]
        st.success(f"🎯 Predicted Credit Rating: **{rating}**")

        # append to CSV under "Predicted Rating"
        new_row = pd.DataFrame([{
            'Issuer Name':       issuer_name,
            'Industry':          industry,
            'Debt to Equity':    debt_to_equity,
            'EBITDA Margin':     ebitda_margin,
            'Interest Coverage': interest_coverage,
            'Issue Size (₹Cr)':  issue_size,
            'DefaultFlag':       default_flag,
            'Predicted Rating':  rating
        }])
        new_row.to_csv(historical_data_path, mode='a', header=False, index=False)

    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
st.markdown('</div>', unsafe_allow_html=True)

# 8) Show history
st.markdown('<div class="historical-data">', unsafe_allow_html=True)
with st.expander("📜 Show Historical Data"):
    hist_df = pd.read_csv(historical_data_path)
    st.dataframe(hist_df)
st.markdown('</div>', unsafe_allow_html=True)

# 9) Footer
st.markdown("""
<div class="footer">
    <hr style="margin-top:2rem; margin-bottom:1rem;" />
    <p>🔒 Secure & Private | 🏦 Powered by ML | 💡 Created by Your Name</p>
</div>
""", unsafe_allow_html=True)
