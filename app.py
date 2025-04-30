import os
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# 1) Page Setup
st.set_page_config(page_title="Credit Rating Predictor", layout="centered", page_icon="🏦")

# 2) Custom CSS Styling
st.markdown("""
<style>
    body {
        background: linear-gradient(to right, #eef2f3, #ffffff);
    }
    .block-container {
        max-width: 800px;
        padding: 2rem;
        background-color: #ffffff;
        border-radius: 15px;
        border: 2px solid #4CAF50;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    h1, h3, h4, h5 {
        color: #4CAF50;
        text-align: center;
    }
    .stTextInput>div>div>input,
    .stNumberInput>div>div>input,
    .stSelectbox>div>div>div {
        font-size: 16px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 12px;
        padding: 10px 24px;
        font-weight: bold;
        font-size: 16px;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        font-size: 0.9rem;
        color: #888;
    }
</style>
""", unsafe_allow_html=True)

# 3) Header Section
st.markdown("""
    <div style="text-align: center;">
        <img src="https://cdn-icons-png.flaticon.com/512/2331/2331970.png" width="50"/>
        <h1>Credit Rating Predictor</h1>
        <p style="color:#555; font-size:1.1rem;">Estimate credit ratings using key financial metrics and machine learning</p>
    </div>
""", unsafe_allow_html=True)

# 4) Load Model and Encoders
model = joblib.load('credit_rating_model.pkl')
rating_encoder = joblib.load('rating_encoder.pkl')
issuer_encoder = joblib.load('issuer_encoder.pkl')
industry_encoder = joblib.load('industry_encoder.pkl')

# 5) Setup CSV File
historical_data_path = 'Simulated_CreditRating_Data.csv'
columns = ['Issuer Name', 'Industry', 'Debt to Equity', 'EBITDA Margin', 'Interest Coverage',
           'Issue Size (₹Cr)', 'DefaultFlag', 'Predicted Rating']

if not os.path.exists(historical_data_path):
    pd.DataFrame(columns=columns).to_csv(historical_data_path, index=False)

# 6) Session State Defaults
defaults = {
    'issuer_name': "",
    'industry': industry_encoder.classes_[0],
    'default_flag': 0,
    'debt_to_equity': 0.0,
    'ebitda_margin': 0.0,
    'interest_coverage': 0.0,
    'issue_size': 0.0
}
for key, value in defaults.items():
    st.session_state.setdefault(key, value)

# 7) Input Form
st.markdown("### 🔍 Enter Financial Details")
col1, col2 = st.columns(2)

with col1:
    issuer_name = st.text_input("🏢 Issuer Name", value=st.session_state.issuer_name)
    industry = st.selectbox("🏭 Industry", sorted(industry_encoder.classes_),
                            index=sorted(industry_encoder.classes_).index(st.session_state.industry))
    default_flag = st.selectbox("⚠️ Default Flag", [0, 1], index=[0, 1].index(st.session_state.default_flag),
                                help="Set to 1 if issuer has defaulted, else 0")

with col2:
    debt_to_equity = st.number_input("📉 Debt to Equity Ratio", step=0.01, value=st.session_state.debt_to_equity)
    ebitda_margin = st.number_input("💰 EBITDA Margin (%)", step=0.01, value=st.session_state.ebitda_margin)
    interest_coverage = st.number_input("🧾 Interest Coverage Ratio", step=0.01, value=st.session_state.interest_coverage)
    issue_size = st.number_input("📦 Issue Size (₹ Crores)", step=1.0, value=st.session_state.issue_size)

# 8) Prediction Logic
st.markdown('<div style="text-align:center; margin-top:2rem;">', unsafe_allow_html=True)
if st.button("🎯 Predict Credit Rating"):
    try:
        issuer_idx = issuer_encoder.transform([issuer_name])[0] if issuer_name in issuer_encoder.classes_ else -1
        industry_idx = industry_encoder.transform([industry])[0]
        X_new = np.array([[debt_to_equity, ebitda_margin, interest_coverage, issue_size,
                           issuer_idx, industry_idx, default_flag]])

        if X_new.shape[1] != model.n_features_in_:
            raise ValueError("Input feature mismatch with model.")

        prediction = model.predict(X_new)
        rating = rating_encoder.inverse_transform(prediction)[0]

        st.success(f"✅ Predicted Credit Rating: **{rating}**")

        # Save result to historical CSV
        pd.DataFrame({
            'Issuer Name': [issuer_name],
            'Industry': [industry],
            'Debt to Equity': [debt_to_equity],
            'EBITDA Margin': [ebitda_margin],
            'Interest Coverage': [interest_coverage],
            'Issue Size (₹Cr)': [issue_size],
            'DefaultFlag': [default_flag],
            'Predicted Rating': [rating]
        }).to_csv(historical_data_path, mode='a', header=False, index=False)

        # Reset session state for form fields
        for key in defaults:
            st.session_state[key] = defaults[key]

        st.experimental_rerun()

    except Exception as e:
        st.error(f"❌ Error in prediction: {e}")
st.markdown('</div>', unsafe_allow_html=True)

# 9) Show Historical Data
with st.expander("📜 View Historical Predictions"):
    df = pd.read_csv(historical_data_path)
    st.dataframe(df, use_container_width=True)

# 10) Footer
st.markdown("""
<div class="footer">
    <hr />
    <p>🔒 All data is confidential | 🧠 Powered by Machine Learning | 📊 Built by <strong>Your Name</strong></p>
</div>
""", unsafe_allow_html=True)
