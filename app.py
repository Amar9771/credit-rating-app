import os
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# 1) Set Streamlit page config
st.set_page_config(page_title="Credit Rating Predictor", layout="centered")

# 2) Custom CSS for Styling
st.markdown("""
    <style>
    body {
        background: linear-gradient(to right, #e0f2f1, #ffffff);
        padding-top: 0;
    }
    .block-container {
        max-width: 700px;
        margin: -2px auto 1rem !important;
        border: 2px solid #4CAF50 !important;
        border-top-left-radius: 0 !important;
        border-top-right-radius: 0 !important;
        border-bottom-left-radius: 15px !important;
        border-bottom-right-radius: 15px !important;
        background-color: white;
        padding: 2rem !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    .block-container h1 {
        text-align: center;
        color: #4CAF50;
        margin-bottom: 0.5rem;
    }
    input[type="text"], input[type="number"], select {
        max-width: 300px !important;
    }
    button[kind="primary"] {
        border-radius: 12px !important;
        padding: 10px 20px !important;
        background-color: #4CAF50;
        color: white;
        border: none;
        font-weight: bold;
        transition: background-color 0.3s ease;
    }
    button[kind="primary"]:hover {
        background-color: #45a049;
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        font-size: 0.9rem;
        color: #777;
    }
    label {
        font-size: 1.2rem !important;
        font-weight: bold;
        color: #333;
    }
    .historical-data {
        margin-top: 4rem;
    }
    </style>
""", unsafe_allow_html=True)

# 3) Header with Logo and Title
st.markdown("""
    <div style="text-align: center; margin-bottom: 0rem;">
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

# 4) Load model and encoders
model = joblib.load('credit_rating_model.pkl')
rating_encoder = joblib.load('rating_encoder.pkl')
issuer_encoder = joblib.load('issuer_encoder.pkl')
industry_encoder = joblib.load('industry_encoder.pkl')

# 5) CSV setup
historical_data_path = 'Simulated_CreditRating_Data.csv'
columns = [
    'Issuer Name', 'Industry', 'Debt to Equity', 'EBITDA Margin',
    'Interest Coverage', 'Issue Size (₹Cr)', 'DefaultFlag', 'Predicted Rating'
]
if not os.path.exists(historical_data_path):
    pd.DataFrame(columns=columns).to_csv(historical_data_path, index=False)

# 6) Session state defaults
session_defaults = {
    'issuer_name': issuer_encoder.classes_[0],
    'industry': industry_encoder.classes_[0],
    'default_flag': 0,
    'debt_to_equity': 0.0,
    'ebitda_margin': 0.0,
    'interest_coverage': 0.0,
    'issue_size': 0.0
}
for key, value in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# 7) Input form
col1, col2 = st.columns([1, 2])
with col1:
    issuer_name = st.selectbox("🏢 Issuer Name", issuer_encoder.classes_,
                               index=issuer_encoder.classes_.tolist().index(st.session_state.issuer_name))
    industry = st.selectbox("🏭 Industry", sorted(industry_encoder.classes_),
                            index=sorted(industry_encoder.classes_).index(st.session_state.industry))
    default_flag = st.selectbox("⚠️ Default Flag", [0, 1],
                                help="Set to 1 if issuer has defaulted, else 0",
                                index=[0, 1].index(st.session_state.default_flag))
with col2:
    debt_to_equity = st.number_input("📉 Debt to Equity Ratio", step=0.01,
                                     value=st.session_state.debt_to_equity)
    ebitda_margin = st.number_input("💰 EBITDA Margin (%)", step=0.01,
                                    value=st.session_state.ebitda_margin)
    interest_coverage = st.number_input("🧾 Interest Coverage Ratio", step=0.01,
                                        value=st.session_state.interest_coverage)
    issue_size = st.number_input("📦 Issue Size (₹ Crores)", step=1.0,
                                 value=st.session_state.issue_size)

# 8) Predict Button
st.markdown('<div style="text-align: center; margin-top: 2rem;">', unsafe_allow_html=True)
if st.button("🔍 Predict Credit Rating"):
    try:
        # Encode features
        issuer_idx = issuer_encoder.transform([issuer_name])[0]
        industry_idx = industry_encoder.transform([industry])[0]

        X_new = np.array([[debt_to_equity, ebitda_margin, interest_coverage,
                           issue_size, issuer_idx, industry_idx, default_flag]])

        if X_new.shape[1] != model.n_features_in_:
            raise ValueError(f"Expected {model.n_features_in_} features, got {X_new.shape[1]}")

        y_pred = model.predict(X_new)
        rating = rating_encoder.inverse_transform(y_pred)[0]
        st.success(f"🎯 Predicted Credit Rating: **{rating}**")

        # Save to CSV
        new_row = pd.DataFrame({
            'Issuer Name': [issuer_name],
            'Industry': [industry],
            'Debt to Equity': [debt_to_equity],
            'EBITDA Margin': [ebitda_margin],
            'Interest Coverage': [interest_coverage],
            'Issue Size (₹Cr)': [issue_size],
            'DefaultFlag': [default_flag],
            'Predicted Rating': [rating]
        })
        new_row.to_csv(historical_data_path, mode='a', header=False, index=False)

        # Reset session state and rerun
        for key in session_defaults:
            st.session_state[key] = session_defaults[key]
        st.experimental_rerun()

    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
st.markdown('</div>', unsafe_allow_html=True)

# 9) Historical data
st.markdown('<div class="historical-data">', unsafe_allow_html=True)
with st.expander("📜 Show Historical Data"):
    hist_df = pd.read_csv(historical_data_path)
    st.dataframe(hist_df)
st.markdown('</div>', unsafe_allow_html=True)

# 10) Footer
st.markdown("""
<div class="footer">
    <hr style="margin-top: 2rem; margin-bottom: 1rem;" />
    <p>🔒 Secure & Private | 🏦 Powered by ML | 💡 Created by Your Name</p>
</div>
""", unsafe_allow_html=True)
