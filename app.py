import os
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# 1) Page config
st.set_page_config(page_title="Credit Rating Predictor", layout="centered")

# 2) Custom CSS (including hover-only subtitle)
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
    .subtitle {
        visibility: hidden;
        opacity: 0;
        transition: opacity 0.3s ease, visibility 0.3s ease;
        color: #666;
        font-size: 1.0rem;
        margin-top: 0;
    }
    .title:hover + .subtitle {
        visibility: visible;
        opacity: 1;
    }
    input[type="number"], select {
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
    .historical-data {
        margin-top: 4rem;
    }
    </style>
""", unsafe_allow_html=True)

# 3) Header with hover-only subtitle
st.markdown("""
    <div style="text-align:center; margin-bottom:1rem;">
        <img src="https://cdn-icons-png.flaticon.com/512/2331/2331970.png"
             width="35" style="margin-bottom:0.2rem;" />
        <h1 class="title" style="color:#4CAF50; margin:0;">
            Credit Rating Predictor
        </h1>
        <p class="subtitle">
            Predict issuer ratings based on key financial indicators
        </p>
    </div>
""", unsafe_allow_html=True)

# 4) Load models & encoders
model = joblib.load('credit_rating_model.pkl')
rating_encoder   = joblib.load('rating_encoder.pkl')
industry_encoder = joblib.load('industry_encoder.pkl')

# 5) Ensure historical data file exists (no Issuer Name column)
historical_data_path = 'Simulated_CreditRating_Data.csv'
columns = [
    'Industry','Debt to Equity','EBITDA Margin',
    'Interest Coverage','Issue Size (₹Cr)','DefaultFlag','Predicted Rating'
]
if not os.path.exists(historical_data_path):
    pd.DataFrame(columns=columns).to_csv(historical_data_path, index=False)

# 6) Initialize session_state defaults (drop issuer_name)
for key, default in {
    'industry':         "Select Industry",
    'debt_to_equity':    0.0,
    'ebitda_margin':     0.0,
    'interest_coverage': 0.0,
    'issue_size':        0.0
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# 7) Build form inputs
industry_list = ["Select Industry"] + sorted(industry_encoder.classes_)

col1, col2 = st.columns([1, 2])
with col1:
    industry = st.selectbox(
        "🏭 Industry",
        industry_list,
        index=industry_list.index(st.session_state['industry']),
        key="industry"
    )
with col2:
    debt_to_equity    = st.number_input("📉 Debt to Equity Ratio", step=0.01, key="debt_to_equity")
    ebitda_margin     = st.number_input("💰 EBITDA Margin (%)", step=0.01, key="ebitda_margin")
    interest_coverage = st.number_input("🧾 Interest Coverage Ratio", step=0.01, key="interest_coverage")
    issue_size        = st.number_input("📦 Issue Size (₹ Crores)", step=1.0, key="issue_size")

default_flag = 0  # hidden from UI

# 8) Prediction & Clear buttons side by side
st.markdown('<div style="text-align:center; margin-top:2rem;">', unsafe_allow_html=True)
btn_col1, btn_col2 = st.columns([1,1])

with btn_col1:
    if st.button("🔍 Predict Credit Rating"):
        try:
            if industry == "Select Industry":
                st.warning("⚠️ Please select an Industry before predicting.")
            else:
                # encode industry
                industry_idx = industry_encoder.transform([industry])[0]
                # prepare features (no issuer_idx)
                X_new = np.array([
                    debt_to_equity,
                    ebitda_margin,
                    interest_coverage,
                    issue_size,
                    industry_idx,
                    default_flag
                ]).reshape(1, -1)

                # sanity check feature count
                if X_new.shape[1] != model.n_features_in_:
                    raise ValueError(f"Expected {model.n_features_in_} features, got {X_new.shape[1]}")

                # predict
                y_pred = model.predict(X_new)
                rating = rating_encoder.inverse_transform(y_pred)[0]
                st.success(f"🎯 Predicted Credit Rating: **{rating}**")

                # append to CSV
                new_row = pd.DataFrame({
                    'Industry':         [industry],
                    'Debt to Equity':   [debt_to_equity],
                    'EBITDA Margin':    [ebitda_margin],
                    'Interest Coverage':[interest_coverage],
                    'Issue Size (₹Cr)': [issue_size],
                    'DefaultFlag':      [default_flag],
                    'Predicted Rating': [rating]
                })
                new_row.to_csv(historical_data_path, mode='a', header=False, index=False)

        except Exception as e:
            st.error(f"❌ Prediction error: {e}")

with btn_col2:
    if st.button("🧹 Clear Inputs"):
        for k in ['industry','debt_to_equity','ebitda_margin','interest_coverage','issue_size']:
            st.session_state[k] = "Select Industry" if k=='industry' else 0.0
        st.experimental_rerun()

st.markdown('</div>', unsafe_allow_html=True)

# 9) Historical data expander
st.markdown('<div class="historical-data">', unsafe_allow_html=True)
with st.expander("📜 Show Historical Data"):
    df_hist = pd.read_csv(historical_data_path)
    st.dataframe(df_hist)
st.markdown('</div>', unsafe_allow_html=True)

# 10) Footer
st.markdown("""
<div class="footer">
    <hr style="margin-top: 2rem; margin-bottom: 1rem;" />
    <p>🔒 Secure & Private | 🏦 Powered by ML | 💡 BWR </p>
</div>
""", unsafe_allow_html=True)
