import os
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# 1) Page config
st.set_page_config(page_title="Credit Rating Predictor", layout="centered")

# 2) Custom CSS
st.markdown("""
    <style>
    body {
        background: #f9f9f9;
    }
    .block-container {
        padding-top: 1rem !important;
    }
    .title-container {
        text-align: center;
        padding: 1rem 0;
        border-bottom: 2px solid #4CAF50;
        margin-bottom: 2rem;
    }
    .title-container img {
        width: 35px;
        margin-bottom: -8px;
    }
    .title-container h1 {
        color: #4CAF50;
        font-size: 2rem;
        margin: 0.3rem 0 0.1rem 0;
    }
    .subtitle {
        color: #666;
        font-size: 0.95rem;
        margin-top: 0;
    }
    .stSelectbox > div, .stNumberInput > div {
        max-width: 400px;
        margin-bottom: 1rem;
    }
    .button-container {
        text-align: center;
        margin-top: 2rem;
    }
    .stButton>button {
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
        font-size: 0.95rem;
    }
    .predict-btn {
        background-color: #4CAF50;
        color: white;
        margin-right: 10px;
    }
    .predict-btn:hover {
        background-color: #45a049;
    }
    .clear-btn {
        background-color: #e0e0e0;
        color: #333;
    }
    .clear-btn:hover {
        background-color: #cfcfcf;
    }
    .footer {
        text-align: center;
        font-size: 0.85rem;
        color: #888;
        margin-top: 4rem;
    }
    .historical {
        margin-top: 3rem;
    }
    </style>
""", unsafe_allow_html=True)

# 3) Header
st.markdown("""
    <div class="title-container">
        <img src="https://cdn-icons-png.flaticon.com/512/2331/2331970.png" />
        <h1>Credit Rating Predictor</h1>
        <p class="subtitle">Predict issuer ratings based on financial inputs</p>
    </div>
""", unsafe_allow_html=True)

# 4) Load models & encoders
model = joblib.load('credit_rating_model.pkl')
rating_encoder = joblib.load('rating_encoder.pkl')
issuer_encoder = joblib.load('issuer_encoder.pkl')
industry_encoder = joblib.load('industry_encoder.pkl')

# 5) Ensure historical data file exists
historical_data_path = 'Simulated_CreditRating_Data.csv'
columns = [
    'Issuer Name','Industry','Debt to Equity','EBITDA Margin',
    'Interest Coverage','Issue Size (₹Cr)','DefaultFlag','Predicted Rating'
]
if not os.path.exists(historical_data_path):
    pd.DataFrame(columns=columns).to_csv(historical_data_path, index=False)

# 6) Defaults
for key, default in {
    'issuer_name': "Select Issuer Name",
    'industry': "Select Industry",
    'debt_to_equity': 0.0,
    'ebitda_margin': 0.0,
    'interest_coverage': 0.0,
    'issue_size': 0.0
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# 7) Inputs
issuer_list = ["Select Issuer Name"] + list(issuer_encoder.classes_)
industry_list = ["Select Industry"] + sorted(industry_encoder.classes_)

st.selectbox("🏢 Issuer Name", issuer_list,
             index=issuer_list.index(st.session_state['issuer_name']),
             key="issuer_name")

st.selectbox("🏭 Industry", industry_list,
             index=industry_list.index(st.session_state['industry']),
             key="industry")

st.number_input("📉 Debt to Equity Ratio", step=0.01, key="debt_to_equity")
st.number_input("💰 EBITDA Margin (%)", step=0.01, key="ebitda_margin")
st.number_input("🧾 Interest Coverage Ratio", step=0.01, key="interest_coverage")
st.number_input("📦 Issue Size (₹ Crores)", step=1.0, key="issue_size")

default_flag = 0  # hidden

# 8) Buttons
st.markdown('<div class="button-container">', unsafe_allow_html=True)
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("🔍 Predict Credit Rating", key="predict"):
        try:
            if st.session_state['issuer_name'] == "Select Issuer Name" or st.session_state['industry'] == "Select Industry":
                st.warning("⚠️ Please select both Issuer Name and Industry before predicting.")
            else:
                # prepare inputs
                issuer_idx = issuer_encoder.transform([st.session_state['issuer_name']])[0]
                industry_idx = industry_encoder.transform([st.session_state['industry']])[0]
                X_new = np.array([[st.session_state['debt_to_equity'],
                                   st.session_state['ebitda_margin'],
                                   st.session_state['interest_coverage'],
                                   st.session_state['issue_size'],
                                   issuer_idx,
                                   industry_idx,
                                   default_flag]])
                # predict
                if X_new.shape[1] != model.n_features_in_:
                    raise ValueError("Incorrect number of features.")
                y_pred = model.predict(X_new)
                rating = rating_encoder.inverse_transform(y_pred)[0]
                st.success(f"🎯 Predicted Credit Rating: **{rating}**")
                # Save result
                new_row = pd.DataFrame({
                    'Issuer Name': [st.session_state['issuer_name']],
                    'Industry': [st.session_state['industry']],
                    'Debt to Equity': [st.session_state['debt_to_equity']],
                    'EBITDA Margin': [st.session_state['ebitda_margin']],
                    'Interest Coverage': [st.session_state['interest_coverage']],
                    'Issue Size (₹Cr)': [st.session_state['issue_size']],
                    'DefaultFlag': [default_flag],
                    'Predicted Rating': [rating]
                })
                new_row.to_csv(historical_data_path, mode='a', header=False, index=False)
        except Exception as e:
            st.error(f"❌ Error: {e}")

with col2:
    if st.button("🧹 Clear Inputs", key="clear"):
        st.session_state.update({
            'issuer_name': "Select Issuer Name",
            'industry': "Select Industry",
            'debt_to_equity': 0.0,
            'ebitda_margin': 0.0,
            'interest_coverage': 0.0,
            'issue_size': 0.0
        })
        st.experimental_rerun()

st.markdown('</div>', unsafe_allow_html=True)

# 9) Historical Data
st.markdown('<div class="historical">', unsafe_allow_html=True)
with st.expander("📜 View Historical Predictions"):
    df_hist = pd.read_csv(historical_data_path)
    st.dataframe(df_hist)
st.markdown('</div>', unsafe_allow_html=True)

# 10) Footer
st.markdown("""
<div class="footer">
    <hr />
    🔒 Secure | ⚙️ ML-Powered | 🏦 Brickwork Ratings
</div>
""", unsafe_allow_html=True)
