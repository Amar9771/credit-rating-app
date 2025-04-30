import os
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# 1) Set Streamlit page config
st.set_page_config(page_title="Credit Rating Predictor", layout="centered")

# 2) Custom CSS
st.markdown("""<style>
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
    </style>""", unsafe_allow_html=True)

# 3) Header
st.markdown("""<div style="text-align: center; margin-bottom: 0rem;">
        <img src="https://cdn-icons-png.flaticon.com/512/2331/2331970.png"
             width="40" style="margin-bottom: 0px;" />
        <h1 style="color: #4CAF50; margin-bottom: 0.0rem;">
            Credit Rating Predictor
        </h1>
        <p style="color: #666; font-size: 1.0rem; margin-top: 0;">
            Predict issuer ratings based on key financial indicators
        </p>
    </div>""", unsafe_allow_html=True)

# 4) Load models & encoders
model = joblib.load('credit_rating_model.pkl')
rating_encoder = joblib.load('rating_encoder.pkl')
issuer_encoder = joblib.load('issuer_encoder.pkl')
industry_encoder = joblib.load('industry_encoder.pkl')

# 5) Data setup
historical_data_path = 'Simulated_CreditRating_Data.csv'
columns = [
    'Issuer Name','Industry','Debt to Equity','EBITDA Margin',
    'Interest Coverage','Issue Size (₹Cr)','DefaultFlag','Predicted Rating'
]

if not os.path.exists(historical_data_path):
    pd.DataFrame(columns=columns).to_csv(historical_data_path, index=False)

# 6) Initialize session state if not already done
if 'reset' not in st.session_state:
    st.session_state['reset'] = False

# 7) Prepare dropdown lists
issuer_list = ["Select Issuer Name"] + list(issuer_encoder.classes_)
industry_list = ["Select Industry"] + sorted(industry_encoder.classes_)

# 8) Form Layout
col1, col2 = st.columns([1, 2])

with col1:
    # Issuer Name and Industry selection
    issuer_name = st.selectbox("🏢 Issuer Name", issuer_list, index=0 if st.session_state['reset'] else issuer_list.index(st.session_state.get('issuer_name', 'Select Issuer Name')), key="issuer_name")
    industry = st.selectbox("🏭 Industry", industry_list, index=0 if st.session_state['reset'] else industry_list.index(st.session_state.get('industry', 'Select Industry')), key="industry")

with col2:
    # Financial inputs
    debt_to_equity = st.number_input("📉 Debt to Equity Ratio", step=0.01, value=st.session_state.get('debt_to_equity', 0.0), key="debt_to_equity")
    ebitda_margin = st.number_input("💰 EBITDA Margin (%)", step=0.01, value=st.session_state.get('ebitda_margin', 0.0), key="ebitda_margin")
    interest_coverage = st.number_input("🧾 Interest Coverage Ratio", step=0.01, value=st.session_state.get('interest_coverage', 0.0), key="interest_coverage")
    issue_size = st.number_input("📦 Issue Size (₹ Crores)", step=1.0, value=st.session_state.get('issue_size', 0.0), key="issue_size")

# Internally set default flag (hidden from UI)
default_flag = 0

# 9) Clear Input Button
if st.button("❌ Clear Inputs"):
    # Reset the session state
    st.session_state['reset'] = True
    st.session_state['issuer_name'] = "Select Issuer Name"
    st.session_state['industry'] = "Select Industry"
    st.session_state['debt_to_equity'] = 0.0
    st.session_state['ebitda_margin'] = 0.0
    st.session_state['interest_coverage'] = 0.0
    st.session_state['issue_size'] = 0.0
    st.experimental_rerun()  # Rerun the app to apply reset

# 10) Prediction Logic
st.markdown('<div style="text-align: center; margin-top: 2rem;">', unsafe_allow_html=True)
if st.button("🔍 Predict Credit Rating"):
    try:
        if issuer_name == "Select Issuer Name" or industry == "Select Industry":
            st.warning("⚠️ Please select both Issuer Name and Industry before predicting.")
        else:
            issuer_idx = issuer_encoder.transform([issuer_name])[0]
            industry_idx = industry_encoder.transform([industry])[0]

            X_new = np.array([[debt_to_equity, ebitda_margin, interest_coverage,
                               issue_size, issuer_idx, industry_idx, default_flag]]).reshape(1, -1)

            if X_new.shape[1] != model.n_features_in_:
                raise ValueError(f"Input features mismatch: Expected {model.n_features_in_} features, got {X_new.shape[1]}")

            y_pred = model.predict(X_new)
            rating = rating_encoder.inverse_transform(y_pred)[0]
            st.success(f"🎯 Predicted Credit Rating: **{rating}**")

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

    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
st.markdown('</div>', unsafe_allow_html=True)

# 11) Historical data
st.markdown('<div class="historical-data">', unsafe_allow_html=True)
with st.expander("📜 Show Historical Data"):
    hist_df = pd.read_csv(historical_data_path)
    st.dataframe(hist_df)
st.markdown('</div>', unsafe_allow_html=True)

# 12) Footer
st.markdown("""<div class="footer">
    <hr style="margin-top: 2rem; margin-bottom: 1rem;" />
    <p>🔒 Secure & Private | 🏦 Powered by ML | 💡 Created by Your Name</p>
</div>""", unsafe_allow_html=True)
