# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# 1) Page Config
st.set_page_config(page_title="Credit Rating Prediction", layout="centered")

# 2) Load model and encoders
model = joblib.load("rating_model.pkl")
issuer_encoder = joblib.load("issuer_encoder.pkl")
industry_encoder = joblib.load("industry_encoder.pkl")
rating_encoder = joblib.load("rating_encoder.pkl")

# 3) Load Issuers and Industries from Training Data
issuer_list = issuer_encoder.classes_.tolist()
industry_list = industry_encoder.classes_.tolist()

# 4) Title and Description
st.title("🏦 Credit Rating Prediction App")
st.markdown("Use financial indicators and metadata to predict the expected credit rating for an issuer.")

# 5) User Inputs
st.subheader("Enter Input Parameters")

col1, col2 = st.columns(2)

with col1:
    debt_to_equity = st.number_input("Debt to Equity Ratio", min_value=0.0, max_value=10.0, value=1.5, step=0.1)
    interest_coverage = st.number_input("Interest Coverage Ratio", min_value=0.0, max_value=20.0, value=3.0, step=0.1)
    issuer_name = st.selectbox("Issuer Name", options=["Select Issuer Name"] + issuer_list)

with col2:
    ebitda_margin = st.number_input("EBITDA Margin (%)", min_value=-100.0, max_value=100.0, value=10.0, step=0.1)
    issue_size = st.number_input("Issue Size (₹ Crores)", min_value=1.0, max_value=10000.0, value=100.0, step=1.0)
    industry = st.selectbox("Industry", options=["Select Industry"] + industry_list)

default_flag = st.selectbox("Has Default History?", options=["No", "Yes"])
default_flag = 1 if default_flag == "Yes" else 0

# 6) Show Inputs
st.markdown("---")
st.subheader("🔎 Review Input Summary")
input_df = pd.DataFrame({
    "Debt to Equity": [debt_to_equity],
    "EBITDA Margin (%)": [ebitda_margin],
    "Interest Coverage": [interest_coverage],
    "Issue Size (₹Cr)": [issue_size],
    "Issuer": [issuer_name],
    "Industry": [industry],
    "DefaultFlag": ["Yes" if default_flag else "No"]
})
st.table(input_df)

# 7) Historical Data Path
historical_data_path = "prediction_history.csv"
if not os.path.exists(historical_data_path):
    pd.DataFrame(columns=[
        'Issuer Name', 'Industry', 'Debt to Equity', 'EBITDA Margin',
        'Interest Coverage', 'Issue Size (₹Cr)', 'DefaultFlag', 'Predicted Rating'
    ]).to_csv(historical_data_path, index=False)

# 8) Buttons: Predict and Clear Inputs
st.markdown('<div style="text-align: center; margin-top: 2rem;">', unsafe_allow_html=True)
col_predict, col_clear = st.columns([1, 1])

with col_predict:
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

with col_clear:
    if st.button("❌ Clear Inputs"):
        st.markdown(f"<meta http-equiv='refresh' content='0; url=/' />", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# 11) View Historical Predictions
st.markdown("---")
with st.expander("📜 View Previous Predictions"):
    try:
        df_history = pd.read_csv(historical_data_path)
        st.dataframe(df_history, use_container_width=True)
    except Exception as e:
        st.error(f"Failed to load history: {e}")
