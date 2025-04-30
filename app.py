import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# Dummy models and encoders (Replace with actual ones in your environment)
issuer_encoder = LabelEncoder()
industry_encoder = LabelEncoder()
rating_encoder = LabelEncoder()

# Example issuer names and industries (Replace with actual data)
existing_issuer_names = ['Issuer A', 'Issuer B', 'Issuer C']
existing_industries = ['Industry X', 'Industry Y', 'Industry Z']
existing_ratings = ['AAA', 'AA', 'A']

# Sample model (Replace with your actual model)
model = LogisticRegression()

# File paths for saving historical data (Replace with your actual paths)
historical_data_path = 'historical_data.csv'

# Initialize session_state values
if 'issuer_name' not in st.session_state:
    st.session_state.issuer_name = ""
if 'industry' not in st.session_state:
    st.session_state.industry = industry_encoder.classes_[0] if industry_encoder.classes_ else ""
if 'default_flag' not in st.session_state:
    st.session_state.default_flag = 0
if 'debt_to_equity' not in st.session_state:
    st.session_state.debt_to_equity = 0.0
if 'ebitda_margin' not in st.session_state:
    st.session_state.ebitda_margin = 0.0
if 'interest_coverage' not in st.session_state:
    st.session_state.interest_coverage = 0.0
if 'issue_size' not in st.session_state:
    st.session_state.issue_size = 0.0

# Input form layout
col1, col2 = st.columns([1, 2])

with col1:
    # Dropdown for Issuer Name with option to type a new name
    issuer_name = st.selectbox("🏢 Issuer Name", options=existing_issuer_names + ['Enter New Issuer Name'])
    if issuer_name == 'Enter New Issuer Name':
        issuer_name = st.text_input("Please enter the Issuer Name:")

    industry = st.selectbox("🏭 Industry", sorted(existing_industries), index=sorted(existing_industries).index(st.session_state.industry) if existing_industries else 0)
    default_flag = st.selectbox("⚠️ Default Flag", [0, 1], help="Set to 1 if issuer has defaulted, else 0", index=[0, 1].index(st.session_state.default_flag))

with col2:
    debt_to_equity = st.number_input("📉 Debt to Equity Ratio", step=0.01, value=st.session_state.debt_to_equity)
    ebitda_margin = st.number_input("💰 EBITDA Margin (%)", step=0.01, value=st.session_state.ebitda_margin)
    interest_coverage = st.number_input("🧾 Interest Coverage Ratio", step=0.01, value=st.session_state.interest_coverage)
    issue_size = st.number_input("📦 Issue Size (₹ Crores)", step=1.0, value=st.session_state.issue_size)

# Prediction button logic
if st.button("🔍 Predict Credit Rating"):
    try:
        # Handle known issuers or allow manual input
        if issuer_name in issuer_encoder.classes_:
            issuer_idx = issuer_encoder.transform([issuer_name])[0]
        else:
            if issuer_name:  # Only proceed if the user entered a valid issuer name
                issuer_idx = len(issuer_encoder.classes_)  # Dynamically assign an index for new issuer
                # You could update the encoder with new issuer data if needed
            else:
                st.error("Please enter a valid issuer name.")
                st.stop()

        # Encode industry correctly
        industry_idx = industry_encoder.transform([industry])[0] if industry else 0

        # Prepare input features for prediction, including DefaultFlag
        X_new = np.array([[debt_to_equity, ebitda_margin, interest_coverage, issue_size,
                           issuer_idx, industry_idx, default_flag]]).reshape(1, -1)

        # Perform prediction
        y_pred = model.predict(X_new)
        rating = rating_encoder.inverse_transform(y_pred)[0]
        st.success(f"🎯 Predicted Credit Rating: **{rating}**")

        # Append to CSV (historical data)
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

        # Reset the input values in session_state
        st.session_state.issuer_name = ""
        st.session_state.industry = industry_encoder.classes_[0] if industry_encoder.classes_ else ""
        st.session_state.default_flag = 0
        st.session_state.debt_to_equity = 0.0
        st.session_state.ebitda_margin = 0.0
        st.session_state.interest_coverage = 0.0
        st.session_state.issue_size = 0.0

        # Clear the input fields visually
        st.experimental_rerun()

    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
