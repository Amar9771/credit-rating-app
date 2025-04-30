import streamlit as st
import numpy as np
import joblib

# Load the pre-trained model and encoders
model = joblib.load('credit_rating_model.pkl')
issuer_encoder = joblib.load('issuer_encoder.pkl')
industry_encoder = joblib.load('industry_encoder.pkl')
rating_encoder = joblib.load('rating_encoder.pkl')

# Initialize session state for input values if they don't exist already
if 'issuer_name' not in st.session_state:
    st.session_state.issuer_name = ""
if 'industry' not in st.session_state:
    st.session_state.industry = industry_encoder.classes_[0]
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

# Layout: Two columns for inputs
col1, col2 = st.columns([1, 2])

with col1:
    # Issuer Name Dropdown
    issuer_name = st.selectbox("🏢 Issuer Name", sorted(issuer_encoder.classes_), index=sorted(issuer_encoder.classes_).index(st.session_state.issuer_name) if st.session_state.issuer_name in issuer_encoder.classes_ else 0)
    
    # Industry Dropdown
    industry = st.selectbox("🏭 Industry", sorted(industry_encoder.classes_), index=sorted(industry_encoder.classes_).index(st.session_state.industry))
    
    # Default Flag Dropdown
    default_flag = st.selectbox("⚠️ Default Flag", [0, 1], help="Set to 1 if issuer has defaulted, else 0", index=[0, 1].index(st.session_state.default_flag))

with col2:
    # Debt to Equity Ratio Input
    debt_to_equity = st.number_input("📉 Debt to Equity Ratio", step=0.01, value=st.session_state.debt_to_equity)
    
    # EBITDA Margin Input
    ebitda_margin = st.number_input("💰 EBITDA Margin (%)", step=0.01, value=st.session_state.ebitda_margin)
    
    # Interest Coverage Ratio Input
    interest_coverage = st.number_input("🧾 Interest Coverage Ratio", step=0.01, value=st.session_state.interest_coverage)
    
    # Issue Size Input
    issue_size = st.number_input("📦 Issue Size (₹ Crores)", step=1.0, value=st.session_state.issue_size)

# Button to trigger the prediction
if st.button("🔍 Predict Credit Rating"):
    try:
        # Ensure issuer is encoded correctly
        if issuer_name in issuer_encoder.classes_:
            issuer_idx = issuer_encoder.transform([issuer_name])[0]
        else:
            st.error(f"Issuer {issuer_name} is not recognized. Please choose a valid issuer.")
            issuer_idx = -1  # Handle unknown issuer
        
        # Encode industry correctly
        industry_idx = industry_encoder.transform([industry])[0]

        # Prepare input features for prediction, including DefaultFlag
        X_new = np.array([[debt_to_equity, ebitda_margin, interest_coverage, issue_size,
                           issuer_idx, industry_idx, default_flag]]).reshape(1, -1)

        # Perform prediction
        y_pred = model.predict(X_new)
        rating = rating_encoder.inverse_transform(y_pred)[0]
        st.success(f"🎯 Predicted Credit Rating: **{rating}**")
        
        # (Optional) You can append to CSV as previously done for historical data
        # Append to CSV file or database (Optional)
        
        # Reset session state and clear input fields
        st.session_state.issuer_name = ""
        st.session_state.industry = industry_encoder.classes_[0]
        st.session_state.default_flag = 0
        st.session_state.debt_to_equity = 0.0
        st.session_state.ebitda_margin = 0.0
        st.session_state.interest_coverage = 0.0
        st.session_state.issue_size = 0.0
        st.experimental_rerun()  # Clears input fields after submission

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.write(f"Full error traceback: {e.__traceback__}")
