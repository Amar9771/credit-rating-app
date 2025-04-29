import os
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# 1) Streamlit page configuration (must be first)
st.set_page_config(page_title="Credit Rating Predictor", layout="centered")

# 2) Custom CSS for a professional, official look
st.markdown("""
<style>
body {
    background-color: #f5f5f5;
    padding-top: 0;
}
.block-container {
    max-width: 720px;
    margin: 2rem auto !important;
    border: 1px solid #333 !important;
    border-radius: 8px !important;
    background-color: #fff;
    padding: 2rem !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}
.block-container h1 {
    text-align: center;
    color: #333;
    margin-bottom: 0.5rem;
}
input[type="text"],
input[type="number"],
select {
    max-width: 300px !important;
}
button[kind="primary"] {
    border-radius: 8px !important;
    padding: 8px 16px !important;
    background-color: #0052cc;
    color: white;
    border: none;
    font-weight: bold;
    transition: background-color 0.3s ease;
}
button[kind="primary"]:hover {
    background-color: #0041a8;
}
.footer {
    text-align: center;
    margin-top: 2rem;
    font-size: 0.9rem;
    color: #555;
}
</style>
""", unsafe_allow_html=True)

# 3) Header with Logo and Title
st.markdown("""
<div style="text-align: center;">
    <img src="https://cdn-icons-png.flaticon.com/512/2331/2331970.png"
         width="48" style="vertical-align: middle;" />
    <h1 style="display: inline; margin-left: 0.5rem; color: #333;">Credit Rating Predictor</h1>
    <p style="color: #666; margin-top: 4px;">Predict issuer ratings based on key financial indicators</p>
</div>
""", unsafe_allow_html=True)

# 4) Load model and encoders
try:
    model            = joblib.load('credit_rating_model.pkl')
    rating_encoder   = joblib.load('rating_encoder.pkl')
    issuer_encoder   = joblib.load('issuer_encoder.pkl')
    industry_encoder = joblib.load('industry_encoder.pkl')
except Exception:
    st.error("❌ Failed to load model or encoders.")
    st.stop()

# 5) CSV setup for historical data
hist_csv = 'Simulated_CreditRating_Data.csv'
columns = [
    'Issuer Name','Industry','Debt to Equity','EBITDA Margin',
    'Interest Coverage','Issue Size (₹Cr)','DefaultFlag','Predicted Rating'
]
if not os.path.exists(hist_csv):
    pd.DataFrame(columns=columns).to_csv(hist_csv, index=False)

# 6) Input form (including Default Flag) with session state keys
if 'issuer_name' not in st.session_state:
    st.session_state['issuer_name'] = ''
if 'industry' not in st.session_state:
    st.session_state['industry'] = industry_encoder.classes_[0]
if 'default_flag' not in st.session_state:
    st.session_state['default_flag'] = 'No'
if 'debt_to_equity' not in st.session_state:
    st.session_state['debt_to_equity'] = 0.0
if 'ebitda_margin' not in st.session_state:
    st.session_state['ebitda_margin'] = 0.0
if 'interest_coverage' not in st.session_state:
    st.session_state['interest_coverage'] = 0.0
if 'issue_size' not in st.session_state:
    st.session_state['issue_size'] = 0.0

col1, col2 = st.columns(2)
with col1:
    issuer_name = st.text_input("🏢 Issuer Name", key='issuer_name')
    industry    = st.selectbox("🏭 Industry", sorted(industry_encoder.classes_), key='industry')
    default_flag = st.selectbox("⚠️ Default Flag", ["No", "Yes"], key='default_flag')
with col2:
    debt_to_equity    = st.number_input("📉 Debt to Equity Ratio", step=0.01, key='debt_to_equity')
    ebitda_margin     = st.number_input("💰 EBITDA Margin (%)", step=0.01, key='ebitda_margin')
    interest_coverage = st.number_input("🧾 Interest Coverage Ratio", step=0.01, key='interest_coverage')
    issue_size        = st.number_input("📦 Issue Size (₹ Crores)", step=1.0, key='issue_size')

# Convert Default Flag to numeric
default_flag_num = 1 if st.session_state['default_flag'] == "Yes" else 0

# 7) Prediction logic
if st.button("🔍 Predict Credit Rating"):
    try:
        # Encode issuer (unknown issuers mapped to -1)
        if st.session_state['issuer_name'] in issuer_encoder.classes_:
            issuer_idx = issuer_encoder.transform([st.session_state['issuer_name']])[0]
        else:
            issuer_idx = -1

        industry_idx = industry_encoder.transform([st.session_state['industry']])[0]

        # Prepare feature vector in same order as training
        X_new = np.array([[
            issuer_idx,
            industry_idx,
            st.session_state['debt_to_equity'],
            st.session_state['ebitda_margin'],
            st.session_state['interest_coverage'],
            st.session_state['issue_size'],
            default_flag_num
        ]])

        # Verify feature count
        if X_new.shape[1] != model.n_features_in_:
            raise ValueError(
                f"Input features mismatch: Expected {model.n_features_in_} features, but got {X_new.shape[1]}"
            )

        # Predict
        y_pred = model.predict(X_new)
        rating = rating_encoder.inverse_transform(y_pred)[0]
        st.success(f"🎯 Predicted Credit Rating: **{rating}**")

        # Append to CSV
        new_row = pd.DataFrame([{  
            'Issuer Name':    st.session_state['issuer_name'],
            'Industry':       st.session_state['industry'],
            'Debt to Equity': st.session_state['debt_to_equity'],
            'EBITDA Margin':  st.session_state['ebitda_margin'],
            'Interest Coverage': st.session_state['interest_coverage'],
            'Issue Size (₹Cr)': st.session_state['issue_size'],
            'DefaultFlag':    st.session_state['default_flag'],
            'Predicted Rating': rating
        }])
        new_row.to_csv(hist_csv, mode='a', header=False, index=False)

        # Clear inputs
        st.session_state['issuer_name'] = ''
        st.session_state['industry'] = industry_encoder.classes_[0]
        st.session_state['default_flag'] = 'No'
        st.session_state['debt_to_equity'] = 0.0
        st.session_state['ebitda_margin'] = 0.0
        st.session_state['interest_coverage'] = 0.0
        st.session_state['issue_size'] = 0.0

    except Exception as e:
        st.error(f"❌ Prediction error: {e}")

# 8) Historical data expander
with st.expander("📜 Show Historical Data"):
    hist_df = pd.read_csv(hist_csv)
    st.dataframe(hist_df)

# 9) Footer
st.markdown("""
<div class="footer">
    <hr style="margin-top: 2rem;" />
    <p>🔒 Secure & Private | 🏦 Powered by ML | 💡 Created by Your Name</p>
</div>
""", unsafe_allow_html=True)
