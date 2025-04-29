import os
import traceback
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# 1) Must be first
st.set_page_config(page_title="Credit Rating Predictor", layout="centered")

# 1a) Show working dir for debugging
st.write("🗂️ Current working directory:", os.getcwd())

# 2) CSS (same as before)
st.markdown("""
    <style>
    /* …your existing CSS… */
    </style>
""", unsafe_allow_html=True)

# 3) Header
st.markdown("""
<div style="text-align: center;">
  <img src="https://cdn-icons-png.flaticon.com/512/2331/2331970.png" width="40" />
  <h1 style="color: #4CAF50; margin:0;">Credit Rating Predictor</h1>
  <p style="color:#666; margin-top:4px;">Predict issuer ratings based on key financial indicators</p>
</div>
""", unsafe_allow_html=True)

# 4) Load model & encoders
try:
    model            = joblib.load('credit_rating_model.pkl')
    rating_encoder   = joblib.load('rating_encoder.pkl')
    issuer_encoder   = joblib.load('issuer_encoder.pkl')
    industry_encoder = joblib.load('industry_encoder.pkl')
except Exception as e:
    st.error("❌ Failed to load model or encoders:\n" + traceback.format_exc())
    st.stop()

# 5) Historical CSV setup
hist_csv = 'Simulated_CreditRating_Data.csv'
cols = [
    'Issuer Name','Industry','Debt to Equity','EBITDA Margin',
    'Interest Coverage','Issue Size (₹Cr)','Predicted Rating'
]

# Create if missing
if not os.path.exists(hist_csv):
    try:
        pd.DataFrame(columns=cols).to_csv(hist_csv, index=False)
        st.info(f"ℹ️ Created new history file at `{hist_csv}`")
    except Exception as e:
        st.error("❌ Could not create history file:\n" + traceback.format_exc())
        st.stop()

# Load history
try:
    hist_df = pd.read_csv(hist_csv)
except Exception as e:
    st.warning("⚠️ Could not load historical data:\n" + str(e))
    hist_df = pd.DataFrame(columns=cols)

# Show preview
st.write("📜 Historical Data Preview:")
st.dataframe(hist_df.head())

# 6) Input form
col1, col2 = st.columns(2)
with col1:
    issuer_name = st.text_input("🏢 Issuer Name")
    industry    = st.selectbox("🏭 Industry", sorted(industry_encoder.classes_))
with col2:
    debt_to_equity    = st.number_input("📉 Debt to Equity Ratio", step=0.01)
    ebitda_margin     = st.number_input("💰 EBITDA Margin (%)", step=0.01)
    interest_coverage = st.number_input("🧾 Interest Coverage Ratio", step=0.01)
    issue_size        = st.number_input("📦 Issue Size (₹ Crores)", step=1.0)

# 7) Prediction
if st.button("🔍 Predict Credit Rating"):
    try:
        # encode issuer
        if issuer_name in issuer_encoder.classes_:
            issuer_idx = issuer_encoder.transform([issuer_name])[0]
        else:
            # optional: grow the encoder (but model may not know new issuers)
            issuer_encoder.classes_ = np.append(issuer_encoder.classes_, issuer_name)
            issuer_idx = issuer_encoder.transform([issuer_name])[0]

        industry_idx = industry_encoder.transform([industry])[0]

        X_new = np.array([[issuer_idx,
                           industry_idx,
                           debt_to_equity,
                           ebitda_margin,
                           interest_coverage,
                           issue_size]]).reshape(1, -1)

        y_pred = model.predict(X_new)
        rating = rating_encoder.inverse_transform(y_pred)[0]
        st.success(f"🎯 Predicted Credit Rating: **{rating}**")

        # append to CSV
        new_row = pd.DataFrame([{
            'Issuer Name':       issuer_name,
            'Industry':          industry,
            'Debt to Equity':    debt_to_equity,
            'EBITDA Margin':     ebitda_margin,
            'Interest Coverage': interest_coverage,
            'Issue Size (₹Cr)':  issue_size,
            'Predicted Rating':  rating
        }])
        new_row.to_csv(hist_csv, mode='a', header=False, index=False)
        st.info("✅ Saved prediction to history.")

    except Exception as e:
        st.error("❌ Prediction error:\n" + traceback.format_exc())

# 8) Show full historical data
with st.expander("📜 Full Historical Data"):
    st.dataframe(hist_df)

# 9) Footer
st.markdown("""
<div style="text-align: center; margin-top: 2rem; color:#777; font-size:0.9rem;">
  🔒 Secure & Private | 🏦 Powered by ML | 💡 Created by Your Name
</div>
""", unsafe_allow_html=True)
