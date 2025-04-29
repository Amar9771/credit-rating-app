import os
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------
# 1) Streamlit must call set_page_config() first
# ------------------------------------------------------------------------------
st.set_page_config(page_title="Credit Rating Predictor", layout="centered")

# ------------------------------------------------------------------------------
# 2) Custom CSS for Styling
# ------------------------------------------------------------------------------
st.markdown("""
    <style>
    /* your existing CSS… */
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 3) Header with Logo and Title
# ------------------------------------------------------------------------------
st.markdown("""
    <div style="text-align: center;">
      <!-- your logo and headings… -->
    </div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 4) Load model and encoders
# ------------------------------------------------------------------------------
model            = joblib.load('credit_rating_model.pkl')
rating_encoder   = joblib.load('rating_encoder.pkl')
issuer_encoder   = joblib.load('issuer_encoder.pkl')
industry_encoder = joblib.load('industry_encoder.pkl')

# ------------------------------------------------------------------------------
# 5) Historical data: detect CSV or Excel, else create blank CSV
# ------------------------------------------------------------------------------
base_dir   = r'F:\credit_rating_app'
base_name  = 'Simulated_CreditRating_Data'
csv_path   = os.path.join(base_dir, f'{base_name}.csv')
xlsx_path  = os.path.join(base_dir, f'{base_name}.xlsx')
columns    = [
    'Issuer Name','Industry','Debt to Equity','EBITDA Margin',
    'Interest Coverage','Issue Size (₹Cr)','DefaultFlag','Predicted Rating'
]

# ensure base folder exists
os.makedirs(base_dir, exist_ok=True)

# load historical_data, preferring CSV
if os.path.exists(csv_path):
    historical_data = pd.read_csv(csv_path)
elif os.path.exists(xlsx_path):
    historical_data = pd.read_excel(xlsx_path)
else:
    # create an empty CSV for future appends
    historical_data = pd.DataFrame(columns=columns)
    historical_data.to_csv(csv_path, index=False)

# preview
st.write("📜 Historical Data Loaded:")
st.dataframe(historical_data.head())

# ------------------------------------------------------------------------------
# 6) Input form (two columns + Default Flag)
# ------------------------------------------------------------------------------
col1, col2 = st.columns(2)
with col1:
    issuer_name    = st.text_input("🏢 Issuer Name")
    industry       = st.selectbox("🏭 Industry", sorted(industry_encoder.classes_))
    default_flag   = st.selectbox("⚠️ Default Flag", ["No","Yes"])
with col2:
    debt_to_equity    = st.number_input("📉 Debt to Equity Ratio", step=0.01)
    ebitda_margin     = st.number_input("💰 EBITDA Margin (%)", step=0.01)
    interest_coverage = st.number_input("🧾 Interest Coverage Ratio", step=0.01)
    issue_size        = st.number_input("📦 Issue Size (₹ Crores)", step=1.0)

# map DefaultFlag to 0/1
default_flag_num = 1 if default_flag == "Yes" else 0

# ------------------------------------------------------------------------------
# 7) Prediction logic: now 7 features in the same order as training
# ------------------------------------------------------------------------------

if st.button("🔍 Predict Credit Rating"):
    try:
        # encode issuer (or new → append class array)
        if issuer_name in issuer_encoder.classes_:
            issuer_idx = issuer_encoder.transform([issuer_name])[0]
        else:
            issuer_encoder.classes_ = np.append(issuer_encoder.classes_, issuer_name)
            issuer_idx = issuer_encoder.transform([issuer_name])[0]

        # encode industry
        industry_idx = industry_encoder.transform([industry])[0]

        # assemble features in the exact same column order:
        # ['Issuer Encoded','Industry Encoded','Debt to Equity','EBITDA Margin',
        #  'Interest Coverage','Issue Size (₹Cr)','DefaultFlag']
        X_new = np.array([[
            issuer_idx,
            industry_idx,
            debt_to_equity,
            ebitda_margin,
            interest_coverage,
            issue_size,
            default_flag_num
        ]])

        # predict & decode
        y_pred = model.predict(X_new)
        rating = rating_encoder.inverse_transform(y_pred)[0]
        st.success(f"🎯 Predicted Credit Rating: **{rating}**")

        # append new row to historical_data & to CSV
        new_row = pd.DataFrame([{
            'Issuer Name':    issuer_name,
            'Industry':       industry,
            'Debt to Equity': debt_to_equity,
            'EBITDA Margin':  ebitda_margin,
            'Interest Coverage': interest_coverage,
            'Issue Size (₹Cr)': issue_size,
            'DefaultFlag':    default_flag,
            'Predicted Rating': rating
        }])
        new_row.to_csv(csv_path, mode='a', header=False, index=False)

    except Exception as e:
        st.error(f"❌ Prediction error: {e}")

# ------------------------------------------------------------------------------
# 8) Show historical data
# ------------------------------------------------------------------------------
with st.expander("📜 Show Historical Data"):
    st.dataframe(pd.read_csv(csv_path))

# ------------------------------------------------------------------------------
# 9) Footer
# ------------------------------------------------------------------------------
st.markdown("""
  <div class="footer">
    <hr />
    <p>🔒 Secure & Private | 🏦 Powered by ML</p>
  </div>
""", unsafe_allow_html=True)
