import os
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# 1) Page config
st.set_page_config(page_title="Credit Rating Predictor", layout="centered")

# 2) Custom CSS (hover subtitle, styling, etc.)
st.markdown("""
    <style>
    body {
        background: linear-gradient(to right, #e0f2f1, #ffffff);
    }
    .block-container {
        max-width: 700px;
        margin: auto;
        border: 2px solid #4CAF50;
        border-radius: 0 0 15px 15px;
        background-color: white;
        padding: 2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .subtitle {
        visibility: hidden;
        opacity: 0;
        transition: 0.3s;
    }
    .title:hover + .subtitle {
        visibility: visible;
        opacity: 1;
        color: #666;
        font-size: 1rem;
        margin-top: .2rem;
    }
    input[type="number"], select { max-width: 300px !important; }
    button[kind="primary"] {
        border-radius: 12px; padding: 10px 20px;
        background-color: #4CAF50; color: white; font-weight: bold;
    }
    button[kind="primary"]:hover { background-color: #45a049; }
    .footer { text-align: center; margin-top: 2rem; color: #777; font-size: .9rem; }
    .historical-data { margin-top: 4rem; }
    </style>
""", unsafe_allow_html=True)

# 3) Header
st.markdown("""
    <div style="text-align:center; margin-bottom:1rem;">
      <img src="https://cdn-icons-png.flaticon.com/512/2331/2331970.png" width="35" />
      <h1 class="title" style="color:#4CAF50; margin:0;">Credit Rating Predictor</h1>
      <p class="subtitle">Predict issuer ratings based on key financial indicators</p>
    </div>
""", unsafe_allow_html=True)

# 4) Load models & encoders
model            = joblib.load('credit_rating_model.pkl')
rating_encoder   = joblib.load('rating_encoder.pkl')
issuer_encoder   = joblib.load('issuer_encoder.pkl')
industry_encoder = joblib.load('industry_encoder.pkl')

# 5) Ensure history CSV exists
hist_path = 'Simulated_CreditRating_Data.csv'
cols = ['Issuer Name','Industry','Debt to Equity','EBITDA Margin',
        'Interest Coverage','Issue Size (₹Cr)','DefaultFlag','Predicted Rating']
if not os.path.exists(hist_path):
    pd.DataFrame(columns=cols).to_csv(hist_path, index=False)

# 6) Session state defaults
for k, default in {
    'industry':         "Select Industry",
    'debt_to_equity':   0.0,
    'ebitda_margin':    0.0,
    'interest_coverage':0.0,
    'issue_size':       0.0
}.items():
    if k not in st.session_state:
        st.session_state[k] = default

# ---- HERE: set issuer_name in code, no UI ----
issuer_name = issuer_encoder.classes_[0]   # pick your default
st.session_state['issuer_name'] = issuer_name

# 7) Build the rest of the form
industry_list = ["Select Industry"] + sorted(industry_encoder.classes_)

col1, col2 = st.columns([1,2])
with col1:
    industry = st.selectbox("🏭 Industry", industry_list,
                            index=industry_list.index(st.session_state['industry']),
                            key="industry")
with col2:
    debt_to_equity    = st.number_input("📉 Debt to Equity Ratio",    step=0.01, key="debt_to_equity")
    ebitda_margin     = st.number_input("💰 EBITDA Margin (%)",       step=0.01, key="ebitda_margin")
    interest_coverage = st.number_input("🧾 Interest Coverage Ratio",  step=0.01, key="interest_coverage")
    issue_size        = st.number_input("📦 Issue Size (₹ Crores)",     step=1.0,  key="issue_size")

default_flag = 0

# 8) Predict & Clear Buttons
st.markdown('<div style="text-align:center; margin-top:2rem;">', unsafe_allow_html=True)
b1, b2 = st.columns(2)
with b1:
    if st.button("🔍 Predict Credit Rating"):
        if industry == "Select Industry":
            st.warning("⚠️ Please select an Industry before predicting.")
        else:
            # assemble features
            issuer_idx   = issuer_encoder.transform([issuer_name])[0]
            industry_idx = industry_encoder.transform([industry])[0]
            X_new = np.array([[
                debt_to_equity,
                ebitda_margin,
                interest_coverage,
                issue_size,
                issuer_idx,
                industry_idx,
                default_flag
            ]])
            try:
                y_pred = model.predict(X_new)
                rating = rating_encoder.inverse_transform(y_pred)[0]
                st.success(f"🎯 Predicted Credit Rating: **{rating}**")
                # log
                pd.DataFrame({
                    'Issuer Name':      [issuer_name],
                    'Industry':         [industry],
                    'Debt to Equity':   [debt_to_equity],
                    'EBITDA Margin':    [ebitda_margin],
                    'Interest Coverage':[interest_coverage],
                    'Issue Size (₹Cr)': [issue_size],
                    'DefaultFlag':      [default_flag],
                    'Predicted Rating': [rating]
                }).to_csv(hist_path, mode='a', header=False, index=False)
            except Exception as e:
                st.error(f"❌ Prediction error: {e}")
with b2:
    if st.button("🧹 Clear Inputs"):
        st.experimental_rerun()
st.markdown('</div>', unsafe_allow_html=True)

# 9) Historical data expander
st.markdown('<div class="historical-data">', unsafe_allow_html=True)
with st.expander("📜 Show Historical Data"):
    st.dataframe(pd.read_csv(hist_path))
st.markdown('</div>', unsafe_allow_html=True)

# 10) Footer
st.markdown("""
  <div class="footer">
    <hr style="margin:1rem 0;" />
    <p>🔒 Secure & Private | 🏦 Powered by ML | 💡 BWR</p>
  </div>
""", unsafe_allow_html=True)
