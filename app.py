import os
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# 1) Page config
st.set_page_config(page_title="Credit Rating Predictor", layout="centered")

# 2) Custom CSS (unchanged, retained)
# [ ... keep your current custom CSS block here as-is ... ]

# 3) Header with hover-only subtitle (unchanged)
# [ ... keep your header markdown block here as-is ... ]

# 4) Load models & encoders
model = joblib.load('credit_rating_model.pkl')
rating_encoder = joblib.load('rating_encoder.pkl')
industry_encoder = joblib.load('industry_encoder.pkl')

# 5) Ensure historical data file exists
historical_data_path = 'Simulated_CreditRating_Data.csv'
columns = [
    'Industry','Debt to Equity','EBITDA Margin',
    'Interest Coverage','Issue Size (₹Cr)','DefaultFlag','Predicted Rating'
]
if not os.path.exists(historical_data_path):
    pd.DataFrame(columns=columns).to_csv(historical_data_path, index=False)

# 6) Initialize session_state defaults
for key, default in {
    'industry':     "Select Industry",
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
    industry = st.selectbox("🏭 Industry", industry_list,
                            index=industry_list.index(st.session_state['industry']),
                            key="industry")
with col2:
    debt_to_equity   = st.number_input("📉 Debt to Equity Ratio",
                                       step=0.01,
                                       key="debt_to_equity")
    ebitda_margin    = st.number_input("💰 EBITDA Margin (%)",
                                       step=0.01,
                                       key="ebitda_margin")
    interest_coverage= st.number_input("🧾 Interest Coverage Ratio",
                                       step=0.01,
                                       key="interest_coverage")
    issue_size       = st.number_input("📦 Issue Size (₹ Crores)",
                                       step=1.0,
                                       key="issue_size")

default_flag = 0  # hidden from UI

# 8) Prediction & Clear buttons side by side
st.markdown('<div style="text-align:center; margin-top:2rem;">', unsafe_allow_html=True)
btn_col1, btn_col2 = st.columns([1,1])

with btn_col1:
    if st.button("🔍 Predict Credit Rating"):
        try:
            if industry == "Select Industry":
                st.warning("⚠️ Please select Industry before predicting.")
            else:
                # prepare features
                industry_idx = industry_encoder.transform([industry])[0]
                X_new = np.array([[debt_to_equity,
                                   ebitda_margin,
                                   interest_coverage,
                                   issue_size,
                                   industry_idx,
                                   default_flag]]).reshape(1,-1)
                # predict
                if X_new.shape[1] != model.n_features_in_:
                    raise ValueError(f"Expected {model.n_features_in_} features, got {X_new.shape[1]}")
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
        st.markdown("<meta http-equiv='refresh' content='0; url=/' />", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# 9) Historical data expander
st.markdown('<div class="historical-data">', unsafe_allow_html=True)
with st.expander("📜 Show Historical Data"):
    df_hist = pd.read_csv(historical_data_path)
    st.dataframe(df_hist)
st.markdown('</div>', unsafe_allow_html=True)

# 10) Footer (unchanged)
# [ ... retain your footer block here ... ]
