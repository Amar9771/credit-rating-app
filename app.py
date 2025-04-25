import streamlit as st
import joblib
import numpy as np

# Load model and encoders
model = joblib.load('credit_rating_model.pkl')
rating_encoder = joblib.load('rating_encoder.pkl')
issuer_encoder = joblib.load('issuer_encoder.pkl')
industry_encoder = joblib.load('industry_encoder.pkl')

# Streamlit UI
st.set_page_config(page_title="Credit Rating Predictor", layout="centered")
st.title("🏦 Credit Rating Predictor")

# Inputs
issuer_name = st.text_input("Issuer Name")
industry = st.selectbox("Industry", sorted(industry_encoder.classes_))
debt_to_equity = st.number_input("Debt to Equity Ratio", step=0.01)
ebitda_margin = st.number_input("EBITDA Margin (%)", step=0.01)
interest_coverage = st.number_input("Interest Coverage Ratio", step=0.01)
issue_size = st.number_input("Issue Size (in ₹ Crores)", step=1.0)

# Predict button
if st.button("Predict Credit Rating"):
    try:
        # Add issuer dynamically if not seen before
        if issuer_name not in issuer_encoder.classes_:
            issuer_encoder.classes_ = np.append(issuer_encoder.classes_, issuer_name)

        # Encode inputs
        issuer_encoded = issuer_encoder.transform([issuer_name])[0]
        industry_encoded = industry_encoder.transform([industry])[0]

        # Prepare input
        input_array = np.array([[debt_to_equity, ebitda_margin, interest_coverage, issue_size, issuer_encoded, industry_encoded]])

        # Predict
        prediction = model.predict(input_array)
        predicted_rating = rating_encoder.inverse_transform(prediction)[0]

        # Show result
        st.success(f"🎯 Predicted Credit Rating: **{predicted_rating}**")

    except Exception as e:
        st.error(f"❌ Error: {e}")
