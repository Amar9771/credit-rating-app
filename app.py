import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Set the page configuration as the first command
st.set_page_config(page_title="Credit Rating Predictor", layout="centered")

# Load model and encoders
model = joblib.load('credit_rating_model.pkl')
rating_encoder = joblib.load('rating_encoder.pkl')
issuer_encoder = joblib.load('issuer_encoder.pkl')
industry_encoder = joblib.load('industry_encoder.pkl')

# Load historical credit rating data
historical_data_path = r'F:\credit_rating_app\Simulated_CreditRating_Data.csv'

# Check if the file exists and load the data
try:
    historical_data = pd.read_csv(historical_data_path)
    st.write("📜 Historical Data Loaded:")
    st.write(historical_data.head())  # Show a preview of the historical data
except FileNotFoundError:
    st.warning("⚠️ Historical data file not found. Predictions will be based on new inputs only.")

# Streamlit UI
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

        # Append the new prediction to historical data
        new_data = {
            'Issuer Name': [issuer_name],
            'Industry': [industry],
            'Debt to Equity': [debt_to_equity],
            'EBITDA Margin': [ebitda_margin],
            'Interest Coverage': [interest_coverage],
            'Issue Size (₹ Crores)': [issue_size],
            'Predicted Rating': [predicted_rating]
        }
        new_data_df = pd.DataFrame(new_data)

        # Append the new data to the historical data CSV
        new_data_df.to_csv(historical_data_path, mode='a', header=False, index=False)

    except Exception as e:
        st.error(f"❌ Error: {e}")
