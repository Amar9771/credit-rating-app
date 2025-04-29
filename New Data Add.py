import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load data
data_path = "F:\\credit_rating_app\\Simulated_CreditRating_Data.csv"
df = pd.read_csv(data_path)

# Print columns for debug
print("Columns in dataset:", df.columns.tolist())

# Encode categorical features
issuer_encoder = LabelEncoder()
industry_encoder = LabelEncoder()
rating_encoder = LabelEncoder()

df['Issuer Encoded'] = issuer_encoder.fit_transform(df['Issuer Name'])
df['Industry Encoded'] = industry_encoder.fit_transform(df['Industry'])
df['Rating Encoded'] = rating_encoder.fit_transform(df['Final Rating'])

# Convert 'DefaultFlag' from Yes/No to 1/0
df['DefaultFlag'] = df['DefaultFlag'].map({'Yes': 1, 'No': 0})

# Save encoders
joblib.dump(issuer_encoder, 'issuer_encoder.pkl')
joblib.dump(industry_encoder, 'industry_encoder.pkl')
joblib.dump(rating_encoder, 'rating_encoder.pkl')

# Prepare X and y
X = df[['Issuer Encoded', 'Industry Encoded', 'Debt to Equity', 'EBITDA Margin', 'Interest Coverage', 'Issue Size (₹Cr)', 'DefaultFlag']]
y = df['Rating Encoded']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'credit_rating_model.pkl')

print("✅ Model trained and saved successfully.")
