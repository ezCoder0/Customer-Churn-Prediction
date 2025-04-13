import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('churn_model.pkl')

st.title("Telco Customer Churn Prediction")

# Collect user input
gender = st.selectbox("Gender", ['Male', 'Female'])
senior_citizen = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Has Partner?", ['Yes', 'No'])
dependents = st.selectbox("Has Dependents?", ['Yes', 'No'])
tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0)
total_charges = st.number_input("Total Charges", min_value=0.0)
contract = st.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'])
paperless_billing = st.selectbox("Paperless Billing", ['Yes', 'No'])
payment_method = st.selectbox("Payment Method", [
    'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
])

# One-hot or binary encoding (just a placeholder — adjust based on your preprocessing)
gender_male = 1 if gender == 'Male' else 0
partner_yes = 1 if partner == 'Yes' else 0
dependents_yes = 1 if dependents == 'Yes' else 0
contract_two_year = 1 if contract == 'Two year' else 0
paperless_yes = 1 if paperless_billing == 'Yes' else 0
payment_electronic = 1 if payment_method == 'Electronic check' else 0

# Construct the feature vector
features = np.array([[total_charges, monthly_charges, tenure,
                      payment_electronic, 1,  # assuming Fiber optic
                      contract_two_year,
                      gender_male,
                      paperless_yes, 1  # assuming has online security
                      ]])

if st.button("Predict"):
    prediction = model.predict(features)
    result = "Churn" if prediction[0] == 1 else "No Churn"
    st.success(f"Prediction: {result}")
