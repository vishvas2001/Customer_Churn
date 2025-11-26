# app.py

import streamlit as st
from src.predict import predict_churn  # uses your saved model


st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered",
)

st.title("📊 Telecom Customer Churn Predictor")
st.write(
    "Enter customer details on the left and get the churn probability. "
    "This app uses a Logistic Regression model trained on the Telco Customer Churn dataset."
)


# ---- SIDEBAR INPUT FORM ----
st.sidebar.header("Customer Details")

gender = st.sidebar.selectbox("Gender", ["Female", "Male"])

senior = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
senior_citizen = 1 if senior == "Yes" else 0

partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])

tenure = st.sidebar.number_input(
    "Tenure (months with company)",
    min_value=0,
    max_value=72,
    value=5,
    step=1,
)

phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.sidebar.selectbox(
    "Multiple Lines",
    ["No", "Yes", "No phone service"],
)

internet_service = st.sidebar.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"],
)

online_security = st.sidebar.selectbox(
    "Online Security",
    ["No", "Yes", "No internet service"],
)

online_backup = st.sidebar.selectbox(
    "Online Backup",
    ["No", "Yes", "No internet service"],
)

device_protection = st.sidebar.selectbox(
    "Device Protection",
    ["No", "Yes", "No internet service"],
)

tech_support = st.sidebar.selectbox(
    "Tech Support",
    ["No", "Yes", "No internet service"],
)

streaming_tv = st.sidebar.selectbox(
    "Streaming TV",
    ["No", "Yes", "No internet service"],
)

streaming_movies = st.sidebar.selectbox(
    "Streaming Movies",
    ["No", "Yes", "No internet service"],
)

contract = st.sidebar.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"],
)

paperless_billing = st.sidebar.selectbox(
    "Paperless Billing",
    ["Yes", "No"],
)

payment_method = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
)

monthly_charges = st.sidebar.number_input(
    "Monthly Charges",
    min_value=0.0,
    max_value=200.0,
    value=70.0,
    step=1.0,
)

total_charges = st.sidebar.number_input(
    "Total Charges",
    min_value=0.0,
    max_value=10000.0,
    value=400.0,
    step=10.0,
)


# ---- PREDICTION BUTTON ----
if st.sidebar.button("Predict Churn"):
    customer_data = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    label, proba = predict_churn(customer_data)

    st.subheader("Prediction Result")

    if label == 1:
        st.error(f"⚠ High chance of churn! Probability: **{proba:.2%}**")
    else:
        st.success(f"✅ Low chance of churn. Probability: **{proba:.2%}**")

    # Risk explanation
    if proba >= 0.7:
        st.write(
            "This customer is at **very high risk** of leaving. "
            "Consider offering discounts, better support, or a longer-term contract."
        )
    elif proba >= 0.4:
        st.write(
            "This customer is at **moderate risk**. "
            "Monitor their usage and satisfaction closely."
        )
    else:
        st.write(
            "This customer is at **low risk** of churn based on current data."
        )
else:
    st.info("Fill the details in the sidebar and click **Predict Churn**.")
