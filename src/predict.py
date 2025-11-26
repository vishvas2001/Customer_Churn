# src/predict.py

from pathlib import Path
import joblib
import pandas as pd


MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_PATH = MODELS_DIR / "churn_model.pkl"


def load_model():
    """
    Load the trained churn model pipeline from disk.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. "
            "Train the model first by running train_model.py."
        )
    return joblib.load(MODEL_PATH)


def predict_churn(customer_data: dict):
    """
    Predict churn for a single customer.

    customer_data: dict with keys matching the original feature columns
    (except 'customerID' and 'Churn').

    Returns:
        predicted_label (0 or 1),
        churn_probability (float between 0 and 1)
    """
    model = load_model()

    # Convert dict -> DataFrame with one row
    input_df = pd.DataFrame([customer_data])

    proba = model.predict_proba(input_df)[0, 1]
    label = int(model.predict(input_df)[0])
    return label, float(proba)


if __name__ == "__main__":
    # Example usage:
    example_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.0,
        "TotalCharges": 400.0,
    }

    label, proba = predict_churn(example_customer)
    print(f"Predicted churn label: {label} (1 = churn, 0 = no churn)")
    print(f"Churn probability: {proba:.3f}")
