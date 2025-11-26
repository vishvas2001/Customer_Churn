# main.py

from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import predict_churn  # reusing your existing function


app = FastAPI(
    title="Customer Churn Prediction API",
    description="FastAPI backend for predicting telecom customer churn.",
    version="1.0.0",
)


class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/")
def root():
    return {"message": "Customer Churn Prediction API is running ✅"}


@app.post("/predict")
def predict(customer: CustomerData):
    """
    Predict churn for a single customer.
    """
    customer_dict = customer.dict()
    label, proba = predict_churn(customer_dict)

    return {
        "input": customer_dict,
        "churn": bool(label),             # True = churn, False = no churn
        "churn_probability": proba,       # 0–1
        "churn_probability_percent": round(proba * 100, 2),
        "message": (
            "High risk of churn" if label == 1 else "Low risk of churn"
        ),
    }
