import pandas as pd
from fastapi import FastAPI
from api.schemas import CreditRiskInput, CreditRiskPrediction
from src.config import DECISION_THRESHOLD
from src.model_loader import load_xgboost_model
from src.risk import get_risk_band


app = FastAPI(
    title="Credit Risk XAI API",
    description="API for credit default risk prediction and explainability.",
    version="1.0.0",
)

model = load_xgboost_model()


@app.get("/health")
def health_check():
    """Return the API health status."""

    return {
        "status": "healthy",
        "service": "credit-risk-xai-api",
    }


@app.post(
    "/predict",
    response_model=CreditRiskPrediction,
    summary="Predict credit default risk",
    description=(
        "Estimate the probability that a customer will default on their "
        "credit payment. The API also assigns a probability-based risk "
        "band and applies the validated 0.30 decision threshold."
    ),
    response_description="Credit default risk prediction and decision.",
)
def predict_credit_risk(customer: CreditRiskInput):
    """Predict credit default risk for a customer."""

    customer_data = pd.DataFrame(
        [customer.model_dump()]
    )

    probability = float(
        model.predict_proba(customer_data)[0, 1]
    )

    risk_band = get_risk_band(probability)

    decision = int(probability >= DECISION_THRESHOLD)

    return {
        "predicted_probability": probability,
        "risk_band": risk_band,
        "decision_at_0_30": decision,
    }
