from pydantic import BaseModel, Field


class CreditRiskInput(BaseModel, extra="forbid"):
    """Input features required for credit-risk prediction."""

    LIMIT_BAL: int = Field(
        description="Amount of given credit in NT dollars."
    )
    SEX: int = Field(
        description="Gender code from the UCI dataset."
    )
    EDUCATION: int = Field(
        description="Education-level code from the UCI dataset."
    )
    MARRIAGE: int = Field(
        description="Marital-status code from the UCI dataset."
    )
    AGE: int = Field(
        description="Age of the customer in years."
    )

    PAY_0: int = Field(
        description="Repayment status in September."
    )
    PAY_2: int = Field(
        description="Repayment status in August."
    )
    PAY_3: int = Field(
        description="Repayment status in July."
    )
    PAY_4: int = Field(
        description="Repayment status in June."
    )
    PAY_5: int = Field(
        description="Repayment status in May."
    )
    PAY_6: int = Field(
        description="Repayment status in April."
    )

    BILL_AMT1: int = Field(
        description="Bill statement amount in September."
    )
    BILL_AMT2: int = Field(
        description="Bill statement amount in August."
    )
    BILL_AMT3: int = Field(
        description="Bill statement amount in July."
    )
    BILL_AMT4: int = Field(
        description="Bill statement amount in June."
    )
    BILL_AMT5: int = Field(
        description="Bill statement amount in May."
    )
    BILL_AMT6: int = Field(
        description="Bill statement amount in April."
    )

    PAY_AMT1: int = Field(
        description="Previous payment amount in September."
    )
    PAY_AMT2: int = Field(
        description="Previous payment amount in August."
    )
    PAY_AMT3: int = Field(
        description="Previous payment amount in July."
    )
    PAY_AMT4: int = Field(
        description="Previous payment amount in June."
    )
    PAY_AMT5: int = Field(
        description="Previous payment amount in May."
    )
    PAY_AMT6: int = Field(
        description="Previous payment amount in April."
    )


class CreditRiskPrediction(BaseModel):
    """Prediction returned by the credit-risk API."""

    predicted_probability: float = Field(
        description="Estimated probability of default, between 0 and 1."
    )
    risk_band: str = Field(
        description="Probability-based risk band assigned to the prediction."
    )
    decision_at_0_30: int = Field(
        description="Binary decision using the validated 0.30 probability threshold: 1 = flagged, 0 = not flagged."
    )
