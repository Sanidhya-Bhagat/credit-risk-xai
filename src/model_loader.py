from xgboost import XGBClassifier

from src.config import MODELS_DIR


MODEL_PATH = MODELS_DIR / "xgboost_credit_risk.json"


def load_xgboost_model() -> XGBClassifier:
    """Load the persisted XGBoost credit-risk model."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Run scripts.save_xgboost_model before loading the model."
        )

    model = XGBClassifier()
    model.load_model(MODEL_PATH)

    return model