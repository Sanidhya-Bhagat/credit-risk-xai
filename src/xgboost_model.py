from xgboost import XGBClassifier

from src.config import RANDOM_STATE


def create_xgboost_model() -> XGBClassifier:
    """Create the baseline XGBoost classifier."""

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return model
