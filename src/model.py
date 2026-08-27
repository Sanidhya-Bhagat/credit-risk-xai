from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_STATE


def create_baseline_model() -> Pipeline:
    """Create the baseline logistic regression model."""

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    random_state=RANDOM_STATE,
                    max_iter=1000,
                ),
            ),
        ]
    )

    return model
