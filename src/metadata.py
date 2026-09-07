from src.config import DECISION_THRESHOLD
from src.data import FEATURE_COLUMNS
from src.model_loader import MODEL_PATH
from src.risk import RISK_BANDS


MODEL_METADATA = {
    "model_type": "XGBClassifier",
    "task": "Binary classification",
    "features": FEATURE_COLUMNS,
    "feature_count": len(FEATURE_COLUMNS),
    "decision_threshold": DECISION_THRESHOLD,
    "risk_band_count": len(RISK_BANDS),
    "model_artifact": MODEL_PATH.name,
    "objective": "binary:logistic",
}
