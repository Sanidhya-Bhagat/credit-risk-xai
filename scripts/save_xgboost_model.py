from src.config import MODELS_DIR
from src.split import create_train_test_split
from src.xgboost_model import create_xgboost_model


X_train, X_test, y_train, y_test = create_train_test_split()

model = create_xgboost_model()
model.fit(X_train, y_train)

MODELS_DIR.mkdir(parents=True, exist_ok=True)

model_path = MODELS_DIR / "xgboost_credit_risk.json"

model.save_model(model_path)

print("=== MODEL PERSISTENCE ===")
print()
print(f"Model saved to: {model_path}")
print(f"Training rows: {len(X_train)}")
print(f"Features: {X_train.shape[1]}")
print(f"Model type: {type(model).__name__}")