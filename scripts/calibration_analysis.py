import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from src.split import create_train_test_split
from src.xgboost_model import create_xgboost_model


X_train, X_test, y_train, y_test = create_train_test_split()

model = create_xgboost_model()
model.fit(X_train, y_train)

y_proba = model.predict_proba(X_test)[:, 1]

prob_true, prob_pred = calibration_curve(
    y_test,
    y_proba,
    n_bins=10,
    strategy="uniform",
)

brier_score = brier_score_loss(y_test, y_proba)

print("=== CALIBRATION ANALYSIS ===")
print()

print(f"Brier score: {brier_score:.4f}")
print()

print("Calibration bins:")
print()

for index, (predicted, actual) in enumerate(
    zip(prob_pred, prob_true),
    start=1,
):
    print(
        f"Bin {index:2d} | "
        f"Mean predicted: {predicted:.4f} | "
        f"Observed default rate: {actual:.4f} | "
        f"Difference: {actual - predicted:+.4f}"
    )