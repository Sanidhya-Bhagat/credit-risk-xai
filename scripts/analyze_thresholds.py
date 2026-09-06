from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.validation import create_train_validation_test_split
from src.xgboost_model import create_xgboost_model


(
    X_train,
    X_validation,
    _,
    y_train,
    y_validation,
    _,
) = create_train_validation_test_split()

model = create_xgboost_model()
model.fit(X_train, y_train)

y_proba = model.predict_proba(X_validation)[:, 1]

thresholds = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
]


print("=== XGBOOST VALIDATION THRESHOLD ANALYSIS ===")
print()

for threshold in thresholds:
    y_pred = (y_proba >= threshold).astype(int)

    precision = precision_score(
        y_validation,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_validation,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_validation,
        y_pred,
        zero_division=0,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_validation,
        y_pred,
    ).ravel()

    predicted_risky = int(y_pred.sum())

    print(f"Threshold: {threshold:.2f}")
    print(f"  Precision:        {precision:.4f}")
    print(f"  Recall:           {recall:.4f}")
    print(f"  F1:               {f1:.4f}")
    print(f"  False positives:  {fp}")
    print(f"  False negatives:  {fn}")
    print(f"  True positives:   {tp}")
    print(f"  Predicted risky:  {predicted_risky}")
    print()
