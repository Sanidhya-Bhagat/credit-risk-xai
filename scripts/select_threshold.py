from sklearn.metrics import (
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

results = []

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

    results.append(
        {
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )


best_result = max(
    results,
    key=lambda result: result["f1"],
)

print("=== VALIDATION THRESHOLD SELECTION ===")
print()

for result in results:
    print(
        f"Threshold: {result['threshold']:.2f} | "
        f"Precision: {result['precision']:.4f} | "
        f"Recall: {result['recall']:.4f} | "
        f"F1: {result['f1']:.4f}"
    )

print()
print("=== SELECTED THRESHOLD ===")
print(f"Threshold: {best_result['threshold']:.2f}")
print(f"Precision: {best_result['precision']:.4f}")
print(f"Recall: {best_result['recall']:.4f}")
print(f"F1: {best_result['f1']:.4f}")
