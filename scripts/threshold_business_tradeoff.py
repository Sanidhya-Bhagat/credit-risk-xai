import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from src.split import create_train_test_split
from src.xgboost_model import create_xgboost_model


X_train, X_test, y_train, y_test = create_train_test_split()

model = create_xgboost_model()
model.fit(X_train, y_train)

y_proba = model.predict_proba(X_test)[:, 1]

thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.50]

print("=== THRESHOLD VS BUSINESS TRADE-OFF ===")
print()

total_defaults = int(y_test.sum())

print(f"Total customers: {len(y_test)}")
print(f"Total actual defaults: {total_defaults}")
print()

for threshold in thresholds:
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred,
    ).ravel()

    flagged = int(y_pred.sum())
    captured_defaults = tp
    captured_default_rate = (
        captured_defaults / total_defaults
        if total_defaults
        else 0
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    print(
        f"Threshold: {threshold:.2f} | "
        f"Flagged: {flagged:4d} | "
        f"TP: {tp:4d} | "
        f"FP: {fp:4d} | "
        f"FN: {fn:4d} | "
        f"Precision: {precision:.4f} | "
        f"Recall: {recall:.4f} | "
        f"F1: {f1:.4f} | "
        f"Defaults captured: {captured_default_rate:.2%}"
    )