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
    X_test,
    y_train,
    y_validation,
    y_test,
) = create_train_validation_test_split()

model = create_xgboost_model()
model.fit(X_train, y_train)

# The validation set was used to select this threshold.
selected_threshold = 0.30


print("=== THRESHOLD SELECTION ===")
print(f"Selected threshold: {selected_threshold:.2f}")
print()

print("=== FINAL TEST EVALUATION ===")
print()


def evaluate_threshold(threshold):
    """Evaluate the model on the untouched test set at a given threshold."""

    y_test_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_test_proba >= threshold).astype(int)

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

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred,
    ).ravel()

    print(f"Threshold: {threshold:.2f}")
    print(f"  Precision:        {precision:.4f}")
    print(f"  Recall:           {recall:.4f}")
    print(f"  F1:               {f1:.4f}")
    print(f"  False positives:  {fp}")
    print(f"  False negatives:  {fn}")
    print(f"  True positives:   {tp}")
    print(f"  True negatives:   {tn}")
    print("  Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print()


evaluate_threshold(0.50)
evaluate_threshold(selected_threshold)
