import numpy as np

from src.split import create_train_test_split
from src.xgboost_model import create_xgboost_model


X_train, X_test, y_train, y_test = create_train_test_split()

model = create_xgboost_model()
model.fit(X_train, y_train)

y_proba = model.predict_proba(X_test)[:, 1]

print("=== RISK PROBABILITY DISTRIBUTION ===")
print()

print(f"Minimum probability: {y_proba.min():.4f}")
print(f"25th percentile:     {np.percentile(y_proba, 25):.4f}")
print(f"Median probability:  {np.median(y_proba):.4f}")
print(f"75th percentile:     {np.percentile(y_proba, 75):.4f}")
print(f"Maximum probability: {y_proba.max():.4f}")
print()

print("Probability ranges:")

ranges = [
    (0.00, 0.10),
    (0.10, 0.20),
    (0.20, 0.30),
    (0.30, 0.40),
    (0.40, 0.50),
    (0.50, 0.60),
    (0.60, 0.70),
    (0.70, 0.80),
    (0.80, 0.90),
    (0.90, 1.00),
]

for lower, upper in ranges:
    count = np.sum(
        (y_proba >= lower) & (y_proba < upper)
    )

    print(
        f"{lower:.2f} - {upper:.2f}: "
        f"{count:4d}"
    )