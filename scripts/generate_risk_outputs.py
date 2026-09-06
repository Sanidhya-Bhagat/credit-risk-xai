import pandas as pd

from src.config import OUTPUTS_DIR
from src.split import create_train_test_split
from src.xgboost_model import create_xgboost_model


X_train, X_test, y_train, y_test = create_train_test_split()

model = create_xgboost_model()
model.fit(X_train, y_train)

y_proba = model.predict_proba(X_test)[:, 1]

risk_data = pd.DataFrame(
    {
        "customer_index": X_test.index,
        "predicted_probability": y_proba,
        "actual_default": y_test.to_numpy(),
    }
)

bins = [
    0.00,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    1.00,
]

labels = [
    "0–10%",
    "10–20%",
    "20–30%",
    "30–40%",
    "40–50%",
    "50–60%",
    "60–70%",
    "70–80%",
    "80–90%",
    "90–100%",
]

risk_data["risk_band"] = pd.cut(
    risk_data["predicted_probability"],
    bins=bins,
    labels=labels,
    include_lowest=True,
    right=False,
)

risk_data["decision_at_0_30"] = (
    risk_data["predicted_probability"] >= 0.30
).astype(int)

risk_data = risk_data[
    [
        "customer_index",
        "predicted_probability",
        "risk_band",
        "decision_at_0_30",
        "actual_default",
    ]
]

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

output_path = OUTPUTS_DIR / "risk_scores.csv"
risk_data.to_csv(output_path, index=False)

print("=== RISK OUTPUT GENERATION ===")
print()
print(f"Customers scored: {len(risk_data)}")
print(f"Output file: {output_path}")
print()

print("Columns:")
for column in risk_data.columns:
    print(f"- {column}")

print()
print("First 10 risk scores:")
print(risk_data.head(10).to_string(index=False))