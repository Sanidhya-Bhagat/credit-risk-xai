import pandas as pd

from src.split import create_train_test_split
from src.xgboost_model import create_xgboost_model


X_train, X_test, y_train, y_test = create_train_test_split()

model = create_xgboost_model()
model.fit(X_train, y_train)

y_proba = model.predict_proba(X_test)[:, 1]

risk_data = pd.DataFrame(
    {
        "probability": y_proba,
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
    risk_data["probability"],
    bins=bins,
    labels=labels,
    include_lowest=True,
    right=False,
)

distribution = (
    risk_data
    .groupby("risk_band", observed=False)
    .size()
    .reset_index(name="customers")
)

total_customers = len(risk_data)

distribution["portfolio_percentage"] = (
    distribution["customers"] / total_customers * 100
)

distribution["cumulative_percentage"] = (
    distribution["portfolio_percentage"].cumsum()
)

print("=== CUSTOMER DISTRIBUTION BY RISK BAND ===")
print()
print(f"Total customers: {total_customers}")
print()

for _, row in distribution.iterrows():
    print(
        f"{row['risk_band']:<10} "
        f"Customers: {int(row['customers']):4d} | "
        f"Portfolio: {row['portfolio_percentage']:6.2f}% | "
        f"Cumulative: {row['cumulative_percentage']:6.2f}%"
    )