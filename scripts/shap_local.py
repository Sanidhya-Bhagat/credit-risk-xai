import matplotlib.pyplot as plt
import shap

from src.config import OUTPUTS_DIR
from src.explain import create_shap_explainer
from src.split import create_train_test_split
from src.xgboost_model import create_xgboost_model


X_train, X_test, y_train, y_test = create_train_test_split()

model = create_xgboost_model()
model.fit(X_train, y_train)

explainer = create_shap_explainer(model)

customer = X_test.iloc[[0]]

shap_values = explainer(customer)

customer_index = customer.index[0]

print("=== LOCAL SHAP EXPLANATION ===")
print()

print(f"Customer index: {customer_index}")
print(f"Actual outcome: {y_test.loc[customer_index]}")
print()

print("Feature contributions:")

for feature, value, shap_value in zip(
    customer.columns,
    customer.iloc[0],
    shap_values.values[0],
):
    print(
        f"{feature:<15} "
        f"Value: {value:<10} "
        f"SHAP: {shap_value:+.6f}"
    )

shap.plots.waterfall(
    shap_values[0],
    max_display=15,
    show=False,
)

plt.tight_layout()

output_path = OUTPUTS_DIR / f"shap_local_customer_{customer_index}.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")

print()
print(f"Saved to: {output_path}")

plt.close()