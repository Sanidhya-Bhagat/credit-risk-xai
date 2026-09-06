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

shap_values = explainer.shap_values(X_test)

shap.summary_plot(
    shap_values,
    X_test,
    show=False,
)

plt.tight_layout()

output_path = OUTPUTS_DIR / "shap_summary.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")

print("=== SHAP SUMMARY PLOT ===")
print(f"Saved to: {output_path}")

plt.close()