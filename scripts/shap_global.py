import pandas as pd

from src.explain import create_shap_explainer
from src.split import create_train_test_split
from src.xgboost_model import create_xgboost_model


X_train, X_test, y_train, y_test = create_train_test_split()

model = create_xgboost_model()
model.fit(X_train, y_train)

explainer = create_shap_explainer(model)

shap_values = explainer.shap_values(X_test)

feature_importance = pd.DataFrame(
    {
        "feature": X_test.columns,
        "mean_abs_shap": abs(shap_values).mean(axis=0),
    }
)

feature_importance = feature_importance.sort_values(
    "mean_abs_shap",
    ascending=False,
)

print("=== GLOBAL SHAP FEATURE IMPORTANCE ===")
print()

for _, row in feature_importance.iterrows():
    print(
        f"{row['feature']:<15} "
        f"{row['mean_abs_shap']:.6f}"
    )