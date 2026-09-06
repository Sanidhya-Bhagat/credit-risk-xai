import pandas as pd

from src.explain import create_shap_explainer
from src.lime_explain import create_lime_explainer
from src.split import create_train_test_split
from src.xgboost_model import create_xgboost_model


X_train, X_test, y_train, y_test = create_train_test_split()

model = create_xgboost_model()
model.fit(X_train, y_train)

customer_index = X_test.index[0]
customer = X_test.iloc[0]

# SHAP explanation
shap_explainer = create_shap_explainer(model)
shap_values = shap_explainer(X_test.iloc[[0]])

shap_importance = pd.DataFrame(
    {
        "feature": X_test.columns,
        "shap_contribution": shap_values.values[0],
        "abs_shap": abs(shap_values.values[0]),
    }
)

shap_importance = shap_importance.sort_values(
    "abs_shap",
    ascending=False,
).head(10)

# LIME explanation
lime_explainer = create_lime_explainer(
    X_train,
    y_train,
)

lime_explanation = lime_explainer.explain_instance(
    customer.values,
    model.predict_proba,
    num_features=10,
    num_samples=5000,
)

lime_results = []

for feature_condition, contribution in lime_explanation.as_list():
    lime_results.append(
        {
            "feature_condition": feature_condition,
            "lime_contribution": contribution,
        }
    )

lime_results = pd.DataFrame(lime_results)

print("=== SHAP VS LIME COMPARISON ===")
print()

print(f"Customer index: {customer_index}")
print(f"Actual outcome: {y_test.loc[customer_index]}")
print()

print("SHAP — Top 10 features:")
print()

for _, row in shap_importance.iterrows():
    print(
        f"{row['feature']:<15} "
        f"{row['shap_contribution']:+.6f}"
    )

print()
print("LIME — Top 10 local conditions:")
print()

for _, row in lime_results.iterrows():
    print(
        f"{row['feature_condition']:<45} "
        f"{row['lime_contribution']:+.6f}"
    )
