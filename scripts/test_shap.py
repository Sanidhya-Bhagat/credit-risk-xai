from src.explain import create_shap_explainer
from src.split import create_train_test_split
from src.xgboost_model import create_xgboost_model


X_train, X_test, y_train, y_test = create_train_test_split()

model = create_xgboost_model()
model.fit(X_train, y_train)

explainer = create_shap_explainer(model)

shap_values = explainer.shap_values(X_test.iloc[:5])

print("=== SHAP EXPLAINER TEST ===")
print(f"Input shape: {X_test.iloc[:5].shape}")
print(f"SHAP output shape: {shap_values.shape}")
print(f"Number of features: {X_test.shape[1]}")