from src.lime_explain import create_lime_explainer
from src.split import create_train_test_split


X_train, X_test, y_train, y_test = create_train_test_split()

explainer = create_lime_explainer(
    X_train,
    y_train,
)

print("=== LIME EXPLAINER TEST ===")
print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")
print(f"Number of features: {len(X_train.columns)}")
print(f"Feature names: {explainer.feature_names}")
