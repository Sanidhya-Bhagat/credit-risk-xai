from src.model_loader import load_xgboost_model


model = load_xgboost_model()

print("=== MODEL LOADER TEST ===")
print()
print(f"Model loaded successfully: {type(model).__name__}")
print(f"Number of features: {model.n_features_in_}")