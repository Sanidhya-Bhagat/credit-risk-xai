from src.lime_explain import create_lime_explainer
from src.split import create_train_test_split
from src.xgboost_model import create_xgboost_model


X_train, X_test, y_train, y_test = create_train_test_split()

model = create_xgboost_model()
model.fit(X_train, y_train)

explainer = create_lime_explainer(
    X_train,
    y_train,
)

customer = X_test.iloc[0].values

prediction = model.predict_proba(
    X_test.iloc[[0]]
)[0]

explanation = explainer.explain_instance(
    customer,
    model.predict_proba,
    num_features=10,
)

print("=== LIME PREDICTION TEST ===")
print()

print(f"Customer index: {X_test.index[0]}")
print(f"Actual outcome: {y_test.loc[X_test.index[0]]}")
print()

print(f"Probability of no default: {prediction[0]:.4f}")
print(f"Probability of default:    {prediction[1]:.4f}")
print()

print("LIME explanation:")
for feature, contribution in explanation.as_list():
    print(
        f"{feature:<45} "
        f"{contribution:+.6f}"
    )
