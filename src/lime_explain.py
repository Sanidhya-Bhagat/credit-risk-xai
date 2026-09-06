from lime.lime_tabular import LimeTabularExplainer


def create_lime_explainer(X_train, y_train):
    """Create a LIME tabular explainer for the credit-risk model."""

    return LimeTabularExplainer(
        X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=["No Default", "Default"],
        mode="classification",
        random_state=42,
    )
