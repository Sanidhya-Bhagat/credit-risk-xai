import shap


def create_shap_explainer(model):
    """Create a SHAP TreeExplainer for the XGBoost model."""

    return shap.TreeExplainer(model)