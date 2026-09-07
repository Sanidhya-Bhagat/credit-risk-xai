# Credit Risk XAI

An explainable machine-learning project for predicting credit-card default risk using the UCI Default of Credit Card Clients dataset.

The project combines:

- Credit-risk classification
- Logistic Regression baseline modeling
- XGBoost benchmarking
- Probability-threshold optimization
- SHAP explainability
- LIME local explanations
- Risk-band analysis
- Calibration analysis
- Business-oriented threshold analysis
- Reusable risk-scoring outputs
- FastAPI model serving
- Automated API validation
- Reproducible data and model pipelines

---

## 1. Project Overview

Credit-risk models are commonly used to estimate whether a customer is likely to default on a credit obligation.

However, predictive performance alone is not sufficient for a practical credit-risk system. A useful system should also provide:

1. A probability of default
2. A clear risk classification
3. A configurable decision threshold
4. Explanations for model predictions
5. Validation of the model's probability behavior
6. Outputs that can be consumed by downstream applications
7. An API through which predictions can be requested

This project was developed to demonstrate that complete workflow.

The final system uses an **XGBoost classifier** as the primary predictive model and supplements its predictions with **SHAP** and **LIME** explanations.

---

## 2. Key Results

The project evaluates both Logistic Regression and XGBoost models.

### XGBoost

Using the held-out test set:

| Metric | XGBoost |
|---|---:|
| ROC-AUC | 0.7775 |
| PR-AUC | 0.5552 |
| Accuracy | 0.8178 |
| Precision | 0.6594 |
| Recall | 0.3647 |
| F1 Score | 0.4697 |

At the original probability threshold of `0.50`, the model was relatively conservative in identifying defaults.

A validation-based threshold analysis selected:

```text
Decision threshold = 0.30