from src.evaluate import evaluate_binary_classifier
from src.model import create_baseline_model
from src.split import create_train_test_split
from src.xgboost_model import create_xgboost_model


X_train, X_test, y_train, y_test = create_train_test_split()

# Logistic Regression
lr = create_baseline_model()
lr.fit(X_train, y_train)
lr_metrics = evaluate_binary_classifier(lr, X_test, y_test)

# XGBoost
xgb = create_xgboost_model()
xgb.fit(X_train, y_train)
xgb_metrics = evaluate_binary_classifier(xgb, X_test, y_test)


def print_metrics(name, metrics):
    print(f"=== {name} ===")
    print(f"ROC-AUC:    {metrics['roc_auc']:.4f}")
    print(f"PR-AUC:     {metrics['pr_auc']:.4f}")
    print(f"Accuracy:   {metrics['accuracy']:.4f}")
    print(f"Precision:  {metrics['precision']:.4f}")
    print(f"Recall:     {metrics['recall']:.4f}")
    print(f"F1:         {metrics['f1']:.4f}")
    print("Confusion Matrix:")
    print(metrics["confusion_matrix"])
    print()


print_metrics("LOGISTIC REGRESSION", lr_metrics)
print_metrics("XGBOOST", xgb_metrics)
