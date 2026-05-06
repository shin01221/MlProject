from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def evaluate_models(models, X_test, y_test, task):
    results = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        if task == "regression":
            results.append(
                {
                    "Model": name,
                    "MAE": mean_absolute_error(y_test, y_pred),
                    "MSE": mean_squared_error(y_test, y_pred),
                    "R2": r2_score(y_test, y_pred),
                }
            )
        else:
            results.append(
                {
                    "Model": name,
                    "Accuracy": accuracy_score(y_test, y_pred),
                    "Precision": precision_score(y_test, y_pred, average="weighted"),
                    "Recall": recall_score(y_test, y_pred, average="weighted"),
                    "F1": f1_score(y_test, y_pred, average="weighted"),
                }
            )
    return results
