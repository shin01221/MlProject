from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate_models(models, X_test, y_test):
    results = []

    for name, model in models.items():
        y_pred = model.predict(X_test)

        results.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, average="weighted"),
                "Recall": recall_score(y_test, y_pred, average="weighted"),
                "F1 Score": f1_score(y_test, y_pred, average="weighted"),
            }
        )

    return results
