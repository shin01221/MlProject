import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess_data(df, target_column):
    y = df[target_column]
    X = df.drop(columns=[target_column])

    task = "regression" if y.dtype == "float64" and y.nunique() > 2 else "classification"

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    num = X_train.select_dtypes(include="number").columns.tolist()  # type: ignore[union-attr]
    cat = X_train.select_dtypes(exclude="number").columns.tolist()  # type: ignore[union-attr]

    X_train = pd.get_dummies(X_train, columns=cat, drop_first=True)
    X_test = pd.get_dummies(X_test, columns=cat, drop_first=True)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    scaler = StandardScaler()
    X_train[num] = scaler.fit_transform(X_train[num])
    X_test[num] = scaler.transform(X_test[num])

    return X_train, X_test, y_train, y_test, task
