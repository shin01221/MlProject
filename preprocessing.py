import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def preprocess_data(df, target_column):
    y = df[target_column]
    X = df.drop(columns=[target_column])

    # Split first to prevent data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Identify numeric and categorical columns
    numeric_cols = X_train.select_dtypes(include="number").columns.tolist()  # type: ignore[union-attr]
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()  # type: ignore[union-attr]

    # Impute and scale numeric features
    num_imputer = SimpleImputer(strategy="mean")
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(
        num_imputer.fit_transform(X_train[numeric_cols])
    )
    X_test[numeric_cols] = scaler.transform(
        num_imputer.transform(X_test[numeric_cols])
    )

    # One-hot encode categorical features (fit on train only)
    X_train = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
    X_test = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)

    # Align columns between train and test (handle unseen categories)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    return X_train, X_test, y_train, y_test
