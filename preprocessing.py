import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess_data(df, target_column):
    y = df[target_column]
    X = df.drop(columns=[target_column])

    # Fill missing values
    X = X.fillna(X.mean(numeric_only=True))

    # Encode features فقط
    X = pd.get_dummies(X, drop_first=True)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
