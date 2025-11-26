# src/data_preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(csv_path: str) -> pd.DataFrame:
    """
    Load the Telco churn dataset and do basic cleaning.
    - converts TotalCharges to numeric and fills missing values
    - drops customerID (not useful for prediction)
    """
    df = pd.read_csv(csv_path)

    # Fix TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Drop ID column
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    return df


def train_test_split_data(df: pd.DataFrame):
    """
    Split the cleaned dataframe into X_train, X_test, y_train, y_test.
    """
    X = df.drop("Churn", axis=1)
    y = df["Churn"].map({"No": 0, "Yes": 1})  # encode target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def get_feature_types(X: pd.DataFrame):
    """
    Return lists of categorical and numerical column names.
    """
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()
    return cat_cols, num_cols
