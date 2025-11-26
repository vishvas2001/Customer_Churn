# src/train_model.py

from pathlib import Path

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

from data_processing import load_data, train_test_split_data, get_feature_types


# Path to CSV (adjust if your path is different)
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODELS_DIR.mkdir(exist_ok=True)


def build_pipeline(cat_cols, num_cols):
    """
    Build a preprocessing + model pipeline.
    - OneHotEncoder for categorical columns
    - Pass-through for numeric columns
    - Logistic Regression as final classifier
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    model = LogisticRegression(max_iter=2000)

    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )

    return clf


def train_and_save_model():
    # 1. Load & clean data
    df = load_data(str(DATA_PATH))

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split_data(df)

    # 3. Detect feature types
    cat_cols, num_cols = get_feature_types(X_train)

    # 4. Build pipeline
    clf = build_pipeline(cat_cols, num_cols)

    # 5. Train
    clf.fit(X_train, y_train)

    # 6. Evaluate
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    print("=== Classification Report (Logistic Regression Pipeline) ===")
    print(classification_report(y_test, y_pred))

    print("ROC-AUC:", roc_auc_score(y_test, y_proba))

    # 7. Save model
    model_path = MODELS_DIR / "churn_model.pkl"
    joblib.dump(clf, model_path)
    print(f"\nModel saved to: {model_path}")


if __name__ == "__main__":
    train_and_save_model()
