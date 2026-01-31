import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)

from sklearn.model_selection import train_test_split


# ==========================================================
# 1. TRAIN/TEST + PREPROCESSOR
# ==========================================================
def prepare_training_data(df: pd.DataFrame):
    """
    Splits into train/test and builds preprocessing pipeline:
      - StandardScaler for numeric
      - OneHotEncoder for categorical
    """

    X = df.drop(columns=["injury_label"])
    y = df["injury_label"]

    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test, preprocessor, numeric_cols, categorical_cols


# ==========================================================
# 2. EVALUATION
# ==========================================================
def evaluate_classifier(model, X_test, y_test, name="Model"):

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        try:
            metrics["roc_auc"] = roc_auc_score(y_test, y_prob)
        except:
            metrics["roc_auc"] = None

    print(f"\n===== {name} Report =====")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return metrics


# ==========================================================
# 3. Dummy Classifier
# ==========================================================
def train_dummy_classifier(X_train, y_train, preprocessor):
    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("clf", DummyClassifier(strategy="most_frequent"))
    ])

    model.fit(X_train, y_train)
    return model


# ==========================================================
# 4. Logistic Regression
# ==========================================================
def train_logistic_regression(X_train, y_train, preprocessor):
    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("clf", LogisticRegression(max_iter=500))
    ])

    model.fit(X_train, y_train)
    return model


# ==========================================================
# 5. Random Forest
# ==========================================================
def train_random_forest(X_train, y_train, preprocessor):
    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced"
        ))
    ])

    model.fit(X_train, y_train)
    return model