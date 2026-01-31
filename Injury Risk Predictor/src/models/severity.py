"""
Severity Regression Models for Injury Duration Prediction.

This module predicts how long an injury will last (severity_days).

Key Features:
- Temporal validation to prevent data leakage
- Log-transformed target for better handling of skewed distributions
- CatBoost hyperparameter tuning with Optuna
- Stacking ensemble combining LightGBM, XGBoost, and CatBoost
"""

import logging
import pandas as pd
import numpy as np
import optuna

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def root_mean_squared_error(y_true, y_pred):
    """RMSE calculation (for sklearn versions without it)."""
    return np.sqrt(mean_squared_error(y_true, y_pred))

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor, Pool

import warnings
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = logging.getLogger(__name__)


# ============================================================================
# TEMPORAL VALIDATION FOR REGRESSION
# ============================================================================
#
# Why temporal validation matters for severity prediction:
#   - Random splits leak future injury patterns into training data
#   - Model learns temporal correlations that won't exist at inference time
#   - Real-world MAE/RMSE will be significantly worse than random split metrics
#   - Example: Training on 2023 injuries to predict 2022 is cheating!
#
# ============================================================================


def get_temporal_severity_splits(df, date_col="date_of_injury", train_ratio=0.6, val_ratio=0.2):
    """
    Temporal split for severity regression - train on past, predict future.

    Args:
        df: DataFrame with severity_days target and date column
        date_col: Name of the date column (date_of_injury, injury_datetime, etc.)
        train_ratio: Fraction for training (default 0.6)
        val_ratio: Fraction for validation (default 0.2)

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test (all with log-transformed target)
    """
    df = df.copy()

    # Find the date column
    possible_date_cols = [date_col, "date_of_injury", "injury_datetime", "Date of Injury"]
    actual_date_col = None
    for col in possible_date_cols:
        if col in df.columns:
            actual_date_col = col
            break

    if actual_date_col is None:
        logger.warning("No date column found. Falling back to random split.")
        return get_severity_splits(df)

    df[actual_date_col] = pd.to_datetime(df[actual_date_col])

    # Log-transform target
    df["severity_log"] = np.log1p(df["severity_days"])

    # Sort by date
    df_sorted = df.sort_values(actual_date_col).reset_index(drop=True)
    n = len(df_sorted)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df_sorted.iloc[:train_end]
    val_df = df_sorted.iloc[train_end:val_end]
    test_df = df_sorted.iloc[val_end:]

    logger.info(f"Temporal severity split:")
    logger.info(f"  Train: {len(train_df)} samples, dates: {train_df[actual_date_col].min().date()} to {train_df[actual_date_col].max().date()}")
    logger.info(f"  Val:   {len(val_df)} samples, dates: {val_df[actual_date_col].min().date()} to {val_df[actual_date_col].max().date()}")
    logger.info(f"  Test:  {len(test_df)} samples, dates: {test_df[actual_date_col].min().date()} to {test_df[actual_date_col].max().date()}")

    # Drop target and date columns
    drop_cols = ["severity_days", "severity_log", actual_date_col]
    # Also drop other date columns that might cause issues
    date_like_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
    drop_cols.extend([c for c in date_like_cols if c not in drop_cols])

    X_train = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
    X_val = val_df.drop(columns=[c for c in drop_cols if c in val_df.columns])
    X_test = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])

    y_train = train_df["severity_log"]
    y_val = val_df["severity_log"]
    y_test = test_df["severity_log"]

    return X_train, X_val, X_test, y_train, y_val, y_test


# ================================================================
# DATA SPLITS & PREPROCESSOR (Random - use for quick experiments only)
# ================================================================

def get_severity_splits(df: pd.DataFrame):

    df = df.copy()
    df["severity_log"] = np.log1p(df["severity_days"])  

    X = df.drop(columns=["severity_days", "severity_log"])
    y = df["severity_log"]                                 # Train on log target

    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42
    )

    return X_train, X_test, y_train, y_test, preprocessor, numeric_cols, categorical_cols


# ================================================================
# MODELS (REGRESSION)
# ================================================================

def train_lightgbm_severity(X_train, y_train, preprocessor):
    model = Pipeline(steps=[
        ("prep", preprocessor),
        ("reg", LGBMRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=-1,
            random_state=42
        ))
    ])
    model.fit(X_train, y_train)
    return model


def train_xgboost_severity(X_train, y_train, preprocessor):
    model = Pipeline(steps=[
        ("prep", preprocessor),
        ("reg", XGBRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            eval_metric="rmse"
        ))
    ])
    model.fit(X_train, y_train)
    return model

def _prepare_catboost_data(X):
    """Prepare data for CatBoost - handle datetime and categorical columns."""
    X = X.copy()

    # Handle datetime columns
    datetime_cols = X.select_dtypes(include=["datetime64[ns]", "datetime64"]).columns
    for col in datetime_cols:
        X[col] = X[col].view("int64")
        X[col] = X[col].fillna(0)

    # Handle categorical columns
    cat_cols = [col for col in X.columns if X[col].dtype == "object"]
    for col in cat_cols:
        X[col] = X[col].astype(str).fillna("Unknown")

    cat_features = [X.columns.get_loc(col) for col in cat_cols]
    return X, cat_features


def _catboost_regression_objective(trial, X_train, y_train, X_val, y_val, cat_features):
    """Optuna objective for CatBoost regression hyperparameter tuning."""
    params = {
        "iterations": trial.suggest_int("iterations", 300, 1500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "random_seed": 42,
        "verbose": False
    }

    model = CatBoostRegressor(**params)
    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    val_pool = Pool(X_val, y_val, cat_features=cat_features)

    model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50, verbose=False)

    preds_log = model.predict(X_val)
    preds = np.expm1(preds_log)
    y_true = np.expm1(y_val)

    # Minimize MAE (negative because Optuna maximizes by default with direction="minimize")
    mae = mean_absolute_error(y_true, preds)
    return mae


def tune_catboost_severity(X_train, y_train, X_val, y_val, n_trials=40):
    """
    Tune CatBoost hyperparameters for severity regression using Optuna.

    Args:
        X_train, y_train: Training data (y should be log-transformed)
        X_val, y_val: Validation data for early stopping
        n_trials: Number of Optuna trials

    Returns:
        dict: Best hyperparameters found
    """
    logger.info(f"Starting CatBoost severity tuning with {n_trials} trials")

    X_train_cb, cat_features = _prepare_catboost_data(X_train)
    X_val_cb, _ = _prepare_catboost_data(X_val)

    study = optuna.create_study(direction="minimize")  # Minimize MAE
    study.optimize(
        lambda trial: _catboost_regression_objective(
            trial, X_train_cb, y_train, X_val_cb, y_val, cat_features
        ),
        n_trials=n_trials
    )

    logger.info(f"CatBoost tuning completed - Best MAE: {study.best_value:.3f} days")
    print(f"\n🎯 Best CatBoost Regression Params:")
    print(study.best_params)
    print(f"Best MAE = {study.best_value:.3f} days")

    return study.best_params


def run_full_catboost_severity_tuning(X_train, y_train, X_test, y_test, n_trials=40):
    """
    Full CatBoost regression tuning pipeline with temporal validation.

    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        n_trials: Number of Optuna trials

    Returns:
        model: Trained CatBoost model
        preds: Predictions on test set (back-transformed to days)
        best_params: Best hyperparameters found
        metrics: Dictionary with MAE, RMSE, R²
    """
    # Split training into train/val for tuning
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    # Tune hyperparameters
    best_params = tune_catboost_severity(X_tr, y_tr, X_val, y_val, n_trials=n_trials)

    # Train final model with best params on full training data
    X_train_cb, cat_features = _prepare_catboost_data(X_train)
    X_test_cb, _ = _prepare_catboost_data(X_test)

    final_params = best_params.copy()
    final_params.update({
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "random_seed": 42,
        "verbose": 200
    })

    model = CatBoostRegressor(**final_params)
    train_pool = Pool(X_train_cb, y_train, cat_features=cat_features)
    test_pool = Pool(X_test_cb, y_test, cat_features=cat_features)

    model.fit(train_pool, eval_set=test_pool)

    # Evaluate
    preds_log = model.predict(X_test_cb)
    preds = np.expm1(preds_log)
    y_true = np.expm1(y_test)

    metrics = {
        "mae": mean_absolute_error(y_true, preds),
        "rmse": root_mean_squared_error(y_true, preds),
        "r2": r2_score(y_true, preds)
    }

    print(f"\n📌 Tuned CatBoost Severity Performance:")
    print(f"MAE:  {metrics['mae']:.3f} days")
    print(f"RMSE: {metrics['rmse']:.3f} days")
    print(f"R²:   {metrics['r2']:.3f}")

    return model, preds, best_params, metrics


def train_catboost_severity(X_train, y_train, X_test, y_test):
    """
    Train CatBoost regressor with default parameters.
    For tuned version, use run_full_catboost_severity_tuning().
    """
    X_train_cb, cat_features = _prepare_catboost_data(X_train)
    X_test_cb, _ = _prepare_catboost_data(X_test)

    train_pool = Pool(X_train_cb, y_train, cat_features=cat_features)
    test_pool = Pool(X_test_cb, y_test, cat_features=cat_features)

    model = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.03,
        depth=8,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=42,
        verbose=200
    )

    model.fit(train_pool, eval_set=test_pool)

    preds_log = model.predict(X_test_cb)
    preds = np.expm1(preds_log)
    y_true = np.expm1(y_test)

    mae = mean_absolute_error(y_true, preds)
    rmse = root_mean_squared_error(y_true, preds)
    r2 = r2_score(y_true, preds)

    print("\n📌 CatBoost Severity — Back-transformed Performance")
    print(f"MAE:  {mae:.3f} days")
    print(f"RMSE: {rmse:.3f} days")
    print(f"R²:   {r2:.3f}")

    return model, preds


# ================================================================
# EVALUATION
# ================================================================

def evaluate_severity(model, X_test, y_test):
    # Model predicts log(severity_days)
    preds_log = model.predict(X_test)

    # Convert back to actual days
    preds = np.expm1(preds_log)
    y_true = np.expm1(y_test)

    mae  = mean_absolute_error(y_true, preds)
    rmse = root_mean_squared_error(y_true, preds)
    r2   = r2_score(y_true, preds)

    print("\n📌 Severity Regression Performance (Back-Transformed):")
    print(f"MAE:  {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R²:   {r2:.3f}")

    return {
        "preds": preds,               # actual days
        "preds_log": preds_log,       # log values for ensemble
        "metrics": {"mae": mae, "rmse": rmse, "r2": r2}
    }


# ============================================================================
# SEVERITY CLASSIFICATION (More Practical Alternative)
# ============================================================================
#
# Why classification may be better than exact regression:
#   1. Exact injury duration is inherently unpredictable (even doctors struggle)
#   2. Practical decisions need categories: "back next week" vs "out for season"
#   3. Classification is more robust to outliers and noise
#   4. Easier to explain to stakeholders
#
# 3-class bins:
#   - Short: 0-21 days (back within 3 weeks)
#   - Medium: 21-60 days (1-2 months)
#   - Long: 60+ days (season-impacting)
#
# ============================================================================

SEVERITY_BINS = [0, 21, 60, float('inf')]
SEVERITY_LABELS = ['short', 'medium', 'long']


def create_severity_bins(severity_days, bins=None, labels=None):
    """
    Convert continuous severity_days to categorical bins.

    Args:
        severity_days: Series or array of severity days
        bins: Custom bin edges (default: [0, 21, 60, inf])
        labels: Custom bin labels (default: short/medium/long)

    Returns:
        Series with categorical severity labels
    """
    if bins is None:
        bins = SEVERITY_BINS
    if labels is None:
        labels = SEVERITY_LABELS

    return pd.cut(severity_days, bins=bins, labels=labels, include_lowest=True)


def get_severity_classification_splits(df, date_col="date_of_injury", train_ratio=0.6, val_ratio=0.2):
    """
    Temporal split for severity CLASSIFICATION (binned).

    This creates a multi-class classification problem instead of regression,
    which is often more practical and achievable.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test (y is categorical labels)
    """
    df = df.copy()

    # Find date column
    possible_date_cols = [date_col, "date_of_injury", "injury_datetime", "Date of Injury"]
    actual_date_col = None
    for col in possible_date_cols:
        if col in df.columns:
            actual_date_col = col
            break

    if actual_date_col is None:
        raise ValueError("No date column found for temporal split")

    df[actual_date_col] = pd.to_datetime(df[actual_date_col])

    # Create severity bins
    df["severity_class"] = create_severity_bins(df["severity_days"])

    # Sort by date
    df_sorted = df.sort_values(actual_date_col).reset_index(drop=True)
    n = len(df_sorted)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df_sorted.iloc[:train_end]
    val_df = df_sorted.iloc[train_end:val_end]
    test_df = df_sorted.iloc[val_end:]

    # Log class distribution
    print(f"\n📊 Severity Classification Distribution:")
    print(f"  Train: {dict(train_df['severity_class'].value_counts())}")
    print(f"  Test:  {dict(test_df['severity_class'].value_counts())}")

    # Drop target and date columns
    drop_cols = ["severity_days", "severity_class", actual_date_col]
    date_like_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
    drop_cols.extend([c for c in date_like_cols if c not in drop_cols])

    X_train = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
    X_val = val_df.drop(columns=[c for c in drop_cols if c in val_df.columns])
    X_test = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])

    y_train = train_df["severity_class"]
    y_val = val_df["severity_class"]
    y_test = test_df["severity_class"]

    return X_train, X_val, X_test, y_train, y_val, y_test


def _sanitize_column_names(df):
    """Sanitize column names for LightGBM (no special JSON characters)."""
    import re
    df = df.copy()
    df.columns = [re.sub(r'[\[\]{}:,"]', '_', str(c)) for c in df.columns]
    return df


def train_severity_classifier(X_train, y_train, X_val, y_val, model_type="lightgbm"):
    """
    Train a multi-class classifier for severity bins.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data for early stopping
        model_type: "lightgbm", "xgboost", or "catboost"

    Returns:
        Trained classifier model
    """
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)

    # Sanitize column names (LightGBM doesn't allow special JSON chars)
    X_train_enc = _sanitize_column_names(X_train)
    X_val_enc = _sanitize_column_names(X_val)

    # Handle categoricals
    for col in X_train_enc.select_dtypes(include=['object', 'category']).columns:
        X_train_enc[col] = X_train_enc[col].astype('category').cat.codes
        X_val_enc[col] = X_val_enc[col].astype('category').cat.codes

    if model_type == "lightgbm":
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_class=3,
            objective='multiclass',
            class_weight='balanced',  # Handle imbalanced classes
            random_state=42,
            verbose=-1
        )
        model.fit(
            X_train_enc, y_train_enc,
            eval_set=[(X_val_enc, y_val_enc)],
            callbacks=[lambda env: None]  # Suppress output
        )

    elif model_type == "catboost":
        from catboost import CatBoostClassifier
        X_train_cb, cat_features = _prepare_catboost_data(X_train)
        X_val_cb, _ = _prepare_catboost_data(X_val)

        model = CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            loss_function='MultiClass',
            random_seed=42,
            verbose=0
        )
        model.fit(X_train_cb, y_train_enc, eval_set=(X_val_cb, y_val_enc), cat_features=cat_features)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Store label encoder for later use
    model._label_encoder = le
    model._classes = le.classes_

    return model


def evaluate_severity_classifier(model, X_test, y_test):
    """
    Evaluate severity classifier with detailed metrics.

    Handles both LightGBM and CatBoost models automatically.
    """
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    # Detect model type and preprocess accordingly
    model_class = type(model).__name__

    if "CatBoost" in model_class:
        # CatBoost: use _prepare_catboost_data (keeps strings as strings)
        X_test_enc, _ = _prepare_catboost_data(X_test)
    else:
        # LightGBM/XGBoost: sanitize column names and encode categoricals
        X_test_enc = _sanitize_column_names(X_test)
        for col in X_test_enc.select_dtypes(include=['object', 'category']).columns:
            X_test_enc[col] = X_test_enc[col].astype('category').cat.codes

    # Get predictions
    y_pred_enc = model.predict(X_test_enc)

    # Transform back to labels
    le = model._label_encoder
    y_test_enc = le.transform(y_test)
    y_pred = le.inverse_transform(y_pred_enc)

    # Metrics
    accuracy = accuracy_score(y_test_enc, y_pred_enc)

    print("\n📊 Severity Classification Results:")
    print(f"Accuracy: {accuracy:.1%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=SEVERITY_LABELS))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test_enc, y_pred_enc)
    cm_df = pd.DataFrame(cm, index=SEVERITY_LABELS, columns=SEVERITY_LABELS)
    print(cm_df)

    # Adjacent accuracy (within one category)
    adjacent = np.sum(np.abs(y_test_enc - y_pred_enc) <= 1)
    adjacent_acc = adjacent / len(y_test_enc)
    print(f"\nAdjacent Accuracy (within 1 category): {adjacent_acc:.1%}")

    return {
        "accuracy": accuracy,
        "adjacent_accuracy": adjacent_acc,
        "predictions": y_pred,
        "confusion_matrix": cm_df
    }


# ============================================================================
# OUTLIER FILTERING FOR REGRESSION
# ============================================================================

def filter_severity_outliers(df, max_days=120, min_days=1):
    """
    Remove extreme outliers from severity data.

    Very long injuries (>120 days) are often:
    - Season-ending (ACL tears, etc.) - hard to predict precisely
    - Complicated by re-injuries or setbacks
    - Dependent on factors not in the data (surgery quality, rehab compliance)

    Args:
        df: DataFrame with severity_days column
        max_days: Maximum days to include (default 120 = ~4 months)
        min_days: Minimum days to include (default 1)

    Returns:
        Filtered DataFrame
    """
    original_len = len(df)
    df_filtered = df[(df["severity_days"] >= min_days) & (df["severity_days"] <= max_days)]
    removed = original_len - len(df_filtered)

    print(f"Outlier filtering: {removed} rows removed ({removed/original_len:.1%})")
    print(f"  Kept injuries between {min_days}-{max_days} days")
    print(f"  New range: {df_filtered['severity_days'].min():.0f} - {df_filtered['severity_days'].max():.0f} days")

    return df_filtered


def get_filtered_temporal_severity_splits(df, max_days=120, min_days=1, **kwargs):
    """
    Convenience function: filter outliers then create temporal splits.
    """
    df_filtered = filter_severity_outliers(df, max_days=max_days, min_days=min_days)
    return get_temporal_severity_splits(df_filtered, **kwargs)


# ============================================================================
# HIGH-LEVEL TRAINING FUNCTIONS (for notebook usage)
# ============================================================================

def train_severity_regressors(X_train, X_test, y_train, y_test):
    """
    Train LightGBM and XGBoost severity regressors with proper preprocessing.

    This is a convenience function that handles:
    - Column name sanitization for LightGBM
    - Categorical encoding
    - Model training
    - Prediction and evaluation

    Args:
        X_train, X_test: Feature DataFrames
        y_train, y_test: Target Series (log-transformed severity)

    Returns:
        dict with:
            - lgb_model: Trained LightGBM model
            - xgb_model: Trained XGBoost model
            - lgb_preds: LightGBM predictions (actual days)
            - xgb_preds: XGBoost predictions (actual days)
            - X_train_enc: Encoded training features
            - X_test_enc: Encoded test features
    """
    import re

    # Sanitize column names
    def sanitize_col(name):
        return re.sub(r'[\[\]{}:,"]', '_', str(name))

    X_train_enc = X_train.copy()
    X_test_enc = X_test.copy()

    X_train_enc.columns = [sanitize_col(c) for c in X_train_enc.columns]
    X_test_enc.columns = [sanitize_col(c) for c in X_test_enc.columns]

    # Encode categoricals
    for col in X_train_enc.select_dtypes(include=['object']).columns:
        X_train_enc[col] = X_train_enc[col].astype('category').cat.codes
        X_test_enc[col] = X_test_enc[col].astype('category').cat.codes

    # Train LightGBM
    lgb_model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=-1,
        random_state=42,
        verbose=-1
    )
    lgb_model.fit(X_train_enc, y_train)

    lgb_preds_log = lgb_model.predict(X_test_enc)
    lgb_preds = np.expm1(lgb_preds_log)
    y_true = np.expm1(y_test)

    print("LightGBM Severity (Temporal Validation):")
    print(f"  MAE:  {mean_absolute_error(y_true, lgb_preds):.1f} days")
    print(f"  R²:   {r2_score(y_true, lgb_preds):.3f}")

    # Train XGBoost
    xgb_model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=6,
        random_state=42,
        verbosity=0
    )
    xgb_model.fit(X_train_enc, y_train)

    xgb_preds_log = xgb_model.predict(X_test_enc)
    xgb_preds = np.expm1(xgb_preds_log)

    print("\nXGBoost Severity (Temporal Validation):")
    print(f"  MAE:  {mean_absolute_error(y_true, xgb_preds):.1f} days")
    print(f"  R²:   {r2_score(y_true, xgb_preds):.3f}")

    return {
        "lgb_model": lgb_model,
        "xgb_model": xgb_model,
        "lgb_preds": lgb_preds,
        "xgb_preds": xgb_preds,
        "X_train_enc": X_train_enc,
        "X_test_enc": X_test_enc
    }


def compare_severity_models(y_test, lgb_preds, xgb_preds, cat_preds=None):
    """
    Compare all severity models and create ensemble.

    Args:
        y_test: Log-transformed target (will be converted back to days)
        lgb_preds: LightGBM predictions (actual days)
        xgb_preds: XGBoost predictions (actual days)
        cat_preds: Optional CatBoost predictions (actual days)

    Returns:
        DataFrame with model comparison metrics
    """
    def rmse(y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred) ** 2))

    y_true = np.expm1(y_test) if y_test.mean() < 10 else y_test  # Check if log-transformed

    results = []

    results.append({
        "Model": "LightGBM",
        "MAE (days)": mean_absolute_error(y_true, lgb_preds),
        "RMSE (days)": rmse(y_true, lgb_preds),
        "R²": r2_score(y_true, lgb_preds)
    })

    results.append({
        "Model": "XGBoost",
        "MAE (days)": mean_absolute_error(y_true, xgb_preds),
        "RMSE (days)": rmse(y_true, xgb_preds),
        "R²": r2_score(y_true, xgb_preds)
    })

    if cat_preds is not None:
        results.append({
            "Model": "CatBoost (Tuned)",
            "MAE (days)": mean_absolute_error(y_true, cat_preds),
            "RMSE (days)": rmse(y_true, cat_preds),
            "R²": r2_score(y_true, cat_preds)
        })

        # Simple average ensemble
        avg_preds = (lgb_preds + xgb_preds + cat_preds) / 3
        results.append({
            "Model": "Ensemble (Average)",
            "MAE (days)": mean_absolute_error(y_true, avg_preds),
            "RMSE (days)": rmse(y_true, avg_preds),
            "R²": r2_score(y_true, avg_preds)
        })

    comparison_df = pd.DataFrame(results).set_index("Model")
    print("\nSeverity Regression Comparison (Temporal Validation):")
    print(comparison_df.round(2))

    return comparison_df


def diagnose_severity_target(y_train, y_test, X_test=None):
    """
    Analyze severity target distribution and compute baselines.

    This helps understand if regression is viable or if classification
    might be more appropriate.

    Args:
        y_train: Training target (log-transformed)
        y_test: Test target (log-transformed)
        X_test: Optional test features to check expected_severity_days

    Returns:
        dict with diagnostic information
    """
    y_true = np.expm1(y_test)
    y_train_days = np.expm1(y_train)

    print('=' * 60)
    print('TARGET DISTRIBUTION ANALYSIS')
    print('=' * 60)
    print(f'Target (severity_days):')
    print(f'  Train mean:   {y_train_days.mean():.1f} days')
    print(f'  Test mean:    {y_true.mean():.1f} days')
    print(f'  Test median:  {np.median(y_true):.1f} days')
    print(f'  Test std:     {y_true.std():.1f} days')
    print(f'  Test range:   {y_true.min():.0f} - {y_true.max():.0f} days')

    # Baseline: predict the mean
    mean_pred = np.full_like(y_true, y_true.mean())
    mean_mae = mean_absolute_error(y_true, mean_pred)
    mean_r2 = r2_score(y_true, mean_pred)

    print(f'\nBASELINE (predict mean):')
    print(f'  MAE:  {mean_mae:.1f} days')
    print(f'  R²:   {mean_r2:.3f}')

    # Baseline: predict median
    median_pred = np.full_like(y_true, np.median(y_true))
    median_mae = mean_absolute_error(y_true, median_pred)

    print(f'\nBASELINE (predict median):')
    print(f'  MAE:  {median_mae:.1f} days')

    result = {
        "mean_baseline_mae": mean_mae,
        "median_baseline_mae": median_mae,
        "test_mean": y_true.mean(),
        "test_std": y_true.std()
    }

    # Feature-based baseline: expected_severity_days
    if X_test is not None and 'expected_severity_days' in X_test.columns:
        expected = X_test['expected_severity_days'].values
        expected_mae = mean_absolute_error(y_true, expected)
        expected_r2 = r2_score(y_true, expected)

        print(f'\nBASELINE (expected_severity_days feature):')
        print(f'  MAE:  {expected_mae:.1f} days')
        print(f'  R²:   {expected_r2:.3f}')

        result["expected_feature_mae"] = expected_mae

    # Show severity distribution
    print('\nSEVERITY BINS:')
    bins = [0, 7, 14, 30, 60, 120, 1000]
    labels = ['<1wk', '1-2wk', '2wk-1mo', '1-2mo', '2-4mo', '>4mo']
    cuts = pd.cut(y_true, bins=bins, labels=labels)
    print(cuts.value_counts().sort_index())

    return result