"""Common data utilities for feature preparation and model training."""

import re
import numpy as np
import pandas as pd


def sanitize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sanitize column names for LightGBM compatibility.

    LightGBM doesn't allow special JSON characters in feature names:
    [ ] { } : , "

    Args:
        df: DataFrame with potentially problematic column names

    Returns:
        DataFrame with sanitized column names
    """
    df = df.copy()
    df.columns = [re.sub(r'[\[\]{}:,"]', '_', str(c)) for c in df.columns]
    return df


def encode_categoricals(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    """
    Encode categorical columns as integer codes.

    Args:
        df: DataFrame with categorical columns
        columns: Specific columns to encode (default: all object/category columns)

    Returns:
        DataFrame with encoded categoricals
    """
    df = df.copy()

    if columns is None:
        columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype('category').cat.codes

    return df


def prepare_features_for_lgb(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Prepare features for LightGBM/XGBoost training.

    This function:
    1. Sanitizes column names (removes special JSON chars)
    2. Encodes categorical columns as integers

    Args:
        X_train: Training features
        X_test: Test features

    Returns:
        Tuple of (X_train_prepared, X_test_prepared)
    """
    X_train_enc = sanitize_column_names(X_train)
    X_test_enc = sanitize_column_names(X_test)

    X_train_enc = encode_categoricals(X_train_enc)
    X_test_enc = encode_categoricals(X_test_enc)

    return X_train_enc, X_test_enc


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error.

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        RMSE value
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))
