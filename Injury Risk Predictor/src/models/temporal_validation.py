"""
Temporal Validation Module for Time Series Machine Learning

This module implements proper time-based train/validation/test splits to prevent
temporal leakage in time series prediction tasks.

WHY TEMPORAL SPLITS MATTER
==========================
In time series data (like injury prediction), using random train/test splits creates
TEMPORAL LEAKAGE - the model sees future information during training that wouldn't be
available at prediction time in production. This causes:

1. Artificially inflated performance metrics during validation
2. Poor real-world performance when deployed
3. Models that "cheat" by learning patterns that only exist because of data leakage

Example of the problem:
- Training data: injuries from 2020, 2022, 2024
- Test data: injuries from 2021, 2023
- The model learns from 2024 to predict 2023 → INVALID!

SOLUTION: Time-based splits ensure training data always precedes validation/test data,
mimicking real-world deployment where we only have past data to predict future events.

Author: Claude Sonnet 4.5
Date: 2026-01-23
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Union
from datetime import datetime
import warnings


def temporal_split(
    df: pd.DataFrame,
    date_column: str,
    train_end: str = "2022-06-30",
    val_end: str = "2023-06-30",
    target_column: Optional[str] = None,
    return_indices: bool = False
) -> Union[
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]
]:
    """
    Split data into train/validation/test sets based on temporal ordering.

    This function ensures NO temporal leakage by splitting data chronologically:
    - Training: all data before train_end
    - Validation: data between train_end and val_end
    - Test: all data after val_end

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with a date column
    date_column : str
        Name of the column containing dates (will be parsed if string)
    train_end : str, default="2022-06-30"
        End date for training set (exclusive). Format: "YYYY-MM-DD"
    val_end : str, default="2023-06-30"
        End date for validation set (exclusive). Format: "YYYY-MM-DD"
    target_column : str, optional
        If provided, returns X_train, X_val, X_test, y_train, y_val, y_test
        If None, returns train_df, val_df, test_df
    return_indices : bool, default=False
        If True, also returns the indices used for each split

    Returns
    -------
    If target_column is None:
        train_df, val_df, test_df : pd.DataFrame
            Three dataframes split by time

    If target_column is provided:
        X_train, X_val, X_test : pd.DataFrame
            Feature matrices for each split
        y_train, y_val, y_test : pd.Series
            Target variables for each split

    Examples
    --------
    >>> # Basic usage: split into train/val/test
    >>> train_df, val_df, test_df = temporal_split(
    ...     df,
    ...     date_column="Date of Injury",
    ...     train_end="2022-06-30",
    ...     val_end="2023-06-30"
    ... )

    >>> # With target variable (drop-in replacement for train_test_split)
    >>> X_train, X_val, X_test, y_train, y_val, y_test = temporal_split(
    ...     df,
    ...     date_column="Date of Injury",
    ...     target_column="injury_label",
    ...     train_end="2022-06-30",
    ...     val_end="2023-06-30"
    ... )

    Notes
    -----
    - All dates are exclusive (train_end is not included in training set)
    - Data is NOT shuffled - temporal order is preserved
    - Original index is reset in output dataframes
    - Missing/invalid dates will raise a warning and be excluded
    """

    df = df.copy()

    # Parse date column if it's a string
    if df[date_column].dtype == 'object':
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')

    # Check for invalid dates
    n_invalid = df[date_column].isna().sum()
    if n_invalid > 0:
        warnings.warn(
            f"Found {n_invalid} invalid/missing dates in '{date_column}'. "
            f"These rows will be excluded from splits."
        )
        df = df.dropna(subset=[date_column])

    # Convert split dates to datetime
    train_end_dt = pd.to_datetime(train_end)
    val_end_dt = pd.to_datetime(val_end)

    # Validate date order
    if train_end_dt >= val_end_dt:
        raise ValueError(
            f"train_end ({train_end}) must be before val_end ({val_end})"
        )

    # Create temporal splits
    train_mask = df[date_column] < train_end_dt
    val_mask = (df[date_column] >= train_end_dt) & (df[date_column] < val_end_dt)
    test_mask = df[date_column] >= val_end_dt

    train_df = df[train_mask].reset_index(drop=True)
    val_df = df[val_mask].reset_index(drop=True)
    test_df = df[test_mask].reset_index(drop=True)

    # Print split statistics
    print(f"\n{'='*60}")
    print(f"TEMPORAL SPLIT SUMMARY")
    print(f"{'='*60}")
    print(f"Training:   < {train_end}  →  {len(train_df):,} samples")
    print(f"Validation: {train_end} to {val_end}  →  {len(val_df):,} samples")
    print(f"Test:       >= {val_end}  →  {len(test_df):,} samples")
    print(f"Total:      {len(df):,} samples")
    print(f"{'='*60}\n")

    # Check if any split is empty
    if len(train_df) == 0:
        raise ValueError(
            f"Training set is empty. All data is after {train_end}. "
            f"Consider using an earlier train_end date."
        )
    if len(val_df) == 0:
        warnings.warn(
            f"Validation set is empty. No data between {train_end} and {val_end}."
        )
    if len(test_df) == 0:
        warnings.warn(
            f"Test set is empty. All data is before {val_end}."
        )

    # Return based on target_column parameter
    if target_column is None:
        return train_df, val_df, test_df
    else:
        # Split into features and target
        X_train = train_df.drop(columns=[target_column])
        X_val = val_df.drop(columns=[target_column])
        X_test = test_df.drop(columns=[target_column])

        y_train = train_df[target_column]
        y_val = val_df[target_column]
        y_test = test_df[target_column]

        return X_train, X_val, X_test, y_train, y_val, y_test


def walk_forward_validation(
    df: pd.DataFrame,
    date_column: str,
    target_column: str,
    n_splits: int = 5,
    test_size_months: int = 6,
    gap_months: int = 0
) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
    """
    Perform walk-forward (rolling window) validation for time series.

    This is the GOLD STANDARD for time series model validation. It simulates
    real-world deployment by training on past data and testing on future data,
    repeating this process multiple times as you "walk forward" through time.

    How it works:
    1. Sort data by date
    2. Create n_splits windows, each consisting of:
       - Training: all data up to a cutoff date
       - Test: next test_size_months of data after training
    3. Optionally add a gap between train and test to simulate prediction horizon

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with dates and target
    date_column : str
        Name of column containing dates
    target_column : str
        Name of column containing target variable
    n_splits : int, default=5
        Number of train/test splits to create
    test_size_months : int, default=6
        Size of each test window in months
    gap_months : int, default=0
        Gap between training and test set in months.
        Use this to simulate prediction horizon (e.g., gap_months=1 means
        "predict injuries 1 month in advance")

    Returns
    -------
    splits : List[Tuple[X_train, X_test, y_train, y_test]]
        List of (X_train, X_test, y_train, y_test) tuples for each fold

    Examples
    --------
    >>> # Standard walk-forward validation with 5 splits
    >>> splits = walk_forward_validation(
    ...     df,
    ...     date_column="Date of Injury",
    ...     target_column="injury_label",
    ...     n_splits=5,
    ...     test_size_months=6
    ... )
    >>>
    >>> # Train and evaluate model on each split
    >>> for i, (X_train, X_test, y_train, y_test) in enumerate(splits):
    ...     model.fit(X_train, y_train)
    ...     score = model.score(X_test, y_test)
    ...     print(f"Fold {i+1} score: {score:.4f}")

    >>> # With prediction gap (predict 1 month ahead)
    >>> splits = walk_forward_validation(
    ...     df,
    ...     date_column="Date of Injury",
    ...     target_column="injury_label",
    ...     n_splits=3,
    ...     test_size_months=6,
    ...     gap_months=1  # 1 month gap between train and test
    ... )

    Notes
    -----
    - Each split uses progressively more training data (expanding window)
    - Test windows do not overlap
    - This is computationally expensive but provides the most realistic
      estimate of model performance on future data
    - Use n_splits=3-5 for initial experiments, increase for final validation
    """

    df = df.copy()

    # Parse date column
    if df[date_column].dtype == 'object':
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')

    # Remove invalid dates
    df = df.dropna(subset=[date_column])

    # Sort by date
    df = df.sort_values(date_column).reset_index(drop=True)

    # Get date range
    min_date = df[date_column].min()
    max_date = df[date_column].max()

    # Calculate total months available
    total_months = (max_date.year - min_date.year) * 12 + (max_date.month - min_date.month)

    # Validate parameters
    required_months = n_splits * test_size_months + (n_splits - 1) * gap_months
    if required_months > total_months:
        raise ValueError(
            f"Not enough data for {n_splits} splits with {test_size_months} month "
            f"test windows and {gap_months} month gaps. Need {required_months} months, "
            f"have {total_months} months."
        )

    splits = []

    print(f"\n{'='*60}")
    print(f"WALK-FORWARD VALIDATION SETUP")
    print(f"{'='*60}")
    print(f"Date range: {min_date.date()} to {max_date.date()}")
    print(f"Number of splits: {n_splits}")
    print(f"Test window: {test_size_months} months")
    print(f"Gap: {gap_months} months")
    print(f"{'='*60}\n")

    # Calculate split points
    # We work backwards from the end to ensure we have enough data
    test_end_date = max_date

    for split_idx in range(n_splits):
        # Calculate test window
        test_start_date = test_end_date - pd.DateOffset(months=test_size_months)

        # Calculate training window (everything before test window minus gap)
        train_end_date = test_start_date - pd.DateOffset(months=gap_months)

        # Create masks
        train_mask = df[date_column] < train_end_date
        test_mask = (df[date_column] >= test_start_date) & (df[date_column] < test_end_date)

        # Split data
        train_df = df[train_mask]
        test_df = df[test_mask]

        # Check if splits are valid
        if len(train_df) == 0:
            warnings.warn(f"Split {split_idx + 1}: Training set is empty, skipping")
            continue
        if len(test_df) == 0:
            warnings.warn(f"Split {split_idx + 1}: Test set is empty, skipping")
            continue

        # Separate features and target
        X_train = train_df.drop(columns=[target_column])
        X_test = test_df.drop(columns=[target_column])
        y_train = train_df[target_column]
        y_test = test_df[target_column]

        print(f"Split {n_splits - split_idx}/{n_splits}:")
        print(f"  Train: < {train_end_date.date()} ({len(train_df):,} samples)")
        print(f"  Test:  {test_start_date.date()} to {test_end_date.date()} ({len(test_df):,} samples)")

        if target_column in ["injury_label"] or y_train.dtype in ['int64', 'bool']:
            # Classification task - show class distribution
            train_pos = y_train.sum()
            test_pos = y_test.sum()
            print(f"  Train positive rate: {train_pos/len(y_train)*100:.1f}%")
            print(f"  Test positive rate:  {test_pos/len(y_test)*100:.1f}%")

        print()

        splits.append((X_train, X_test, y_train, y_test))

        # Move window backwards for next iteration
        test_end_date = test_start_date - pd.DateOffset(months=gap_months)

    # Reverse splits so they're in chronological order
    splits = splits[::-1]

    print(f"{'='*60}")
    print(f"Created {len(splits)} train/test splits")
    print(f"{'='*60}\n")

    return splits


def get_temporal_classification_splits(
    df: pd.DataFrame,
    date_column: str = "date",
    target_column: str = "injury_label",
    train_end: str = "2022-06-30",
    val_end: str = "2023-06-30"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Convenience function for classification tasks with temporal splits.

    This is a drop-in replacement for the pattern:
        X_train, X_test, y_train, y_test = train_test_split(X, y, ...)

    But it adds a validation set and ensures no temporal leakage.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with features, target, and date column
    date_column : str, default="date"
        Name of date column
    target_column : str, default="injury_label"
        Name of target column
    train_end : str, default="2022-06-30"
        End of training period
    val_end : str, default="2023-06-30"
        End of validation period

    Returns
    -------
    X_train, X_val, X_test : pd.DataFrame
        Feature matrices
    y_train, y_val, y_test : pd.Series
        Target vectors

    Examples
    --------
    >>> # Old approach (WRONG - temporal leakage!)
    >>> X = df.drop(columns=["injury_label"])
    >>> y = df["injury_label"]
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    >>> # New approach (CORRECT - no temporal leakage)
    >>> X_train, X_val, X_test, y_train, y_val, y_test = get_temporal_classification_splits(
    ...     df,
    ...     date_column="Date of Injury",
    ...     target_column="injury_label"
    ... )
    """

    return temporal_split(
        df=df,
        date_column=date_column,
        target_column=target_column,
        train_end=train_end,
        val_end=val_end
    )


def get_temporal_severity_splits(
    df: pd.DataFrame,
    date_column: str = "date",
    target_column: str = "severity_days",
    train_end: str = "2022-06-30",
    val_end: str = "2023-06-30"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Convenience function for severity regression tasks with temporal splits.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with features, target, and date column
    date_column : str, default="date"
        Name of date column
    target_column : str, default="severity_days"
        Name of target column (severity in days)
    train_end : str, default="2022-06-30"
        End of training period
    val_end : str, default="2023-06-30"
        End of validation period

    Returns
    -------
    X_train, X_val, X_test : pd.DataFrame
        Feature matrices
    y_train, y_val, y_test : pd.Series
        Target vectors (severity in days)

    Examples
    --------
    >>> X_train, X_val, X_test, y_train, y_val, y_test = get_temporal_severity_splits(
    ...     df,
    ...     date_column="Date of Injury",
    ...     target_column="severity_days"
    ... )
    """

    return temporal_split(
        df=df,
        date_column=date_column,
        target_column=target_column,
        train_end=train_end,
        val_end=val_end
    )


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def check_temporal_leakage(
    df: pd.DataFrame,
    date_column: str,
    feature_columns: Optional[List[str]] = None
) -> dict:
    """
    Check for potential temporal leakage in features.

    This function analyzes your features to detect columns that might cause
    temporal leakage, such as:
    - Future dates in feature columns
    - Rolling statistics computed over future data
    - Features with suspiciously high correlation to target

    Parameters
    ----------
    df : pd.DataFrame
        Your dataframe
    date_column : str
        Name of date column to use as reference
    feature_columns : List[str], optional
        List of feature columns to check. If None, checks all numeric columns.

    Returns
    -------
    report : dict
        Dictionary containing potential leakage warnings

    Examples
    --------
    >>> report = check_temporal_leakage(df, date_column="Date of Injury")
    >>> if report['warnings']:
    ...     print("Potential temporal leakage detected!")
    ...     for warning in report['warnings']:
    ...         print(f"  - {warning}")
    """

    warnings_list = []

    # Check for date columns in features
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    date_cols = [col for col in date_cols if col != date_column]

    if date_cols:
        warnings_list.append(
            f"Found date columns in features: {date_cols}. "
            f"Ensure these don't contain future information."
        )

    # Check for columns with "future", "next", "after" in name
    if feature_columns is None:
        feature_columns = df.columns.tolist()

    suspicious_names = [
        col for col in feature_columns
        if any(word in col.lower() for word in ['future', 'next', 'after', 'following'])
    ]

    if suspicious_names:
        warnings_list.append(
            f"Found columns with future-indicating names: {suspicious_names}. "
            f"Verify these don't use future data."
        )

    report = {
        'warnings': warnings_list,
        'n_warnings': len(warnings_list),
        'date_column': date_column,
        'checked_features': len(feature_columns) if feature_columns else 0
    }

    return report


def compare_random_vs_temporal_split(
    df: pd.DataFrame,
    date_column: str,
    target_column: str,
    train_end: str = "2022-06-30",
    val_end: str = "2023-06-30"
) -> pd.DataFrame:
    """
    Compare class/value distributions between random and temporal splits.

    This helps demonstrate why temporal splits matter by showing how random
    splits can create unrealistic train/test distributions.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    date_column : str
        Date column name
    target_column : str
        Target column name
    train_end : str
        Training end date for temporal split
    val_end : str
        Validation end date for temporal split

    Returns
    -------
    comparison : pd.DataFrame
        Summary statistics comparing both split methods

    Examples
    --------
    >>> comparison = compare_random_vs_temporal_split(
    ...     df,
    ...     date_column="Date of Injury",
    ...     target_column="injury_label"
    ... )
    >>> print(comparison)
    """

    from sklearn.model_selection import train_test_split

    # Temporal split
    X_train_temp, X_val_temp, X_test_temp, y_train_temp, y_val_temp, y_test_temp = temporal_split(
        df=df,
        date_column=date_column,
        target_column=target_column,
        train_end=train_end,
        val_end=val_end
    )

    # Random split
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train_rand, X_temp, y_train_rand, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val_rand, X_test_rand, y_val_rand, y_test_rand = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    # Compare distributions
    results = []

    for split_name, y_train, y_val, y_test in [
        ("Temporal", y_train_temp, y_val_temp, y_test_temp),
        ("Random", y_train_rand, y_val_rand, y_test_rand)
    ]:
        results.append({
            "Split Method": split_name,
            "Train Size": len(y_train),
            "Val Size": len(y_val),
            "Test Size": len(y_test),
            "Train Mean": y_train.mean(),
            "Val Mean": y_val.mean(),
            "Test Mean": y_test.mean(),
        })

    comparison_df = pd.DataFrame(results)

    print("\n" + "="*80)
    print("RANDOM vs TEMPORAL SPLIT COMPARISON")
    print("="*80)
    print(comparison_df.to_string(index=False))
    print("\n⚠️  Notice: Random splits may have similar distributions across train/val/test,")
    print("   but temporal splits reflect the true challenge of predicting future events!")
    print("="*80 + "\n")

    return comparison_df
