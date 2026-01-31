"""
Example: Using Temporal Validation to Prevent Temporal Leakage

This script demonstrates how to use the temporal_validation module to create
proper time-based splits for injury prediction models.

Author: Claude Sonnet 4.5
Date: 2026-01-23
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.temporal_validation import (
    temporal_split,
    walk_forward_validation,
    get_temporal_classification_splits,
    get_temporal_severity_splits,
    compare_random_vs_temporal_split,
    check_temporal_leakage
)


def example_1_basic_temporal_split():
    """
    Example 1: Basic temporal split for classification
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Temporal Split")
    print("="*80)

    # Load your injury data (adjust path as needed)
    # df = pd.read_csv("path/to/your/injury_data.csv")

    # For demonstration, create synthetic data
    dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='D')
    df = pd.DataFrame({
        'Date of Injury': np.random.choice(dates, size=1000),
        'injury_label': np.random.randint(0, 2, size=1000),
        'feature_1': np.random.randn(1000),
        'feature_2': np.random.randn(1000),
    })

    # Create temporal splits
    X_train, X_val, X_test, y_train, y_val, y_test = get_temporal_classification_splits(
        df=df,
        date_column="Date of Injury",
        target_column="injury_label",
        train_end="2022-06-30",
        val_end="2023-06-30"
    )

    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")

    # Now you can train your model on X_train, y_train
    # Tune hyperparameters on X_val, y_val
    # Final evaluation on X_test, y_test


def example_2_walk_forward_validation():
    """
    Example 2: Walk-forward validation (gold standard)
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Walk-Forward Validation")
    print("="*80)

    # Create synthetic data
    dates = pd.date_range(start='2018-01-01', end='2024-12-31', freq='D')
    df = pd.DataFrame({
        'Date of Injury': np.random.choice(dates, size=2000),
        'injury_label': np.random.randint(0, 2, size=2000),
        'feature_1': np.random.randn(2000),
        'feature_2': np.random.randn(2000),
        'feature_3': np.random.randn(2000),
    })

    # Perform walk-forward validation
    splits = walk_forward_validation(
        df=df,
        date_column="Date of Injury",
        target_column="injury_label",
        n_splits=5,
        test_size_months=6,
        gap_months=0
    )

    # Train and evaluate on each fold
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score

    scores = []
    for i, (X_train, X_test, y_train, y_test) in enumerate(splits):
        # Simple model for demonstration
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        probs = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probs)
        scores.append(auc)

        print(f"Fold {i+1} ROC-AUC: {auc:.4f}")

    print(f"\nMean ROC-AUC: {np.mean(scores):.4f} ± {np.std(scores):.4f}")


def example_3_severity_regression():
    """
    Example 3: Temporal splits for severity regression
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Severity Regression with Temporal Splits")
    print("="*80)

    # Create synthetic severity data
    dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='W')
    df = pd.DataFrame({
        'Date of Injury': np.random.choice(dates, size=500),
        'severity_days': np.random.exponential(scale=14, size=500),  # Average 14 days
        'age': np.random.uniform(18, 35, size=500),
        'workload': np.random.uniform(0, 100, size=500),
        'previous_injuries': np.random.randint(0, 5, size=500),
    })

    # Create temporal splits
    X_train, X_val, X_test, y_train, y_val, y_test = get_temporal_severity_splits(
        df=df,
        date_column="Date of Injury",
        target_column="severity_days",
        train_end="2022-06-30",
        val_end="2023-06-30"
    )

    # Train a simple regression model
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, root_mean_squared_error

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate on validation set
    preds_val = model.predict(X_val)
    mae_val = mean_absolute_error(y_val, preds_val)
    rmse_val = root_mean_squared_error(y_val, preds_val)

    print(f"\nValidation Performance:")
    print(f"  MAE:  {mae_val:.2f} days")
    print(f"  RMSE: {rmse_val:.2f} days")

    # Final evaluation on test set
    preds_test = model.predict(X_test)
    mae_test = mean_absolute_error(y_test, preds_test)
    rmse_test = root_mean_squared_error(y_test, preds_test)

    print(f"\nTest Performance:")
    print(f"  MAE:  {mae_test:.2f} days")
    print(f"  RMSE: {rmse_test:.2f} days")


def example_4_compare_splits():
    """
    Example 4: Compare random vs temporal splits
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Random vs Temporal Split Comparison")
    print("="*80)

    # Create synthetic data with temporal trend
    dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='D')

    # Add temporal drift: injury rate increases over time
    df = pd.DataFrame({
        'Date of Injury': dates[:1000],
        'injury_label': [
            1 if np.random.random() < 0.1 + 0.0001 * i else 0
            for i in range(1000)
        ],
        'feature_1': np.random.randn(1000),
        'feature_2': np.random.randn(1000),
    })

    # Compare distributions
    comparison = compare_random_vs_temporal_split(
        df=df,
        date_column="Date of Injury",
        target_column="injury_label",
        train_end="2022-06-30",
        val_end="2023-06-30"
    )

    print("\nNotice how temporal splits capture the changing injury rate over time,")
    print("while random splits artificially balance the distributions.")


def example_5_check_leakage():
    """
    Example 5: Check for potential temporal leakage in features
    """
    print("\n" + "="*80)
    print("EXAMPLE 5: Check for Temporal Leakage")
    print("="*80)

    # Create data with potentially problematic features
    dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='D')
    df = pd.DataFrame({
        'Date of Injury': np.random.choice(dates, size=500),
        'injury_label': np.random.randint(0, 2, size=500),
        'age': np.random.uniform(18, 35, size=500),
        'workload': np.random.uniform(0, 100, size=500),
        'next_injury_date': np.random.choice(dates, size=500),  # LEAKAGE!
        'future_workload': np.random.uniform(0, 100, size=500),  # LEAKAGE!
    })

    # Check for leakage
    report = check_temporal_leakage(
        df=df,
        date_column="Date of Injury",
        feature_columns=['age', 'workload', 'next_injury_date', 'future_workload']
    )

    print(f"\nTemporal Leakage Report:")
    print(f"  Warnings found: {report['n_warnings']}")

    if report['warnings']:
        print("\n  Issues detected:")
        for warning in report['warnings']:
            print(f"    - {warning}")
    else:
        print("\n  ✓ No obvious temporal leakage detected")


def main():
    """
    Run all examples
    """
    print("\n" + "="*80)
    print("TEMPORAL VALIDATION EXAMPLES")
    print("="*80)
    print("\nThis script demonstrates how to prevent temporal leakage in")
    print("injury prediction models using proper time-based splits.")

    # Run examples
    example_1_basic_temporal_split()
    example_2_walk_forward_validation()
    example_3_severity_regression()
    example_4_compare_splits()
    example_5_check_leakage()

    print("\n" + "="*80)
    print("EXAMPLES COMPLETED")
    print("="*80)
    print("\nKey Takeaways:")
    print("  1. Always use temporal splits for time series data")
    print("  2. Walk-forward validation is the gold standard")
    print("  3. Check for temporal leakage in your features")
    print("  4. Random splits create unrealistic performance estimates")
    print("\nSee src/models/temporal_validation.py for full documentation.")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
