# Temporal Validation Module

## Overview

This module provides tools for proper time-based train/validation/test splits to prevent **temporal leakage** in time series machine learning tasks.

## What is Temporal Leakage?

Temporal leakage occurs when a model sees future information during training that wouldn't be available at prediction time in production. This is a critical issue in time series prediction tasks like injury forecasting.

### The Problem with Random Splits

```python
# ❌ WRONG: Random split causes temporal leakage
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

**Why this is wrong:**
- Training data might contain injuries from 2024
- Test data might contain injuries from 2022
- The model learns from the future to predict the past
- Validation metrics are artificially inflated
- Real-world performance will be significantly worse

### The Solution: Temporal Splits

```python
# ✅ CORRECT: Temporal split prevents leakage
from src.models.temporal_validation import get_temporal_classification_splits

X_train, X_val, X_test, y_train, y_val, y_test = get_temporal_classification_splits(
    df=injury_df,
    date_column="Date of Injury",
    target_column="injury_label",
    train_end="2022-06-30",
    val_end="2023-06-30"
)
```

**Why this works:**
- Training: all data before 2022-06-30
- Validation: data from 2022-06-30 to 2023-06-30
- Test: all data after 2023-06-30
- Model only sees past data to predict future events
- Realistic performance estimates

## Quick Start

### Installation

The module is already included in the project. No additional installation required.

### Basic Usage

```python
from src.models.temporal_validation import (
    temporal_split,
    get_temporal_classification_splits,
    get_temporal_severity_splits,
    walk_forward_validation
)
```

## Usage Examples

### Example 1: Classification (Injury Risk Prediction)

```python
import pandas as pd
from src.models.temporal_validation import get_temporal_classification_splits

# Load your data
df = pd.read_csv("injury_data.csv")

# Create temporal splits
X_train, X_val, X_test, y_train, y_val, y_test = get_temporal_classification_splits(
    df=df,
    date_column="Date of Injury",
    target_column="injury_label",
    train_end="2022-06-30",  # Adjust based on your data
    val_end="2023-06-30"
)

# Train your model
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Tune on validation set
val_score = model.score(X_val, y_val)

# Final evaluation on test set
test_score = model.score(X_test, y_test)
```

### Example 2: Regression (Injury Severity Prediction)

```python
from src.models.temporal_validation import get_temporal_severity_splits

# Create temporal splits for severity regression
X_train, X_val, X_test, y_train, y_val, y_test = get_temporal_severity_splits(
    df=severity_df,
    date_column="Date of Injury",
    target_column="severity_days",
    train_end="2022-06-30",
    val_end="2023-06-30"
)

# Train regression model
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()
model.fit(X_train, y_train)

# Evaluate
from sklearn.metrics import mean_absolute_error

preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
print(f"Test MAE: {mae:.2f} days")
```

### Example 3: Walk-Forward Validation (Gold Standard)

Walk-forward validation is the most rigorous way to evaluate time series models. It simulates real-world deployment by repeatedly:
1. Training on all past data
2. Testing on future data
3. Moving the window forward in time

```python
from src.models.temporal_validation import walk_forward_validation

# Perform walk-forward validation
splits = walk_forward_validation(
    df=injury_df,
    date_column="Date of Injury",
    target_column="injury_label",
    n_splits=5,              # Number of train/test splits
    test_size_months=6,      # Each test window is 6 months
    gap_months=0             # No gap between train and test
)

# Evaluate model on each fold
from sklearn.metrics import roc_auc_score
import numpy as np

scores = []
for i, (X_train, X_test, y_train, y_test) in enumerate(splits):
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    scores.append(auc)
    print(f"Fold {i+1} ROC-AUC: {auc:.4f}")

print(f"\nMean ROC-AUC: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
```

### Example 4: Custom Temporal Split

```python
from src.models.temporal_validation import temporal_split

# Get train/val/test dataframes
train_df, val_df, test_df = temporal_split(
    df=injury_df,
    date_column="Date of Injury",
    train_end="2022-06-30",
    val_end="2023-06-30",
    target_column=None  # Don't separate X and y yet
)

# Now you can process each split separately
# For example, apply different feature engineering to each split
```

## API Reference

### `temporal_split()`

Core function for creating time-based splits.

**Parameters:**
- `df` (pd.DataFrame): Input dataframe with date column
- `date_column` (str): Name of the column containing dates
- `train_end` (str): End date for training set (format: "YYYY-MM-DD")
- `val_end` (str): End date for validation set (format: "YYYY-MM-DD")
- `target_column` (str, optional): If provided, returns X and y separately
- `return_indices` (bool): If True, also returns indices used for each split

**Returns:**
- If `target_column=None`: `train_df, val_df, test_df`
- If `target_column` provided: `X_train, X_val, X_test, y_train, y_val, y_test`

### `walk_forward_validation()`

Perform walk-forward (rolling window) validation.

**Parameters:**
- `df` (pd.DataFrame): Input dataframe
- `date_column` (str): Name of date column
- `target_column` (str): Name of target column
- `n_splits` (int): Number of train/test splits to create
- `test_size_months` (int): Size of each test window in months
- `gap_months` (int): Gap between training and test set (prediction horizon)

**Returns:**
- `splits` (List[Tuple]): List of (X_train, X_test, y_train, y_test) tuples

### `get_temporal_classification_splits()`

Convenience wrapper for classification tasks.

**Parameters:**
- `df` (pd.DataFrame): Input dataframe
- `date_column` (str): Name of date column
- `target_column` (str): Name of target column (default: "injury_label")
- `train_end` (str): Training end date (default: "2022-06-30")
- `val_end` (str): Validation end date (default: "2023-06-30")

**Returns:**
- `X_train, X_val, X_test, y_train, y_val, y_test`

### `get_temporal_severity_splits()`

Convenience wrapper for severity regression tasks.

**Parameters:**
- `df` (pd.DataFrame): Input dataframe
- `date_column` (str): Name of date column
- `target_column` (str): Name of target column (default: "severity_days")
- `train_end` (str): Training end date (default: "2022-06-30")
- `val_end` (str): Validation end date (default: "2023-06-30")

**Returns:**
- `X_train, X_val, X_test, y_train, y_val, y_test`

### `check_temporal_leakage()`

Check for potential temporal leakage in features.

**Parameters:**
- `df` (pd.DataFrame): Input dataframe
- `date_column` (str): Reference date column
- `feature_columns` (List[str], optional): Features to check

**Returns:**
- `report` (dict): Dictionary containing warnings about potential leakage

### `compare_random_vs_temporal_split()`

Compare distributions between random and temporal splits.

**Parameters:**
- `df` (pd.DataFrame): Input dataframe
- `date_column` (str): Date column name
- `target_column` (str): Target column name
- `train_end` (str): Training end date
- `val_end` (str): Validation end date

**Returns:**
- `comparison` (pd.DataFrame): Summary statistics comparing both methods

## Best Practices

### 1. Choose Appropriate Split Dates

```python
# Analyze your data's date range first
print(f"Data range: {df['Date of Injury'].min()} to {df['Date of Injury'].max()}")

# Choose split dates that:
# - Give enough training data (at least 60% of total)
# - Leave enough validation and test data (15-20% each)
# - Align with natural breakpoints (e.g., end of season)
```

### 2. Use Walk-Forward Validation for Final Model Selection

```python
# Use simple temporal split for quick experiments
X_train, X_val, X_test, y_train, y_val, y_test = get_temporal_classification_splits(...)

# Use walk-forward validation for final model evaluation
splits = walk_forward_validation(...)
```

### 3. Check for Temporal Leakage in Features

```python
from src.models.temporal_validation import check_temporal_leakage

report = check_temporal_leakage(df, date_column="Date of Injury")

if report['warnings']:
    print("⚠️ Potential temporal leakage detected!")
    for warning in report['warnings']:
        print(f"  - {warning}")
```

### 4. Document Your Split Dates

```python
# Add comments explaining your choice of split dates
# train_end="2022-06-30"  <- End of 2021/22 season
# val_end="2023-06-30"    <- End of 2022/23 season
# Test set: 2023/24 season onwards
```

## Common Pitfalls

### ❌ Using Future Features

```python
# DON'T create features using future data
df['future_injury_count'] = df.groupby('player_id')['injury'].shift(-1)  # WRONG!

# DO create features using only past data
df['past_injury_count'] = df.groupby('player_id')['injury'].shift(1)  # CORRECT
```

### ❌ Leaking Information Through Preprocessing

```python
# DON'T fit scaler on all data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # WRONG! Includes test data

# DO fit scaler only on training data
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)  # CORRECT
```

### ❌ Using Cross-Validation on Time Series

```python
# DON'T use standard K-fold cross-validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5)  # WRONG! Random folds

# DO use walk-forward validation
splits = walk_forward_validation(...)  # CORRECT
```

## Migration Guide

### Migrating from Random Splits

**Before (with temporal leakage):**
```python
from sklearn.model_selection import train_test_split

X = df.drop(columns=["injury_label"])
y = df["injury_label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

**After (no temporal leakage):**
```python
from src.models.temporal_validation import get_temporal_classification_splits

X_train, X_val, X_test, y_train, y_val, y_test = get_temporal_classification_splits(
    df=df,
    date_column="Date of Injury",
    target_column="injury_label",
    train_end="2022-06-30",
    val_end="2023-06-30"
)

# Note: You now have a validation set!
# Use it for hyperparameter tuning before final test evaluation
```

## Performance Considerations

### Memory Usage

Walk-forward validation creates multiple copies of the data. For large datasets:

```python
# Use fewer splits
splits = walk_forward_validation(df, n_splits=3)  # Instead of 5

# Or use simple temporal split for initial experiments
X_train, X_val, X_test, y_train, y_val, y_test = temporal_split(...)
```

### Computation Time

Walk-forward validation trains multiple models:

```python
# For quick experiments: simple temporal split
# For model selection: 3-5 splits
# For final validation: 5-10 splits

# Example: Progressive validation depth
if quick_experiment:
    X_train, X_val, X_test, ... = temporal_split(...)
elif model_selection:
    splits = walk_forward_validation(n_splits=3)
else:  # final_validation
    splits = walk_forward_validation(n_splits=5)
```

## Examples

See `examples/temporal_validation_example.py` for complete working examples:

```bash
cd /path/to/project
python examples/temporal_validation_example.py
```

## Integration with Existing Code

The module is designed as a drop-in replacement for `train_test_split`. See comments in:
- `src/models/classification.py` (lines 27-70)
- `src/models/severity.py` (lines 18-61)

## References

- [Time Series Validation](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split)
- [Temporal Leakage in ML](https://en.wikipedia.org/wiki/Leakage_(machine_learning))
- [Walk-Forward Optimization](https://en.wikipedia.org/wiki/Walk_forward_optimization)

## Support

For questions or issues with temporal validation:
1. Check the docstrings in `src/models/temporal_validation.py`
2. Run `examples/temporal_validation_example.py` for working examples
3. Review the comments in `classification.py` and `severity.py`

## License

Part of the Injury Risk Predictor project.

---

**Remember:** Temporal leakage is one of the most common and serious mistakes in time series machine learning. When in doubt, use temporal splits!
