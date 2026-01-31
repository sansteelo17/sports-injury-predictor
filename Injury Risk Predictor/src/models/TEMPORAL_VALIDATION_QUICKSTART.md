# Temporal Validation Quick Start Guide

## TL;DR

**Never use random splits for time series data!** Use temporal splits instead.

```python
# ❌ WRONG (causes temporal leakage)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ✅ CORRECT (prevents temporal leakage)
from src.models.temporal_validation import get_temporal_classification_splits
X_train, X_val, X_test, y_train, y_val, y_test = get_temporal_classification_splits(
    df=df, date_column="Date of Injury", target_column="injury_label"
)
```

---

## Three Ways to Use Temporal Validation

### 1. Quick Classification Split (Most Common)

```python
from src.models.temporal_validation import get_temporal_classification_splits

X_train, X_val, X_test, y_train, y_val, y_test = get_temporal_classification_splits(
    df=injury_df,
    date_column="Date of Injury",
    target_column="injury_label",
    train_end="2022-06-30",
    val_end="2023-06-30"
)

# Train → Validate → Test
model.fit(X_train, y_train)
model.tune(X_val, y_val)
final_score = model.score(X_test, y_test)
```

### 2. Quick Regression Split (Severity Prediction)

```python
from src.models.temporal_validation import get_temporal_severity_splits

X_train, X_val, X_test, y_train, y_val, y_test = get_temporal_severity_splits(
    df=severity_df,
    date_column="Date of Injury",
    target_column="severity_days",
    train_end="2022-06-30",
    val_end="2023-06-30"
)

# Train regression model
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### 3. Walk-Forward Validation (Gold Standard)

```python
from src.models.temporal_validation import walk_forward_validation

splits = walk_forward_validation(
    df=injury_df,
    date_column="Date of Injury",
    target_column="injury_label",
    n_splits=5,
    test_size_months=6
)

# Evaluate across multiple time windows
for X_train, X_test, y_train, y_test in splits:
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"Score: {score:.4f}")
```

---

## When to Use Each Method

| Method | Use Case | Pros | Cons |
|--------|----------|------|------|
| **get_temporal_classification_splits()** | Quick experiments, model development | Fast, simple | Single split might not capture variability |
| **get_temporal_severity_splits()** | Regression tasks | Fast, simple | Single split might not capture variability |
| **walk_forward_validation()** | Final model evaluation, production | Most realistic, robust | Slower, uses more memory |

---

## Common Adjustments

### Adjust Split Dates

```python
# Check your data range first
print(df['Date of Injury'].min())  # e.g., 2018-01-01
print(df['Date of Injury'].max())  # e.g., 2024-12-31

# Adjust dates based on your data
X_train, X_val, X_test, y_train, y_val, y_test = get_temporal_classification_splits(
    df=df,
    date_column="Date of Injury",
    target_column="injury_label",
    train_end="2021-12-31",  # ← Adjust this
    val_end="2023-06-30"     # ← Adjust this
)
```

### More/Fewer Walk-Forward Splits

```python
# Quick experiment: 3 splits
splits = walk_forward_validation(df, date_column="Date of Injury", n_splits=3)

# Standard validation: 5 splits
splits = walk_forward_validation(df, date_column="Date of Injury", n_splits=5)

# Rigorous validation: 10 splits
splits = walk_forward_validation(df, date_column="Date of Injury", n_splits=10)
```

### Add Prediction Gap

```python
# Predict injuries 1 month in advance
splits = walk_forward_validation(
    df=injury_df,
    date_column="Date of Injury",
    target_column="injury_label",
    n_splits=5,
    test_size_months=6,
    gap_months=1  # ← 1 month gap between train and test
)
```

---

## Integration with Existing Code

### Replace in classification.py

```python
# OLD CODE (with temporal leakage)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# NEW CODE (no temporal leakage)
from src.models.temporal_validation import get_temporal_classification_splits
X_train, X_val, X_test, y_train, y_val, y_test = get_temporal_classification_splits(
    df=df,  # Use full dataframe, not X and y
    date_column="Date of Injury",
    target_column="injury_label",
    train_end="2022-06-30",
    val_end="2023-06-30"
)
```

### Replace in severity.py

```python
# OLD CODE
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# NEW CODE
from src.models.temporal_validation import get_temporal_severity_splits
X_train, X_val, X_test, y_train, y_val, y_test = get_temporal_severity_splits(
    df=df,
    date_column="Date of Injury",
    target_column="severity_days"
)
```

---

## Troubleshooting

### "Training set is empty"

Your `train_end` date is too early. Check your data range and increase it:

```python
print(df['Date of Injury'].min())  # Check minimum date
# Adjust train_end to be after this date
```

### "Test set is empty"

Your `val_end` date is too late. Check your data range and decrease it:

```python
print(df['Date of Injury'].max())  # Check maximum date
# Adjust val_end to be before this date
```

### "Not enough data for walk-forward validation"

Reduce `n_splits` or `test_size_months`:

```python
# Instead of:
splits = walk_forward_validation(df, n_splits=10, test_size_months=12)

# Try:
splits = walk_forward_validation(df, n_splits=3, test_size_months=6)
```

---

## Complete Example

```python
import pandas as pd
from src.models.temporal_validation import get_temporal_classification_splits
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report

# Load data
df = pd.read_csv("injury_data.csv")

# Check date range
print(f"Data from {df['Date of Injury'].min()} to {df['Date of Injury'].max()}")

# Create temporal splits
X_train, X_val, X_test, y_train, y_val, y_test = get_temporal_classification_splits(
    df=df,
    date_column="Date of Injury",
    target_column="injury_label",
    train_end="2022-06-30",
    val_end="2023-06-30"
)

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Validate (for hyperparameter tuning)
val_probs = model.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, val_probs)
print(f"Validation ROC-AUC: {val_auc:.4f}")

# Final test (only run once!)
test_probs = model.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, test_probs)
print(f"Test ROC-AUC: {test_auc:.4f}")
```

---

## Next Steps

1. **Read the full documentation**: `src/models/README_temporal_validation.md`
2. **Run the examples**: `python examples/temporal_validation_example.py`
3. **Check for leakage**: Use `check_temporal_leakage()` to verify your features
4. **Update your code**: Replace random splits with temporal splits

---

## Remember

- ✅ Always use temporal splits for time series data
- ✅ Test set should ALWAYS be from the future (relative to training)
- ✅ Use walk-forward validation for final model evaluation
- ❌ Never use random K-fold cross-validation on time series
- ❌ Never fit scalers/encoders on the full dataset
- ❌ Never create features using future information

**The #1 rule: Past data predicts future data, not the other way around!**
