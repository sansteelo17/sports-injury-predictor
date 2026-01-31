# Validation Quick Reference

## Quick Start

```python
from src.inference.inference_pipeline import build_full_inference_df
from src.inference.validation import ValidationError

# Run inference with validation (recommended)
try:
    results = build_full_inference_df(
        all_matches,
        player_metadata,
        severity_df,
        cat_model,
        lgb_model,
        xgb_model,
        validate_input=True,  # Enable validation
        strict=True           # Fail fast on errors
    )
except ValidationError as e:
    print(f"Validation failed: {e}")
```

## Common Validation Errors

### 1. Age Out of Range

**Error:**
```
Age validation failed: Age 50.0 exceeds maximum (45). Check for data entry errors.
```

**Fix:**
- Ensure age is between 16 and 45
- Check for typos in age field
- Verify birth year calculations are correct

### 2. Invalid Position

**Error:**
```
Position validation failed: Invalid position: 'STRIKER'. Valid positions include: GK, DF, MF, FW...
```

**Fix:**
- Use standardized positions: `GK`, `DF`, `MF`, `FW`
- Or use full names: `Goalkeeper`, `Defender`, `Midfielder`, `Forward`
- Combined positions are allowed: `DF,MF`, `FW,MF`, etc.

### 3. Missing Required Features

**Error:**
```
Missing 10 required features: matches_last_7, matches_last_14, ...
```

**Fix:**
- Ensure all required features are computed before inference
- Run feature engineering pipeline first
- Check column names match exactly (case-sensitive)

### 4. Workload Metric Out of Range

**Error:**
```
matches_last_7 value 10.0 out of valid range (0 to 7)
```

**Fix:**
- Check feature computation logic
- Verify date ranges for rolling windows
- Ensure no duplicate match records

### 5. FIFA Rating Out of Range

**Error:**
```
FIFA rating 105.0 out of valid range (0-99)
```

**Fix:**
- Verify FIFA rating data source
- Check for data type errors (e.g., multiplied by 100)
- FIFA rating can be null, don't fill with invalid values

## Validation Modes

| Mode | validate_input | strict | Behavior | Use Case |
|------|---------------|--------|----------|----------|
| **Production** | `True` | `True` | Raise error on failure | APIs, critical systems |
| **Development** | `True` | `False` | Log warnings only | Testing, debugging |
| **Disabled** | `False` | N/A | Skip validation | Only if certain inputs are valid |

## Validation Constraints at a Glance

| Feature | Valid Range | Nullable |
|---------|-------------|----------|
| `age` | 16-45 | No |
| `position` | GK, DF, MF, FW, combinations | No |
| `fifa_rating` | 0-99 | Yes |
| `matches_last_7` | 0-7 | No |
| `matches_last_14` | 0-14 | No |
| `matches_last_30` | 0-30 | No |
| `acwr` | 0-5.0 | No |
| `acute_load` | 0-50 | No |
| `chronic_load` | 0-200 | No |
| `previous_injuries` | 0-100 | No |
| `days_since_last_injury` | 0-10000 | No |

## Quick Validation Functions

### Validate Single Player

```python
from src.inference.validation import validate_player_input

result = validate_player_input(player_data)
if not result['valid']:
    for error in result['errors']:
        print(error)
```

### Validate DataFrame

```python
from src.inference.validation import validate_inference_dataframe

result = validate_inference_dataframe(df, strict=True)
print(result['summary'])
```

### Validate and Raise

```python
from src.inference.validation import validate_and_raise, ValidationError

try:
    validate_and_raise(player_data, context="API request")
except ValidationError as e:
    # Handle error
    pass
```

## Warning Messages

### ACWR Danger Zone

**Warning:**
```
ACWR ratio 1.80 exceeds 1.5 (danger zone for injury risk)
```

**Meaning:**
- Player's acute:chronic workload ratio is high
- Elevated injury risk
- Not a validation error, but important flag

### All Workload Metrics Zero

**Warning:**
```
All workload metrics are zero. This may indicate missing data and could result in unreliable predictions.
```

**Meaning:**
- Feature engineering may have failed
- Check upstream data pipeline
- Predictions will be unreliable

## Debugging Tips

### 1. Check Data Types

```python
print(df.dtypes)
# Ensure numeric columns are float/int, not object
```

### 2. Inspect Null Values

```python
print(df.isnull().sum())
# High null counts may indicate data quality issues
```

### 3. Check Value Ranges

```python
print(df['age'].describe())
print(df['matches_last_7'].describe())
# Look for outliers and impossible values
```

### 4. Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
# Will show detailed validation messages
```

### 5. Run Test Suite

```bash
cd src/inference
python test_validation.py
```

## Common Patterns

### Pattern 1: API Endpoint

```python
def predict_injury_api(player_data):
    result = validate_player_input(player_data)

    if not result['valid']:
        return {
            "error": "Invalid input",
            "details": result['errors']
        }

    # Run inference
    prediction = model.predict(...)
    return {"risk": prediction}
```

### Pattern 2: Batch Processing

```python
# Validate first
validation_result = validate_inference_dataframe(df, strict=False)

# Filter invalid rows
valid_rows = [i for i in df.index if i not in validation_result['row_errors']]
df_valid = df.loc[valid_rows]

# Process valid data
results = build_full_inference_df(df_valid, ..., validate_input=False)
```

### Pattern 3: Pre-flight Check

```python
# Check before expensive computation
try:
    validate_dataframe_and_raise(df, context="pre-flight check")
except ValidationError as e:
    logger.error(f"Pre-flight validation failed: {e}")
    # Fix data or abort
```

## Error Handling Best Practices

1. **Always catch ValidationError in production**
   ```python
   try:
       results = build_full_inference_df(...)
   except ValidationError as e:
       # Log and return user-friendly error
       logger.error(f"Validation failed: {e}")
       return {"error": "Invalid player data"}
   ```

2. **Log validation results**
   ```python
   result = validate_inference_dataframe(df)
   logger.info(f"Validated {result['total_rows']} rows: "
               f"{result['valid_rows']} valid, "
               f"{result['invalid_rows']} invalid")
   ```

3. **Provide actionable feedback**
   ```python
   if not result['valid']:
       for idx, errors in result['row_errors'].items():
           player = df.loc[idx, 'name']
           print(f"{player}: {errors[0]}")
           # Tell user exactly what to fix
   ```

## Performance Notes

- Validation adds ~10-50ms per 1000 rows
- DataFrame validation is vectorized and efficient
- Use `strict=False` in development for faster iteration
- Validation is worth it - prevents garbage predictions

## Need Help?

1. Check `VALIDATION_README.md` for full documentation
2. Run `test_validation.py` to verify setup
3. Use `example_validation_usage.py` for code examples
4. Enable debug logging for detailed messages
