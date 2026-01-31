# Input Validation for Inference Pipeline

## Overview

The validation module provides comprehensive input validation for the injury risk prediction inference pipeline. It prevents garbage predictions by validating all player data before inference and providing clear, actionable error messages.

## Features

- **Age Validation**: Ensures player ages are within professional football range (16-45 years)
- **Position Validation**: Validates against known football positions (GK, DF, MF, FW, and combinations)
- **FIFA Rating Validation**: Checks ratings are in valid range (0-99, can be null)
- **Workload Metrics Validation**: Ensures all workload metrics are non-negative and within reasonable ranges
- **Required Features Check**: Verifies all required features exist before inference
- **Batch Validation**: Validates entire DataFrames for batch inference
- **Clear Error Messages**: Provides helpful, actionable error messages explaining what's wrong

## Usage

### Single Player Validation

```python
from src.inference.validation import validate_player_input

player_data = {
    "name": "John Doe",
    "age": 25,
    "position": "MF",
    "fifa_rating": 85,
    # ... other features
}

result = validate_player_input(player_data)

if not result['valid']:
    print("Validation errors:")
    for error in result['errors']:
        print(f"  - {error}")

if result['warnings']:
    print("Warnings:")
    for warning in result['warnings']:
        print(f"  - {warning}")
```

### DataFrame Batch Validation

```python
from src.inference.validation import validate_inference_dataframe

# Validate entire DataFrame
result = validate_inference_dataframe(df, strict=True)

if not result['valid']:
    print(result['summary'])

    # Show detailed errors for invalid rows
    for idx, errors in result['row_errors'].items():
        player_name = df.loc[idx, 'name']
        print(f"\n{player_name}:")
        for error in errors:
            print(f"  - {error}")
```

### Inference Pipeline with Validation

```python
from src.inference.inference_pipeline import build_full_inference_df

# Production mode (strict validation - recommended)
df = build_full_inference_df(
    all_matches,
    player_metadata,
    severity_df,
    cat_model,
    lgb_model,
    xgb_model,
    validate_input=True,
    strict=True
)

# Development mode (warnings only)
df = build_full_inference_df(
    all_matches,
    player_metadata,
    severity_df,
    cat_model,
    lgb_model,
    xgb_model,
    validate_input=True,
    strict=False
)

# Skip validation (not recommended)
df = build_full_inference_df(
    all_matches,
    player_metadata,
    severity_df,
    cat_model,
    lgb_model,
    xgb_model,
    validate_input=False
)
```

## Validation Rules

### Age Constraints

- **Minimum**: 16 years (youth professional level)
- **Maximum**: 45 years (oldest active players)
- **Cannot be null**

### Position Constraints

Valid positions include:
- `GK`, `Goalkeeper`
- `DF`, `Defender`, `Defense`, `Centre-Back`, `Center-Back`, `Left-Back`, `Right-Back`
- `MF`, `Midfielder`, `Midfield`, `Defensive Midfield`, `Central Midfield`, `Attacking Midfield`
- `FW`, `Forward`, `Attack`, `Striker`, `Winger`, `Left Winger`, `Right Winger`
- Combined positions: `DF,MF`, `MF,DF`, `FW,MF`, `MF,FW`, `DF,FW`, `FW,DF`

### FIFA Rating Constraints

- **Range**: 0-99
- **Can be null** (for players without FIFA ratings)

### Workload Metrics Constraints

| Metric | Min | Max | Description |
|--------|-----|-----|-------------|
| `matches_last_7` | 0 | 7 | Matches in last 7 days |
| `matches_last_14` | 0 | 14 | Matches in last 14 days |
| `matches_last_30` | 0 | 30 | Matches in last 30 days |
| `rest_days_before_injury` | 0 | 365 | Days of rest before injury |
| `avg_rest_last_5` | 0 | 365 | Average rest days in last 5 matches |
| `acute_load` | 0 | 50 | 7-day workload |
| `chronic_load` | 0 | 200 | 28-day workload |
| `acwr` | 0 | 5.0 | Acute:Chronic Workload Ratio |
| `monotony` | 0 | 100 | Workload monotony index |
| `strain` | 0 | 1000 | Training strain |
| `fatigue_index` | -100 | 100 | Fatigue accumulation |
| `workload_slope` | -10 | 10 | Workload trend |

### Team Performance Metrics Constraints

| Metric | Min | Max | Description |
|--------|-----|-----|-------------|
| `goals_for_last_5` | 0 | 50 | Goals scored in last 5 matches |
| `goals_against_last_5` | 0 | 50 | Goals conceded in last 5 matches |
| `goal_diff_last_5` | -50 | 50 | Goal difference in last 5 matches |
| `avg_goal_diff_last_5` | -10 | 10 | Average goal difference |
| `form_last_5` | 0 | 15 | Team form points (0-15) |
| `form_avg_last_5` | 0 | 3 | Average form per match |
| `win_ratio_last_5` | 0 | 1 | Win ratio (0.0-1.0) |
| `win_streak` | 0 | 50 | Current win streak |
| `loss_streak` | 0 | 50 | Current loss streak |

### Injury History Constraints

| Metric | Min | Max | Description |
|--------|-----|-----|-------------|
| `previous_injuries` | 0 | 100 | Number of previous injuries |
| `days_since_last_injury` | 0 | 10000 | Days since last injury (999 = no injury) |

## Required Features

All inference inputs must include these features:

**Player & Static Info:**
- `player_team`
- `position`
- `age`
- `fifa_rating`

**Match Workload & Congestion:**
- `matches_last_7`
- `matches_last_14`
- `matches_last_30`
- `rest_days_before_injury`
- `avg_rest_last_5`

**Performance Rolling Windows:**
- `goals_for_last_5`
- `goals_against_last_5`
- `goal_diff_last_5`
- `avg_goal_diff_last_5`
- `form_last_5`
- `form_avg_last_5`
- `win_ratio_last_5`
- `win_streak`
- `loss_streak`

**Injury History:**
- `previous_injuries`
- `days_since_last_injury`

**Workload Analytics:**
- `acute_load`
- `chronic_load`
- `acwr`
- `monotony`
- `strain`
- `fatigue_index`
- `workload_slope`
- `spike_flag`

## Validation Modes

### Strict Mode (Recommended for Production)

```python
validate_input=True, strict=True
```

- Validates all inputs before inference
- Raises `ValidationError` if any validation fails
- Prevents garbage predictions
- **Use this in production APIs and critical systems**

### Warning Mode (Development)

```python
validate_input=True, strict=False
```

- Validates all inputs
- Logs warnings but allows inference to proceed
- Useful for debugging and development
- **Use this during development and testing**

### Disabled (Not Recommended)

```python
validate_input=False
```

- Skips validation entirely
- May produce unreliable predictions
- Only use if you're absolutely certain inputs are valid

## Error Handling

The validation module provides custom exceptions:

```python
from src.inference.validation import (
    ValidationError,
    AgeValidationError,
    PositionValidationError,
    FIFARatingValidationError,
    WorkloadValidationError,
    MissingFeatureError,
    DataTypeError
)

try:
    result = validate_player_input(player_data)
    if not result['valid']:
        raise ValidationError(result['errors'])
except ValidationError as e:
    logger.error(f"Validation failed: {e}")
    # Handle error appropriately
```

## Testing

Run the validation test suite:

```bash
cd src/inference
python test_validation.py
```

This will run comprehensive tests covering:
- Age validation
- Position validation
- FIFA rating validation
- Workload metrics validation
- Complete player validation
- DataFrame batch validation

## Best Practices

1. **Always validate in production**: Use `strict=True` for production systems
2. **Log validation results**: Monitor validation failures to identify data quality issues
3. **Handle errors gracefully**: Catch `ValidationError` and provide user-friendly error messages
4. **Review warnings**: Even if validation passes, warnings may indicate data quality issues
5. **Keep constraints updated**: Update validation rules if model requirements change

## Warnings

The validation system also generates warnings for suspicious patterns:

- **All workload metrics are zero**: May indicate missing data
- **ACWR > 1.5**: Player in danger zone for elevated injury risk
- **Unusual patterns**: Other suspicious data patterns that don't fail validation but may affect prediction quality

## Example: Complete Validation Flow

```python
import pandas as pd
from src.inference.validation import validate_inference_dataframe, ValidationError
from src.inference.inference_pipeline import build_full_inference_df

# Load your data
df = pd.read_csv('player_data.csv')

# Validate before inference
print("Validating input data...")
validation_result = validate_inference_dataframe(df, strict=True)

if not validation_result['valid']:
    print(f"Validation failed:")
    print(validation_result['summary'])

    # Fix or filter invalid rows
    valid_indices = [
        i for i in df.index
        if i not in validation_result['row_errors']
    ]
    df = df.loc[valid_indices]
    print(f"Proceeding with {len(df)} valid rows")

# Run inference with validation enabled
try:
    results = build_full_inference_df(
        df,
        player_metadata,
        severity_df,
        cat_model,
        lgb_model,
        xgb_model,
        validate_input=True,
        strict=True
    )
    print(f"Successfully generated predictions for {len(results)} players")

except ValidationError as e:
    print(f"Inference failed due to validation error:")
    print(e)
```

## Changelog

### Version 1.0.0 (2026-01-23)

- Initial release
- Comprehensive validation for all input features
- Age, position, FIFA rating validation
- Workload metrics validation
- Team performance metrics validation
- Injury history validation
- Batch DataFrame validation
- Clear, actionable error messages
- Integration with inference pipeline

## Support

For issues or questions about validation:

1. Check this README for validation rules and constraints
2. Run `test_validation.py` to verify validation is working
3. Review logs for detailed validation error messages
4. Update validation constraints in `validation.py` if needed
