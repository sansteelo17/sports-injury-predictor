"""
Input validation for inference pipeline.

Prevents garbage predictions by validating all player data before inference.
Provides clear, actionable error messages for invalid inputs.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union


# ============================================================
# VALIDATION CONSTANTS
# ============================================================

# Valid football positions (simplified categories used in model)
VALID_POSITIONS = {
    "GK", "Goalkeeper",
    "DF", "Defender", "Defense", "Centre-Back", "Center-Back", "Left-Back", "Right-Back",
    "MF", "Midfielder", "Midfield", "Defensive Midfield", "Central Midfield", "Attacking Midfield",
    "FW", "Forward", "Attack", "Striker", "Winger", "Left Winger", "Right Winger",
    # Combined positions (from dataset)
    "DF,MF", "MF,DF", "FW,MF", "MF,FW", "DF,FW", "FW,DF"
}

# Age constraints (professional football players)
MIN_AGE = 16
MAX_AGE = 45

# FIFA rating constraints
MIN_FIFA_RATING = 0
MAX_FIFA_RATING = 99

# Workload metric constraints
WORKLOAD_CONSTRAINTS = {
    "matches_last_7": (0, 7),
    "matches_last_14": (0, 14),
    "matches_last_30": (0, 30),
    "rest_days_before_injury": (0, 365),
    "avg_rest_last_5": (0, 365),
    "acute_load": (0, 50),
    "chronic_load": (0, 200),
    "acwr": (0, 5.0),
    "monotony": (0, 100),
    "strain": (0, 1000),
    "fatigue_index": (-100, 100),
    "workload_slope": (-10, 10),
}

# Team performance metric constraints
TEAM_PERFORMANCE_CONSTRAINTS = {
    "goals_for_last_5": (0, 50),
    "goals_against_last_5": (0, 50),
    "goal_diff_last_5": (-50, 50),
    "avg_goal_diff_last_5": (-10, 10),
    "form_last_5": (0, 15),
    "form_avg_last_5": (0, 3),
    "win_ratio_last_5": (0, 1),
    "win_streak": (0, 50),
    "loss_streak": (0, 50),
}

# Injury history constraints
INJURY_HISTORY_CONSTRAINTS = {
    "previous_injuries": (0, 100),
    "days_since_last_injury": (0, 10000),
}

# Required features for inference (from classification.py)
REQUIRED_FEATURES = [
    # Player + static info
    "player_team", "position", "age", "fifa_rating",

    # Match workload + congestion
    "matches_last_7", "matches_last_14", "matches_last_30",
    "rest_days_before_injury", "avg_rest_last_5",

    # Performance rolling windows
    "goals_for_last_5", "goals_against_last_5",
    "goal_diff_last_5", "avg_goal_diff_last_5",
    "form_last_5", "form_avg_last_5",
    "win_ratio_last_5", "win_streak", "loss_streak",

    # Injury history
    "previous_injuries", "days_since_last_injury",

    # Workload analytics
    "acute_load", "chronic_load", "acwr",
    "monotony", "strain", "fatigue_index",
    "workload_slope", "spike_flag",
]


# ============================================================
# VALIDATION EXCEPTIONS
# ============================================================

class ValidationError(Exception):
    """Base exception for validation errors."""
    pass


class AgeValidationError(ValidationError):
    """Exception for invalid age values."""
    pass


class PositionValidationError(ValidationError):
    """Exception for invalid position values."""
    pass


class FIFARatingValidationError(ValidationError):
    """Exception for invalid FIFA rating values."""
    pass


class WorkloadValidationError(ValidationError):
    """Exception for invalid workload metric values."""
    pass


class MissingFeatureError(ValidationError):
    """Exception for missing required features."""
    pass


class DataTypeError(ValidationError):
    """Exception for incorrect data types."""
    pass


# ============================================================
# VALIDATION FUNCTIONS
# ============================================================

def validate_age(age: Union[int, float, None]) -> Tuple[bool, Optional[str]]:
    """
    Validate player age.

    Args:
        age: Player age

    Returns:
        Tuple of (is_valid, error_message)
    """
    if age is None or pd.isna(age):
        return False, "Age is required and cannot be null"

    try:
        age_float = float(age)
    except (ValueError, TypeError):
        return False, f"Age must be numeric, got: {type(age).__name__}"

    if age_float < MIN_AGE:
        return False, f"Age {age_float} is below minimum ({MIN_AGE}). Professional players must be at least {MIN_AGE} years old."

    if age_float > MAX_AGE:
        return False, f"Age {age_float} exceeds maximum ({MAX_AGE}). Check for data entry errors."

    return True, None


def validate_fifa_rating(rating: Union[int, float, None]) -> Tuple[bool, Optional[str]]:
    """
    Validate FIFA rating.

    Args:
        rating: FIFA rating (can be null)

    Returns:
        Tuple of (is_valid, error_message)
    """
    # FIFA rating can be null
    if rating is None or pd.isna(rating):
        return True, None

    try:
        rating_float = float(rating)
    except (ValueError, TypeError):
        return False, f"FIFA rating must be numeric, got: {type(rating).__name__}"

    if rating_float < MIN_FIFA_RATING or rating_float > MAX_FIFA_RATING:
        return False, f"FIFA rating {rating_float} out of valid range ({MIN_FIFA_RATING}-{MAX_FIFA_RATING})"

    return True, None


def validate_position(position: Union[str, None]) -> Tuple[bool, Optional[str]]:
    """
    Validate player position.

    Args:
        position: Player position

    Returns:
        Tuple of (is_valid, error_message)
    """
    if position is None or pd.isna(position):
        return False, "Position is required and cannot be null"

    if not isinstance(position, str):
        return False, f"Position must be a string, got: {type(position).__name__}"

    position_clean = position.strip()

    if not position_clean:
        return False, "Position cannot be empty string"

    if position_clean not in VALID_POSITIONS:
        valid_list = sorted(VALID_POSITIONS)[:10]  # Show first 10 examples
        return False, (
            f"Invalid position: '{position_clean}'. "
            f"Valid positions include: {', '.join(valid_list)}, etc. "
            f"Total {len(VALID_POSITIONS)} valid positions."
        )

    return True, None


def validate_workload_metric(
    metric_name: str,
    value: Union[int, float, None],
    allow_null: bool = False
) -> Tuple[bool, Optional[str]]:
    """
    Validate workload metric value.

    Args:
        metric_name: Name of the workload metric
        value: Metric value
        allow_null: Whether null values are allowed

    Returns:
        Tuple of (is_valid, error_message)
    """
    if value is None or pd.isna(value):
        if allow_null:
            return True, None
        return False, f"{metric_name} cannot be null"

    try:
        value_float = float(value)
    except (ValueError, TypeError):
        return False, f"{metric_name} must be numeric, got: {type(value).__name__}"

    # Check if metric has defined constraints
    if metric_name in WORKLOAD_CONSTRAINTS:
        min_val, max_val = WORKLOAD_CONSTRAINTS[metric_name]
        if value_float < min_val or value_float > max_val:
            return False, (
                f"{metric_name} value {value_float} out of valid range "
                f"({min_val} to {max_val})"
            )
    elif metric_name in TEAM_PERFORMANCE_CONSTRAINTS:
        min_val, max_val = TEAM_PERFORMANCE_CONSTRAINTS[metric_name]
        if value_float < min_val or value_float > max_val:
            return False, (
                f"{metric_name} value {value_float} out of valid range "
                f"({min_val} to {max_val})"
            )
    elif metric_name in INJURY_HISTORY_CONSTRAINTS:
        min_val, max_val = INJURY_HISTORY_CONSTRAINTS[metric_name]
        if value_float < min_val or value_float > max_val:
            return False, (
                f"{metric_name} value {value_float} out of valid range "
                f"({min_val} to {max_val})"
            )
    else:
        # Generic non-negative check for unspecified metrics
        if value_float < 0:
            return False, f"{metric_name} cannot be negative (got {value_float})"

    return True, None


def validate_required_features(data: Dict) -> Tuple[bool, Optional[str]]:
    """
    Validate that all required features exist.

    Args:
        data: Dictionary of player data

    Returns:
        Tuple of (is_valid, error_message)
    """
    missing_features = [f for f in REQUIRED_FEATURES if f not in data]

    if missing_features:
        return False, (
            f"Missing {len(missing_features)} required features: "
            f"{', '.join(missing_features[:10])}"
            f"{'...' if len(missing_features) > 10 else ''}"
        )

    return True, None


def validate_player_input(player_data: Dict) -> Dict:
    """
    Comprehensive validation for a single player's inference input.

    Validates:
    - All required features exist
    - Age is within valid range
    - Position is valid
    - FIFA rating is valid (if provided)
    - Workload metrics are within reasonable ranges
    - Team performance metrics are valid
    - Injury history metrics are valid

    Args:
        player_data: Dictionary containing player features

    Returns:
        Dictionary with validation results:
        {
            'valid': bool,
            'errors': list of error messages,
            'warnings': list of warning messages
        }

    Raises:
        ValidationError: If validation fails and raise_on_error=True

    Example:
        >>> result = validate_player_input({
        ...     'name': 'John Doe',
        ...     'age': 25,
        ...     'position': 'MF',
        ...     ...
        ... })
        >>> if not result['valid']:
        ...     print(result['errors'])
    """
    errors = []
    warnings = []

    # 1. Check required features
    is_valid, error_msg = validate_required_features(player_data)
    if not is_valid:
        errors.append(error_msg)
        # Return early if required features are missing
        return {
            'valid': False,
            'errors': errors,
            'warnings': warnings
        }

    # 2. Validate age
    is_valid, error_msg = validate_age(player_data.get('age'))
    if not is_valid:
        errors.append(f"Age validation failed: {error_msg}")

    # 3. Validate position
    is_valid, error_msg = validate_position(player_data.get('position'))
    if not is_valid:
        errors.append(f"Position validation failed: {error_msg}")

    # 4. Validate FIFA rating
    is_valid, error_msg = validate_fifa_rating(player_data.get('fifa_rating'))
    if not is_valid:
        errors.append(f"FIFA rating validation failed: {error_msg}")

    # 5. Validate workload metrics
    all_constraints = {
        **WORKLOAD_CONSTRAINTS,
        **TEAM_PERFORMANCE_CONSTRAINTS,
        **INJURY_HISTORY_CONSTRAINTS
    }

    for metric_name in all_constraints.keys():
        if metric_name in player_data:
            is_valid, error_msg = validate_workload_metric(
                metric_name,
                player_data[metric_name],
                allow_null=False
            )
            if not is_valid:
                errors.append(error_msg)

    # 6. Check for suspicious patterns
    # Warn if all workload metrics are zero (likely missing data)
    workload_keys = list(WORKLOAD_CONSTRAINTS.keys())
    workload_values = [player_data.get(k, 0) for k in workload_keys if k in player_data]

    if workload_values and all(v == 0 for v in workload_values):
        warnings.append(
            "All workload metrics are zero. This may indicate missing data "
            "and could result in unreliable predictions."
        )

    # Warn if ACWR is in danger zone
    acwr = player_data.get('acwr')
    if acwr is not None and not pd.isna(acwr):
        if float(acwr) > 1.5:
            warnings.append(
                f"ACWR ratio {acwr:.2f} exceeds 1.5 (danger zone for injury risk)"
            )

    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings
    }


def validate_inference_dataframe(df: pd.DataFrame, strict: bool = True) -> Dict:
    """
    Validate a DataFrame for batch inference.

    Validates all rows and provides detailed error reporting.

    Args:
        df: DataFrame containing player data
        strict: If True, any validation error fails the entire batch.
                If False, only logs errors but allows inference to proceed.

    Returns:
        Dictionary with validation results:
        {
            'valid': bool,
            'total_rows': int,
            'valid_rows': int,
            'invalid_rows': int,
            'row_errors': dict mapping row index to list of errors,
            'row_warnings': dict mapping row index to list of warnings,
            'summary': str
        }

    Example:
        >>> result = validate_inference_dataframe(df, strict=True)
        >>> if not result['valid']:
        ...     print(result['summary'])
        ...     for idx, errors in result['row_errors'].items():
        ...         print(f"Row {idx}: {errors}")
    """
    row_errors = {}
    row_warnings = {}

    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        validation_result = validate_player_input(row_dict)

        if not validation_result['valid']:
            row_errors[idx] = validation_result['errors']

        if validation_result['warnings']:
            row_warnings[idx] = validation_result['warnings']

    valid_rows = len(df) - len(row_errors)
    invalid_rows = len(row_errors)

    # Generate summary
    summary_lines = [
        f"Validation Results:",
        f"  Total rows: {len(df)}",
        f"  Valid rows: {valid_rows}",
        f"  Invalid rows: {invalid_rows}",
        f"  Rows with warnings: {len(row_warnings)}"
    ]

    if row_errors:
        summary_lines.append(f"\nFirst 5 error examples:")
        for idx, errors in list(row_errors.items())[:5]:
            name = df.loc[idx, 'name'] if 'name' in df.columns else f"Row {idx}"
            summary_lines.append(f"  {name}:")
            for error in errors[:3]:  # Show first 3 errors per row
                summary_lines.append(f"    - {error}")

    summary = "\n".join(summary_lines)

    is_valid = (len(row_errors) == 0) if strict else True

    return {
        'valid': is_valid,
        'total_rows': len(df),
        'valid_rows': valid_rows,
        'invalid_rows': invalid_rows,
        'row_errors': row_errors,
        'row_warnings': row_warnings,
        'summary': summary
    }


def validate_and_raise(player_data: Dict, context: str = "inference") -> None:
    """
    Validate player input and raise exception if invalid.

    This is a convenience function for code that needs to fail fast on invalid data.

    Args:
        player_data: Dictionary containing player features
        context: Context string for error message (e.g., "inference", "training")

    Raises:
        ValidationError: If validation fails

    Example:
        >>> try:
        ...     validate_and_raise(player_data, context="API request")
        ... except ValidationError as e:
        ...     print(f"Validation failed: {e}")
    """
    result = validate_player_input(player_data)

    if not result['valid']:
        error_msg = f"Validation failed for {context}:\n"
        error_msg += "\n".join(f"  - {e}" for e in result['errors'])

        if result['warnings']:
            error_msg += "\n\nWarnings:\n"
            error_msg += "\n".join(f"  - {w}" for w in result['warnings'])

        raise ValidationError(error_msg)


def validate_dataframe_and_raise(df: pd.DataFrame, context: str = "inference") -> None:
    """
    Validate DataFrame and raise exception if invalid.

    Args:
        df: DataFrame containing player data
        context: Context string for error message

    Raises:
        ValidationError: If validation fails
    """
    result = validate_inference_dataframe(df, strict=True)

    if not result['valid']:
        error_msg = f"Batch validation failed for {context}:\n"
        error_msg += result['summary']
        raise ValidationError(error_msg)
