"""
Test script to demonstrate validation functionality.

Run this to verify validation is working correctly.
"""

import pandas as pd
import numpy as np
from validation import (
    validate_player_input,
    validate_inference_dataframe,
    validate_age,
    validate_position,
    validate_fifa_rating,
    validate_workload_metric
)


def test_age_validation():
    """Test age validation with various inputs."""
    print("\n" + "="*60)
    print("Testing Age Validation")
    print("="*60)

    test_cases = [
        (25, True, "Valid age"),
        (16, True, "Minimum age"),
        (45, True, "Maximum age"),
        (15, False, "Below minimum"),
        (50, False, "Above maximum"),
        (None, False, "Null age"),
        ("abc", False, "Non-numeric"),
    ]

    for age, expected_valid, description in test_cases:
        is_valid, error_msg = validate_age(age)
        status = "✓ PASS" if is_valid == expected_valid else "✗ FAIL"
        print(f"{status} | Age {age}: {description}")
        if error_msg:
            print(f"       Error: {error_msg}")


def test_position_validation():
    """Test position validation with various inputs."""
    print("\n" + "="*60)
    print("Testing Position Validation")
    print("="*60)

    test_cases = [
        ("GK", True, "Valid goalkeeper"),
        ("MF", True, "Valid midfielder"),
        ("DF", True, "Valid defender"),
        ("FW", True, "Valid forward"),
        ("DF,MF", True, "Valid combined position"),
        ("INVALID", False, "Invalid position"),
        (None, False, "Null position"),
        ("", False, "Empty string"),
    ]

    for position, expected_valid, description in test_cases:
        is_valid, error_msg = validate_position(position)
        status = "✓ PASS" if is_valid == expected_valid else "✗ FAIL"
        print(f"{status} | Position '{position}': {description}")
        if error_msg:
            print(f"       Error: {error_msg}")


def test_fifa_rating_validation():
    """Test FIFA rating validation."""
    print("\n" + "="*60)
    print("Testing FIFA Rating Validation")
    print("="*60)

    test_cases = [
        (85, True, "Valid rating"),
        (0, True, "Minimum rating"),
        (99, True, "Maximum rating"),
        (None, True, "Null (allowed)"),
        (100, False, "Above maximum"),
        (-1, False, "Negative rating"),
    ]

    for rating, expected_valid, description in test_cases:
        is_valid, error_msg = validate_fifa_rating(rating)
        status = "✓ PASS" if is_valid == expected_valid else "✗ FAIL"
        print(f"{status} | FIFA Rating {rating}: {description}")
        if error_msg:
            print(f"       Error: {error_msg}")


def test_workload_validation():
    """Test workload metric validation."""
    print("\n" + "="*60)
    print("Testing Workload Metric Validation")
    print("="*60)

    test_cases = [
        ("matches_last_7", 3, True, "Valid match count"),
        ("matches_last_7", 8, False, "Exceeds maximum"),
        ("matches_last_7", -1, False, "Negative value"),
        ("acwr", 1.2, True, "Valid ACWR"),
        ("acwr", 10, False, "ACWR too high"),
        ("acute_load", 5, True, "Valid acute load"),
        ("rest_days_before_injury", 3, True, "Valid rest days"),
    ]

    for metric_name, value, expected_valid, description in test_cases:
        is_valid, error_msg = validate_workload_metric(metric_name, value)
        status = "✓ PASS" if is_valid == expected_valid else "✗ FAIL"
        print(f"{status} | {metric_name}={value}: {description}")
        if error_msg:
            print(f"       Error: {error_msg}")


def test_player_validation():
    """Test full player input validation."""
    print("\n" + "="*60)
    print("Testing Complete Player Validation")
    print("="*60)

    # Valid player
    valid_player = {
        "name": "Test Player",
        "player_team": "Test FC",
        "position": "MF",
        "age": 25,
        "fifa_rating": 85,
        "matches_last_7": 2,
        "matches_last_14": 4,
        "matches_last_30": 8,
        "rest_days_before_injury": 3,
        "avg_rest_last_5": 3.5,
        "goals_for_last_5": 7,
        "goals_against_last_5": 3,
        "goal_diff_last_5": 4,
        "avg_goal_diff_last_5": 0.8,
        "form_last_5": 9,
        "form_avg_last_5": 1.8,
        "win_ratio_last_5": 0.6,
        "win_streak": 2,
        "loss_streak": 0,
        "previous_injuries": 1,
        "days_since_last_injury": 180,
        "acute_load": 5,
        "chronic_load": 20,
        "acwr": 1.25,
        "monotony": 2.5,
        "strain": 12.5,
        "fatigue_index": -15,
        "workload_slope": 0.5,
        "spike_flag": 0,
    }

    print("\n1. Testing VALID player data:")
    result = validate_player_input(valid_player)
    print(f"   Valid: {result['valid']}")
    if result['errors']:
        print(f"   Errors: {result['errors']}")
    if result['warnings']:
        print(f"   Warnings: {result['warnings']}")

    # Invalid age
    invalid_age = valid_player.copy()
    invalid_age["age"] = 50
    print("\n2. Testing INVALID age (50):")
    result = validate_player_input(invalid_age)
    print(f"   Valid: {result['valid']}")
    print(f"   Errors: {result['errors']}")

    # Invalid position
    invalid_position = valid_player.copy()
    invalid_position["position"] = "INVALID_POS"
    print("\n3. Testing INVALID position:")
    result = validate_player_input(invalid_position)
    print(f"   Valid: {result['valid']}")
    print(f"   Errors: {result['errors']}")

    # Missing features
    incomplete_player = {
        "name": "Incomplete Player",
        "age": 25,
        "position": "FW"
    }
    print("\n4. Testing MISSING required features:")
    result = validate_player_input(incomplete_player)
    print(f"   Valid: {result['valid']}")
    print(f"   Errors (first 2): {result['errors'][:2]}")


def test_dataframe_validation():
    """Test DataFrame batch validation."""
    print("\n" + "="*60)
    print("Testing DataFrame Batch Validation")
    print("="*60)

    # Create test DataFrame
    data = []

    # Add 3 valid players
    for i in range(3):
        data.append({
            "name": f"Valid Player {i+1}",
            "player_team": "Test FC",
            "position": "MF",
            "age": 25 + i,
            "fifa_rating": 80 + i,
            "matches_last_7": 2,
            "matches_last_14": 4,
            "matches_last_30": 8,
            "rest_days_before_injury": 3,
            "avg_rest_last_5": 3.5,
            "goals_for_last_5": 7,
            "goals_against_last_5": 3,
            "goal_diff_last_5": 4,
            "avg_goal_diff_last_5": 0.8,
            "form_last_5": 9,
            "form_avg_last_5": 1.8,
            "win_ratio_last_5": 0.6,
            "win_streak": 2,
            "loss_streak": 0,
            "previous_injuries": 1,
            "days_since_last_injury": 180,
            "acute_load": 5,
            "chronic_load": 20,
            "acwr": 1.25,
            "monotony": 2.5,
            "strain": 12.5,
            "fatigue_index": -15,
            "workload_slope": 0.5,
            "spike_flag": 0,
        })

    # Add 2 invalid players
    invalid_player_1 = data[0].copy()
    invalid_player_1["name"] = "Invalid Player 1"
    invalid_player_1["age"] = 50  # Too old
    data.append(invalid_player_1)

    invalid_player_2 = data[0].copy()
    invalid_player_2["name"] = "Invalid Player 2"
    invalid_player_2["position"] = "INVALID"  # Invalid position
    data.append(invalid_player_2)

    df = pd.DataFrame(data)

    print(f"\nValidating DataFrame with {len(df)} rows...")
    result = validate_inference_dataframe(df, strict=False)

    print(f"\n{result['summary']}")

    print("\nDetailed errors:")
    for idx, errors in result['row_errors'].items():
        print(f"  Row {idx} ({df.loc[idx, 'name']}): {errors[0]}")


def run_all_tests():
    """Run all validation tests."""
    print("\n" + "="*60)
    print("VALIDATION MODULE TEST SUITE")
    print("="*60)

    test_age_validation()
    test_position_validation()
    test_fifa_rating_validation()
    test_workload_validation()
    test_player_validation()
    test_dataframe_validation()

    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)


if __name__ == "__main__":
    run_all_tests()
