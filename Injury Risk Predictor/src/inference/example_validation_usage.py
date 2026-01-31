"""
Example usage of validation in the inference pipeline.

This demonstrates how to use validation in real-world scenarios.
"""

import pandas as pd
import numpy as np
from validation import (
    validate_player_input,
    validate_inference_dataframe,
    validate_and_raise,
    ValidationError
)


def example_1_single_player_validation():
    """Example 1: Validate a single player's data before inference."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Single Player Validation")
    print("="*70)

    player_data = {
        "name": "Mohamed Salah",
        "player_team": "Liverpool",
        "position": "FW",
        "age": 29,
        "fifa_rating": 90,
        "matches_last_7": 2,
        "matches_last_14": 4,
        "matches_last_30": 8,
        "rest_days_before_injury": 3,
        "avg_rest_last_5": 3.2,
        "goals_for_last_5": 12,
        "goals_against_last_5": 4,
        "goal_diff_last_5": 8,
        "avg_goal_diff_last_5": 1.6,
        "form_last_5": 12,
        "form_avg_last_5": 2.4,
        "win_ratio_last_5": 0.8,
        "win_streak": 3,
        "loss_streak": 0,
        "previous_injuries": 2,
        "days_since_last_injury": 120,
        "acute_load": 6.5,
        "chronic_load": 22.0,
        "acwr": 1.35,
        "monotony": 2.8,
        "strain": 18.2,
        "fatigue_index": -15.5,
        "workload_slope": 0.3,
        "spike_flag": 0,
    }

    print("\nValidating player data...")
    result = validate_player_input(player_data)

    if result['valid']:
        print("✓ Validation PASSED - Data is ready for inference")
        print(f"  Player: {player_data['name']}")
        print(f"  Position: {player_data['position']}")
        print(f"  Age: {player_data['age']}")
        print(f"  FIFA Rating: {player_data['fifa_rating']}")
    else:
        print("✗ Validation FAILED")
        for error in result['errors']:
            print(f"  ERROR: {error}")

    if result['warnings']:
        print("\nWarnings:")
        for warning in result['warnings']:
            print(f"  WARNING: {warning}")


def example_2_batch_validation():
    """Example 2: Validate a batch of players before inference."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Batch Validation")
    print("="*70)

    # Create sample data for multiple players
    players = []

    # Good data
    for i in range(3):
        players.append({
            "name": f"Player {i+1}",
            "player_team": "Test FC",
            "position": "MF",
            "age": 24 + i,
            "fifa_rating": 78 + i,
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

    # Add problematic data
    problematic_player = players[0].copy()
    problematic_player["name"] = "Overworked Player"
    problematic_player["acwr"] = 1.8  # Danger zone
    problematic_player["acute_load"] = 12
    problematic_player["chronic_load"] = 15
    problematic_player["spike_flag"] = 1
    players.append(problematic_player)

    df = pd.DataFrame(players)

    print(f"\nValidating {len(df)} players...")
    result = validate_inference_dataframe(df, strict=True)

    print(f"\nValidation Summary:")
    print(f"  Total Players: {result['total_rows']}")
    print(f"  Valid: {result['valid_rows']}")
    print(f"  Invalid: {result['invalid_rows']}")
    print(f"  With Warnings: {len(result['row_warnings'])}")

    if result['row_warnings']:
        print("\nWarnings:")
        for idx, warnings in result['row_warnings'].items():
            player_name = df.loc[idx, 'name']
            print(f"  {player_name}:")
            for warning in warnings:
                print(f"    - {warning}")

    if result['valid']:
        print("\n✓ All players passed validation - ready for inference")
    else:
        print("\n✗ Some players failed validation")
        for idx, errors in result['row_errors'].items():
            player_name = df.loc[idx, 'name']
            print(f"  {player_name}: {errors[0]}")


def example_3_handling_invalid_data():
    """Example 3: Handle invalid data gracefully."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Handling Invalid Data")
    print("="*70)

    invalid_player = {
        "name": "Invalid Player",
        "player_team": "Test FC",
        "position": "UNKNOWN",  # Invalid position
        "age": 50,  # Too old
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

    print("\nAttempting to validate invalid player data...")

    try:
        validate_and_raise(invalid_player, context="API Request")
        print("✓ Validation passed")
    except ValidationError as e:
        print("✗ Validation failed (as expected)")
        print("\nError Details:")
        print(str(e))

    print("\n--- How to fix ---")
    print("1. Check that position is one of: GK, DF, MF, FW (or combinations)")
    print("2. Ensure age is between 16 and 45")
    print("3. Verify all required features are present")


def example_4_filtering_invalid_rows():
    """Example 4: Filter out invalid rows and proceed with valid data."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Filtering Invalid Rows")
    print("="*70)

    # Create mixed data (some valid, some invalid)
    data = []

    # Add 5 valid players
    for i in range(5):
        data.append({
            "name": f"Valid Player {i+1}",
            "player_team": "Test FC",
            "position": "MF",
            "age": 23 + i,
            "fifa_rating": 75 + i,
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
    invalid_1 = data[0].copy()
    invalid_1["name"] = "Too Young Player"
    invalid_1["age"] = 15
    data.append(invalid_1)

    invalid_2 = data[0].copy()
    invalid_2["name"] = "Invalid Position Player"
    invalid_2["position"] = "INVALID"
    data.append(invalid_2)

    df = pd.DataFrame(data)

    print(f"\nOriginal dataset: {len(df)} players")

    # Validate
    result = validate_inference_dataframe(df, strict=False)

    print(f"\nValidation Results:")
    print(f"  Valid: {result['valid_rows']}")
    print(f"  Invalid: {result['invalid_rows']}")

    if result['invalid_rows'] > 0:
        print(f"\nInvalid players will be filtered out:")
        for idx in result['row_errors'].keys():
            print(f"  - {df.loc[idx, 'name']}: {result['row_errors'][idx][0]}")

        # Filter out invalid rows
        valid_indices = [i for i in df.index if i not in result['row_errors']]
        df_valid = df.loc[valid_indices].copy()

        print(f"\n✓ Proceeding with {len(df_valid)} valid players for inference")
        print("Valid players:")
        for name in df_valid['name'].values:
            print(f"  - {name}")


def example_5_api_integration():
    """Example 5: Validation in an API endpoint."""
    print("\n" + "="*70)
    print("EXAMPLE 5: API Integration Pattern")
    print("="*70)

    def predict_injury_risk_api(player_data):
        """
        Simulated API endpoint for injury risk prediction.

        Args:
            player_data: Dictionary with player features

        Returns:
            Dictionary with prediction results or error message
        """
        # Validate input
        validation_result = validate_player_input(player_data)

        if not validation_result['valid']:
            return {
                "success": False,
                "error": "Invalid input data",
                "validation_errors": validation_result['errors'],
                "validation_warnings": validation_result['warnings']
            }

        # If validation passes, proceed with inference (simulated here)
        # In real code: run model prediction
        return {
            "success": True,
            "player": player_data['name'],
            "risk_probability": 0.23,  # Simulated
            "confidence": "high",
            "validation_warnings": validation_result['warnings']
        }

    # Test with valid data
    print("\nTest 1: Valid player data")
    valid_player = {
        "name": "Kevin De Bruyne",
        "player_team": "Manchester City",
        "position": "MF",
        "age": 30,
        "fifa_rating": 91,
        "matches_last_7": 2,
        "matches_last_14": 5,
        "matches_last_30": 10,
        "rest_days_before_injury": 2,
        "avg_rest_last_5": 2.8,
        "goals_for_last_5": 14,
        "goals_against_last_5": 2,
        "goal_diff_last_5": 12,
        "avg_goal_diff_last_5": 2.4,
        "form_last_5": 13,
        "form_avg_last_5": 2.6,
        "win_ratio_last_5": 0.8,
        "win_streak": 4,
        "loss_streak": 0,
        "previous_injuries": 3,
        "days_since_last_injury": 90,
        "acute_load": 7.2,
        "chronic_load": 24.0,
        "acwr": 1.3,
        "monotony": 3.1,
        "strain": 22.3,
        "fatigue_index": -16.8,
        "workload_slope": 0.4,
        "spike_flag": 0,
    }

    response = predict_injury_risk_api(valid_player)
    print(f"Response: {response}")

    # Test with invalid data
    print("\n\nTest 2: Invalid player data (age too high)")
    invalid_player = valid_player.copy()
    invalid_player["age"] = 52

    response = predict_injury_risk_api(invalid_player)
    print(f"Response: {response}")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("VALIDATION USAGE EXAMPLES")
    print("="*70)

    example_1_single_player_validation()
    example_2_batch_validation()
    example_3_handling_invalid_data()
    example_4_filtering_invalid_rows()
    example_5_api_integration()

    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70)
    print("\nKey Takeaways:")
    print("1. Always validate input data before inference")
    print("2. Use strict=True in production for safety")
    print("3. Handle ValidationError gracefully with clear messages")
    print("4. Filter invalid rows in batch processing")
    print("5. Log warnings even when validation passes")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
