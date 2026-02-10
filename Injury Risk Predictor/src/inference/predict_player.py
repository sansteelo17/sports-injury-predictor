"""
Predict injury risk for ANY player - the whole point of the model.

This module allows prediction for players NOT in the training data
by fetching their info and running through the trained model.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.model_io import load_artifacts
from src.data_loaders.transfermarkt_scraper import TransfermarktScraper


def predict_for_player(player_name: str, artifacts: dict = None) -> Optional[Dict]:
    """
    Predict injury risk for ANY player by name.

    This is the core inference function - works for players not in training data.

    Args:
        player_name: Full player name (e.g., "Martin Odegaard")
        artifacts: Pre-loaded model artifacts (optional, will load if not provided)

    Returns:
        Dict with prediction results or None if player not found
    """
    # Load models if not provided
    if artifacts is None:
        artifacts = load_artifacts()
        if artifacts is None:
            raise RuntimeError("No trained models found. Train the model first.")

    ensemble = artifacts["ensemble"]
    feature_cols = ensemble.feature_names_

    # Fetch player data from Transfermarkt
    scraper = TransfermarktScraper()
    player_data = scraper.fetch_player_injuries(player_name)

    if player_data is None:
        return None

    # Build feature vector for this player
    features = build_feature_vector(player_data, feature_cols)

    # Run prediction
    X = pd.DataFrame([features])[feature_cols].fillna(0)

    # Suppress sklearn warnings
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        prob = ensemble.predict_proba(X)[0, 1]
        base_preds = ensemble.get_base_predictions(X)

    # Assign archetype
    archetype = assign_archetype(player_data)

    # Determine risk level
    if prob >= 0.6:
        risk_level = "High"
    elif prob >= 0.4:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return {
        "name": player_data["name"],
        "team": player_data.get("team", "Unknown"),
        "risk_probability": prob,
        "risk_level": risk_level,
        "archetype": archetype,
        "previous_injuries": player_data["total_injuries"],
        "total_days_lost": player_data.get("total_days_out", 0),
        "days_since_last_injury": player_data["days_since_last"],
        "last_injury_date": player_data.get("last_injury_date"),
        "model_predictions": {
            "lgb": base_preds["lgb_prob"].iloc[0],
            "xgb": base_preds["xgb_prob"].iloc[0],
            "catboost": base_preds["catboost_prob"].iloc[0],
        },
        "recent_injuries": player_data.get("injuries", [])[:5],
    }


def build_feature_vector(player_data: dict, feature_cols: list) -> dict:
    """
    Build feature vector for a player from Transfermarkt data.

    Uses sensible defaults for features we can't fetch (like live workload).
    """
    injuries = player_data.get("total_injuries", 0)
    days_lost = player_data.get("total_days_out", 0)
    days_since = player_data.get("days_since_last", 365)

    # Calculate derived features
    avg_days_per_injury = days_lost / injuries if injuries > 0 else 0

    # Default feature values (reasonable averages)
    features = {
        # Injury history
        "previous_injuries": injuries,
        "days_since_last_injury": min(days_since, 730),  # Cap at 2 years

        # Workload features (use league averages as defaults)
        "acute_load": 0.5,
        "chronic_load": 0.5,
        "acwr": 1.0,
        "monotony": 1.5,
        "strain": 0.75,
        "fatigue_index": 0,
        "workload_slope": 0,
        "spike_flag": 0,

        # Match features
        "matches_last_7": 1,
        "matches_last_14": 2,
        "matches_last_30": 4,
        "rest_days_before_injury": 4,
        "avg_rest_last_5": 4,

        # Form features (neutral defaults)
        "goals_for_last_5": 7,
        "goals_against_last_5": 5,
        "goal_diff_last_5": 2,
        "avg_goal_diff_last_5": 0.4,
        "form_last_5": 7,
        "form_avg_last_5": 1.4,
        "win_ratio_last_5": 0.4,
        "win_streak": 0,
        "loss_streak": 0,

        # Player features
        "age": 26,  # Default age
        "fifa_rating": 75,

        # Derived injury features
        "is_injury_prone": 1 if injuries >= 10 and avg_days_per_injury >= 25 else 0,
        "player_avg_severity": avg_days_per_injury,
    }

    # Fill any missing features with 0
    for col in feature_cols:
        if col not in features:
            features[col] = 0

    return features


def assign_archetype(player_data: dict) -> str:
    """
    Assign injury archetype based on player's injury history.
    """
    injuries = player_data.get("total_injuries", 0)
    days_lost = player_data.get("total_days_out", 0)
    days_since = player_data.get("days_since_last", 365)

    if injuries == 0:
        return "Clean Record"

    avg_days = days_lost / injuries

    # Recently injured
    if days_since < 60:
        return "Currently Vulnerable"

    # High severity
    if avg_days >= 40:
        return "Fragile"

    # Moderate severity with many injuries
    if avg_days >= 25:
        if injuries >= 10:
            return "Injury Prone"
        return "Moderate Risk"

    # Low severity (quick recovery)
    if avg_days < 20:
        if injuries >= 10 and days_since < 180:
            return "Recurring"
        return "Durable"

    return "Moderate Risk"


def predict_batch(player_names: list, artifacts: dict = None) -> pd.DataFrame:
    """
    Predict for multiple players.

    Args:
        player_names: List of player names
        artifacts: Pre-loaded model artifacts

    Returns:
        DataFrame with predictions for all found players
    """
    if artifacts is None:
        artifacts = load_artifacts()

    results = []
    for name in player_names:
        try:
            pred = predict_for_player(name, artifacts)
            if pred:
                results.append(pred)
        except Exception as e:
            print(f"Failed for {name}: {e}")

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results)


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict injury risk for any player")
    parser.add_argument("player", help="Player name (e.g., 'Martin Odegaard')")
    args = parser.parse_args()

    print(f"\nPredicting for: {args.player}")
    print("=" * 50)

    result = predict_for_player(args.player)

    if result is None:
        print(f"Player '{args.player}' not found on Transfermarkt")
    else:
        print(f"Name: {result['name']}")
        print(f"Team: {result['team']}")
        print(f"Risk: {result['risk_level']} ({result['risk_probability']:.1%})")
        print(f"Archetype: {result['archetype']}")
        print(f"Injuries: {result['previous_injuries']} ({result['total_days_lost']} days lost)")
        print(f"Days since last injury: {result['days_since_last_injury']}")

        print(f"\nModel predictions:")
        for model, prob in result['model_predictions'].items():
            print(f"  {model}: {prob:.1%}")

        if result['recent_injuries']:
            print(f"\nRecent injuries:")
            for inj in result['recent_injuries'][:3]:
                print(f"  - {inj['date']}: {inj['injury']} ({inj['days_out']} days)")
