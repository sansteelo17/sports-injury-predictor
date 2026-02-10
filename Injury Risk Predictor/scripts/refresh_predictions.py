#!/usr/bin/env python3
"""
Refresh predictions with current data.

Two modes available:
1. FBref (default): Scrapes actual player match logs for accurate workload
2. API: Uses football-data.org for team-level workload (less accurate, faster)

Usage:
    # Default: FBref scraper (accurate, ~20 min for all players)
    python scripts/refresh_predictions.py

    # Quick mode: API-based (team-level, ~2 min)
    python scripts/refresh_predictions.py --mode api --api-key YOUR_KEY

    # Preview without saving
    python scripts/refresh_predictions.py --dry-run

    # Specific players only
    python scripts/refresh_predictions.py --players "Bukayo Saka,Martin Odegaard"

Get API key at: https://www.football-data.org/client/register
"""

import os
import sys
import argparse
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.logger import get_logger
from src.utils.model_io import load_artifacts, save_artifacts

logger = get_logger(__name__)


# =============================================================================
# FBREF MODE: Scrape actual player match logs
# =============================================================================

def refresh_with_fbref(artifacts, dry_run=False, player_filter=None):
    """
    Refresh predictions using FBref scraper for accurate player workload.

    Args:
        artifacts: Loaded model artifacts
        dry_run: If True, don't save results
        player_filter: Optional list of player names to filter to
    """
    from src.data_loaders.fbref_scraper import FBrefScraper, fetch_all_player_workloads

    print("\n2. Scraping FBref for player match logs...")
    print("   (This takes ~15-20 min for all players, uses 24h cache)")

    scraper = FBrefScraper(cache_hours=24)

    # Get all players
    players_df = scraper.get_all_premier_league_players()
    print(f"   Found {len(players_df)} players from {players_df['team'].nunique()} teams")

    # Filter if requested
    if player_filter:
        players_df = players_df[players_df["name"].isin(player_filter)]
        print(f"   Filtered to {len(players_df)} requested players")

    # Calculate workload for each player
    print("\n3. Calculating player workloads from match logs...")

    workloads = []
    total = len(players_df)

    for i, (_, player) in enumerate(players_df.iterrows()):
        if i % 50 == 0 and i > 0:
            print(f"   Progress: {i}/{total} players ({i*100//total}%)")

        try:
            workload = scraper.get_player_workload(player["player_url"])
            workload["name"] = player["name"]
            workload["team"] = player["team"]
            workload["position"] = player["position"]
            workload["age"] = player["age"]
            workload["player_team"] = player["team"]  # For compatibility
            workloads.append(workload)
        except Exception as e:
            logger.warning(f"Failed for {player['name']}: {e}")
            workloads.append({
                "name": player["name"],
                "team": player["team"],
                "player_team": player["team"],
                "position": player["position"],
                "age": player["age"],
                "acute_load": 0,
                "chronic_load": 0,
                "acwr": 1.0,
                "monotony": 1.0,
                "strain": 0,
                "fatigue_index": 0,
                "workload_slope": 0,
                "spike_flag": 0,
                "matches_last_7": 0,
                "matches_last_14": 0,
                "matches_last_30": 0,
            })

    snapshots_df = pd.DataFrame(workloads)
    snapshots_df["snapshot_date"] = datetime.now().strftime("%Y-%m-%d")
    snapshots_df["injury_datetime"] = datetime.now()

    return snapshots_df


# =============================================================================
# API MODE: Team-level workload (less accurate but faster)
# =============================================================================

def compute_team_workload(matches_df, team, as_of_date):
    """
    Compute team-level workload metrics.

    Returns all workload features expected by the model:
    - acute_load, chronic_load, acwr (basic)
    - monotony, strain, fatigue_index (advanced)
    - workload_slope, spike_flag (trend indicators)
    """
    team_matches = matches_df[
        ((matches_df["Home"] == team) | (matches_df["Away"] == team)) &
        (matches_df["Date"] < as_of_date)
    ].copy()

    team_matches = team_matches.sort_values("Date")

    if len(team_matches) == 0:
        return {
            "acute_load": 0, "chronic_load": 0, "acwr": 1.0,
            "monotony": 1.0, "strain": 0, "fatigue_index": 0,
            "workload_slope": 0, "spike_flag": 0,
            "matches_last_7": 0, "matches_last_14": 0, "matches_last_30": 0,
        }

    date_7 = as_of_date - timedelta(days=7)
    date_14 = as_of_date - timedelta(days=14)
    date_28 = as_of_date - timedelta(days=28)
    date_30 = as_of_date - timedelta(days=30)

    matches_7 = len(team_matches[team_matches["Date"] >= date_7])
    matches_14 = len(team_matches[team_matches["Date"] >= date_14])
    matches_28 = len(team_matches[team_matches["Date"] >= date_28])
    matches_30 = len(team_matches[team_matches["Date"] >= date_30])

    acute_load = matches_7
    chronic_load = matches_28 / 4

    if chronic_load > 0:
        acwr = acute_load / chronic_load
    else:
        acwr = 1.0 if acute_load == 0 else 2.0

    # Monotony: 14-day load regularity
    daily_loads = []
    for i in range(14):
        day = as_of_date - timedelta(days=i)
        day_matches = len(team_matches[team_matches["Date"].dt.date == day.date()])
        daily_loads.append(day_matches)

    mean_load = np.mean(daily_loads)
    std_load = np.std(daily_loads)
    if std_load < 0.1:
        monotony = min(mean_load / 0.1, 5.0) if mean_load > 0 else 1.0
    else:
        monotony = min(mean_load / std_load, 5.0)

    strain = acute_load * monotony
    fatigue_index = acute_load - chronic_load
    spike_flag = 1 if acwr > 1.5 else 0

    # Workload slope: trend over last 5 matches
    recent_5 = team_matches.tail(5)
    if len(recent_5) >= 2:
        # Use match count per week as proxy
        loads = [1] * len(recent_5)  # Each match = 1 load unit
        x = np.arange(len(loads))
        workload_slope = np.polyfit(x, loads, 1)[0]
    else:
        workload_slope = 0

    return {
        "acute_load": acute_load, "chronic_load": chronic_load,
        "acwr": round(acwr, 3), "monotony": round(monotony, 3),
        "strain": round(strain, 3), "fatigue_index": round(fatigue_index, 3),
        "workload_slope": round(workload_slope, 4), "spike_flag": spike_flag,
        "matches_last_7": matches_7, "matches_last_14": matches_14,
        "matches_last_30": matches_30,
    }


def load_player_minutes_lookup():
    """
    Load player playing time data from historical stats CSV.

    Returns a dict mapping player name -> playing time ratio (0-1).
    Uses Min% from the most recent season available.
    """
    stats_path = os.path.join(PROJECT_ROOT, "data", "raw", "All_Players_1992-2025.csv")

    if not os.path.exists(stats_path):
        logger.warning(f"Player stats not found at {stats_path}")
        return {}

    df = pd.read_csv(stats_path)

    # Get most recent season for each player
    df = df.sort_values("Season", ascending=False)
    latest = df.groupby("Player").first().reset_index()

    # Create lookup: player name -> playing time ratio (Min% / 100)
    lookup = {}
    for _, row in latest.iterrows():
        if pd.notna(row.get("Min%")):
            lookup[row["Player"]] = row["Min%"] / 100.0

    logger.info(f"Loaded playing time data for {len(lookup)} players")
    return lookup


def load_injury_history_lookup():
    """
    Load real injury history from raw injury CSV.

    Returns a dict mapping player name -> {
        'previous_injuries': count,
        'days_since_last_injury': days since most recent injury,
        'last_injury_date': date of most recent injury
    }
    """
    injury_path = os.path.join(PROJECT_ROOT, "data", "raw", "player_injuries_impact.csv")

    if not os.path.exists(injury_path):
        logger.warning(f"Injury data not found at {injury_path}")
        return {}

    df = pd.read_csv(injury_path)

    # Parse dates
    date_col = None
    for col in ['Date of Injury', 'Date', 'date', 'Injury Date', 'injury_date']:
        if col in df.columns:
            date_col = col
            break

    if date_col is None:
        logger.warning(f"No date column found in injury data. Available: {df.columns.tolist()}")
        return {}

    df['injury_date'] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=['injury_date'])

    # Get player name column
    name_col = None
    for col in ['Name', 'Player Name', 'player_name', 'Player', 'name']:
        if col in df.columns:
            name_col = col
            break

    if name_col is None:
        logger.warning(f"No player name column found in injury data. Available: {df.columns.tolist()}")
        return {}

    # Calculate stats per player
    today = datetime.now()
    lookup = {}

    for player, group in df.groupby(name_col):
        injuries = group.sort_values('injury_date', ascending=False)
        last_injury = injuries.iloc[0]['injury_date']
        days_since = (today - last_injury).days

        lookup[player] = {
            'previous_injuries': len(injuries),
            'days_since_last_injury': max(0, days_since),
            'last_injury_date': last_injury.strftime('%Y-%m-%d')
        }

    logger.info(f"Loaded injury history for {len(lookup)} players from CSV")
    return lookup


def fetch_transfermarkt_injuries(player_names: list, existing_lookup: dict) -> dict:
    """
    Fetch injury history from Transfermarkt for players not in local CSV.

    Args:
        player_names: List of all player names
        existing_lookup: Dict of players already found in local CSV

    Returns:
        Updated lookup with Transfermarkt data merged in
    """
    from src.data_loaders.transfermarkt_scraper import TransfermarktScraper

    # Find players missing from local data
    missing = [name for name in player_names if name not in existing_lookup]

    if not missing:
        logger.info("All players found in local CSV, skipping Transfermarkt")
        return existing_lookup

    print(f"   Fetching {len(missing)} players from Transfermarkt...")
    scraper = TransfermarktScraper()

    lookup = existing_lookup.copy()
    found = 0
    errors = 0

    for i, name in enumerate(missing):
        if i % 20 == 0 and i > 0:
            print(f"   Progress: {i}/{len(missing)} ({found} found, {errors} not found)")

        try:
            data = scraper.fetch_player_injuries(name)
            if data and data["total_injuries"] > 0:
                lookup[name] = {
                    'previous_injuries': data["total_injuries"],
                    'days_since_last_injury': data["days_since_last"],
                    'last_injury_date': data["last_injury_date"],
                    'total_days_lost': data.get("total_days_out", 0)
                }
                found += 1
            elif data:
                # Player found but no injuries - still useful info
                lookup[name] = {
                    'previous_injuries': 0,
                    'days_since_last_injury': 365 * 3,  # 3 years for "never injured"
                    'last_injury_date': None,
                    'total_days_lost': 0
                }
                found += 1
        except Exception as e:
            logger.debug(f"Failed to fetch {name}: {e}")
            errors += 1

    print(f"   Transfermarkt: found {found}/{len(missing)} players")
    logger.info(f"Transfermarkt lookup complete: {found} found, {errors} errors")

    return lookup


def refresh_with_api(artifacts, api_key, dry_run=False):
    """
    Refresh predictions using football-data.org API.

    Uses team schedule + player's historical playing time ratio to estimate
    individual workload. More accurate than pure team-level approximation.
    """
    from src.data_loaders.api_client import FootballDataClient

    print("\n2. Connecting to football-data.org API...")
    client = FootballDataClient(api_key)
    print("   Connected!")

    # Load player minutes lookup for scaling
    print("\n3. Loading player playing time data...")
    minutes_lookup = load_player_minutes_lookup()
    print(f"   Loaded playing time ratios for {len(minutes_lookup)} players")

    # Fetch matches
    print("\n4. Fetching current season matches...")
    now = datetime.now()
    season = now.year if now.month >= 8 else now.year - 1
    matches = client.get_premier_league_matches(season=season)
    print(f"   Fetched {len(matches)} matches from {season}-{season+1} season")

    # Fetch squads
    print("\n5. Fetching current Premier League squads...")
    players = client.get_all_team_squads()
    print(f"   Fetched {len(players)} players from {players['team'].nunique()} teams")

    # Build snapshots with player-scaled workload
    print("\n6. Computing player workloads (team schedule × playing time)...")
    snapshot_date = datetime.now()
    rows = []

    players_with_ratio = 0
    for _, player in players.iterrows():
        # Get team-level workload
        team_workload = compute_team_workload(matches, player["team"], snapshot_date)

        # Get player's playing time ratio (0-1), default to 0.5 if unknown
        player_name = player["name"]
        play_ratio = minutes_lookup.get(player_name, 0.5)

        # Scale workload metrics by playing time ratio
        # A player who plays 50% of minutes has ~50% of the team's workload
        scaled_workload = {
            "acute_load": round(team_workload["acute_load"] * play_ratio, 2),
            "chronic_load": round(team_workload["chronic_load"] * play_ratio, 2),
            "acwr": team_workload["acwr"],  # Ratio stays the same
            "monotony": team_workload["monotony"],  # Schedule-based, same for team
            "strain": round(team_workload["strain"] * play_ratio, 2),
            "fatigue_index": round(team_workload["fatigue_index"] * play_ratio, 2),
            "workload_slope": team_workload["workload_slope"],
            "spike_flag": team_workload["spike_flag"],
            "matches_last_7": round(team_workload["matches_last_7"] * play_ratio, 1),
            "matches_last_14": round(team_workload["matches_last_14"] * play_ratio, 1),
            "matches_last_30": round(team_workload["matches_last_30"] * play_ratio, 1),
        }

        if player_name in minutes_lookup:
            players_with_ratio += 1

        rows.append({
            "name": player_name,
            "player_team": player["team"],
            "team": player["team"],
            "position": player.get("position", "Unknown"),
            "age": player.get("age", 25),
            "injury_datetime": snapshot_date,
            "snapshot_date": snapshot_date.strftime("%Y-%m-%d"),
            "playing_time_ratio": play_ratio,
            **scaled_workload,
        })

    print(f"   {players_with_ratio}/{len(players)} players matched with playing time data")
    snapshots_df = pd.DataFrame(rows)
    return snapshots_df


# =============================================================================
# INFERENCE
# =============================================================================

def assign_archetype_heuristic(row):
    """
    Assign injury-focused archetype based on injury patterns, not raw counts.

    Key insight: Total injuries means nothing. What matters is:
    - Severity (avg days per injury)
    - Recency (how recently injured)
    - Impact (total days lost)

    Archetypes:
    - Durable: Rarely injured, quick recovery when it happens
    - Fragile: When injured, out for long periods (high severity)
    - Recurring: Frequent minor injuries, often returns quickly
    - Currently Vulnerable: Recently injured or returning
    - Moderate Risk: Average injury profile
    - Clean Record: No significant injury history
    """
    prev_injuries = row.get("previous_injuries", 0)
    days_since = row.get("days_since_last_injury", 365)
    total_days_lost = row.get("total_days_lost", 0)
    age = row.get("age", 25)

    # Calculate severity metrics
    avg_days_per_injury = total_days_lost / prev_injuries if prev_injuries > 0 else 0

    # Priority 1: Currently vulnerable (recently injured or returning)
    if days_since < 60:
        return "Currently Vulnerable"

    # Priority 2: No injury history = clean record
    if prev_injuries == 0:
        return "Clean Record"

    # Priority 3: Classify by injury SEVERITY (avg days per injury)
    # Low severity (<20 days avg) = minor injuries, quick recovery
    # High severity (>40 days avg) = serious injuries, long absences

    if avg_days_per_injury >= 40:
        # Severe injuries when they happen
        return "Fragile"

    if avg_days_per_injury >= 25:
        # Moderate severity injuries
        if prev_injuries >= 10:
            return "Injury Prone"
        else:
            return "Moderate Risk"

    if avg_days_per_injury < 20:
        # Quick recovery from injuries
        if prev_injuries >= 10 and days_since < 180:
            return "Recurring"  # Frequent but minor
        else:
            return "Durable"  # Minor issues, handles them well

    # Default: moderate risk
    return "Moderate Risk"


def run_inference(snapshots_df, ensemble, severity_clf, player_history, archetype_df, use_transfermarkt=False):
    """
    Run model inference on player snapshots.

    Applies full feature engineering pipeline to match training data format.

    Args:
        use_transfermarkt: If True, fetch injury history from Transfermarkt for
                          players not found in local CSV (slower but more complete)
    """
    from src.inference.inference_pipeline import apply_all_feature_engineering, add_player_history_features

    print("\n7. Running inference...")

    df = snapshots_df.copy()

    # Ensure required columns exist for feature engineering
    if "player_team" not in df.columns and "team" in df.columns:
        df["player_team"] = df["team"]

    # Apply full feature engineering (temporal, position, interactions)
    print("   Applying feature engineering...")
    df = apply_all_feature_engineering(
        df,
        date_column="injury_datetime",
        position_column="position",
        team_column="player_team"
    )

    # Load real injury history from CSV (not just the saved lookup)
    print("   Loading injury history from source data...")
    injury_history = load_injury_history_lookup()
    print(f"   Found injury records for {len(injury_history)} players in CSV")

    # Fetch from Transfermarkt for players not in local CSV
    player_names = df["name"].dropna().unique().tolist()
    if use_transfermarkt:
        injury_history = fetch_transfermarkt_injuries(player_names, injury_history)
        print(f"   Total injury records after Transfermarkt: {len(injury_history)}")

    # Merge real injury history
    matched_injuries = 0
    for idx, row in df.iterrows():
        player_name = row.get("name", "")
        if player_name in injury_history:
            hist = injury_history[player_name]
            df.at[idx, "previous_injuries"] = hist["previous_injuries"]
            df.at[idx, "days_since_last_injury"] = hist["days_since_last_injury"]
            matched_injuries += 1

    print(f"   Matched {matched_injuries}/{len(df)} players with injury history")

    # Merge player history features (player_avg_severity, is_injury_prone, etc.)
    if player_history is not None and len(player_history) > 0:
        df = add_player_history_features(df, player_history)

    # Set defaults for players without injury history
    if "previous_injuries" not in df.columns:
        df["previous_injuries"] = 0
    df["previous_injuries"] = df["previous_injuries"].fillna(0)

    if "days_since_last_injury" not in df.columns:
        df["days_since_last_injury"] = 365  # 1 year default for unknown
    df["days_since_last_injury"] = df["days_since_last_injury"].fillna(365)

    # Assign archetypes using heuristic (instead of just merging old assignments)
    print("   Assigning player archetypes...")
    df["archetype"] = df.apply(assign_archetype_heuristic, axis=1)

    # Get expected features from model
    feature_cols = ensemble.feature_names_
    missing_features = [c for c in feature_cols if c not in df.columns]

    # Set sensible defaults for known missing features (not just zeros)
    feature_defaults = {
        "fifa_rating": 70,  # Average rating
        "rest_days_before_injury": 4,  # Typical rest between matches
        "avg_rest_last_5": 4,
        "goals_for_last_5": 7,  # ~1.4 per match
        "goals_against_last_5": 5,
        "goal_diff_last_5": 2,
        "avg_goal_diff_last_5": 0.4,
        "form_last_5": 7,  # ~1.4 pts per match
        "form_avg_last_5": 1.4,
        "win_ratio_last_5": 0.4,
        "win_streak": 0,
        "loss_streak": 0,
    }

    if missing_features:
        print(f"   Warning: {len(missing_features)} missing features, filling with defaults")
        logger.warning(f"Missing features after engineering: {missing_features}")
        for col in missing_features:
            df[col] = feature_defaults.get(col, 0)

    X = df[feature_cols].fillna(0)

    # Ensemble predictions
    print("   Running ensemble predictions...")

    # Suppress sklearn feature name warnings (expected when ensemble converts to numpy)
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        probs = ensemble.predict_proba(X)
        df["ensemble_prob"] = probs[:, 1] if len(probs.shape) > 1 else probs

        # Individual base model predictions (using get_base_predictions method)
        base_preds = ensemble.get_base_predictions(X)
        df["lgb_prob"] = base_preds["lgb_prob"]
        df["xgb_prob"] = base_preds["xgb_prob"]
        df["catboost_prob"] = base_preds["catboost_prob"]

    # Agreement (how many models predict >50% risk)
    model_probs = np.column_stack([df["lgb_prob"], df["xgb_prob"], df["catboost_prob"]])
    df["agreement"] = (model_probs > 0.5).sum(axis=1)
    df["confidence"] = df["agreement"].map({3: "very-high", 2: "high", 1: "medium", 0: "low"})

    # Severity prediction using heuristic for live data
    # (Severity classifier requires 202 features including match-level data we don't have)
    # Heuristic: based on workload spike, age, and injury history
    def estimate_severity(row):
        """Estimate severity class based on available features."""
        score = 0

        # High workload spike → more severe injuries
        if row.get("spike_flag", 0) == 1:
            score += 1
        if row.get("acwr", 1.0) > 1.8:
            score += 1

        # Age factor (older players → longer recovery)
        age = row.get("age", 25)
        if age >= 32:
            score += 1
        elif age >= 28:
            score += 0.5

        # Injury history (injury-prone players → more severe)
        if row.get("is_injury_prone", 0) == 1:
            score += 1
        if row.get("previous_injuries", 0) >= 5:
            score += 0.5

        # Map score to severity class
        if score >= 2.5:
            return "long"
        elif score >= 1:
            return "medium"
        else:
            return "short"

    df["predicted_severity_class"] = df.apply(estimate_severity, axis=1)

    print(f"   Inference complete. Mean risk: {df['ensemble_prob'].mean():.1%}")
    return df


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Refresh predictions with live data")
    parser.add_argument("--mode", choices=["fbref", "api"], default="fbref",
                        help="Data source: 'fbref' (accurate) or 'api' (fast)")
    parser.add_argument("--api-key", help="Football-data.org API key (for api mode)")
    parser.add_argument("--dry-run", action="store_true", help="Don't save, just preview")
    parser.add_argument("--players", help="Comma-separated player names to filter to")
    parser.add_argument("--transfermarkt", action="store_true",
                        help="Fetch injury history from Transfermarkt for missing players (slower)")
    args = parser.parse_args()

    print("=" * 60)
    print("INJURY RISK PREDICTOR - Live Data Refresh")
    print("=" * 60)
    print(f"Mode: {args.mode.upper()}")

    # Load models
    print("\n1. Loading trained models...")
    artifacts = load_artifacts()
    if artifacts is None:
        print("ERROR: No trained models found. Run the notebook first.")
        sys.exit(1)

    ensemble = artifacts["ensemble"]
    severity_clf = artifacts["severity_clf"]
    player_history = artifacts["player_history"]
    archetype_df = artifacts["df_clusters"]
    print(f"   Loaded models with {len(player_history)} player histories")

    # Get player snapshots based on mode
    if args.mode == "fbref":
        player_filter = args.players.split(",") if args.players else None
        snapshots_df = refresh_with_fbref(artifacts, args.dry_run, player_filter)
    else:
        api_key = args.api_key or os.environ.get("FOOTBALL_DATA_API_KEY")
        if not api_key:
            print("ERROR: API key required for api mode.")
            print("Set FOOTBALL_DATA_API_KEY env var or use --api-key")
            sys.exit(1)
        snapshots_df = refresh_with_api(artifacts, api_key, args.dry_run)

    # Run inference
    inference_df = run_inference(
        snapshots_df, ensemble, severity_clf, player_history, archetype_df,
        use_transfermarkt=args.transfermarkt
    )

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    high_risk = inference_df[inference_df["ensemble_prob"] >= 0.6]
    medium_risk = inference_df[(inference_df["ensemble_prob"] >= 0.4) & (inference_df["ensemble_prob"] < 0.6)]
    low_risk = inference_df[inference_df["ensemble_prob"] < 0.4]

    print(f"\nRisk distribution ({len(inference_df)} players):")
    print(f"  High risk (>=60%):    {len(high_risk):3d} ({len(high_risk)/len(inference_df)*100:.1f}%)")
    print(f"  Medium risk (40-60%): {len(medium_risk):3d} ({len(medium_risk)/len(inference_df)*100:.1f}%)")
    print(f"  Low risk (<40%):      {len(low_risk):3d} ({len(low_risk)/len(inference_df)*100:.1f}%)")

    print(f"\nTop 10 highest risk players:")
    top_risk = inference_df.nlargest(10, "ensemble_prob")[["name", "player_team", "ensemble_prob", "acwr"]]
    for _, row in top_risk.iterrows():
        acwr_str = f"ACWR={row.get('acwr', 'N/A')}" if pd.notna(row.get('acwr')) else ""
        print(f"  {row['ensemble_prob']:.1%} - {row['name']} ({row.get('player_team', 'Unknown')}) {acwr_str}")

    # Save
    if args.dry_run:
        print("\n[DRY RUN - not saving]")
    else:
        print("\n7. Saving updated predictions...")
        save_artifacts(
            ensemble=ensemble,
            severity_clf=severity_clf,
            df_clusters=archetype_df,
            player_history=player_history,
            inference_df=inference_df,
        )
        print(f"   Saved {len(inference_df)} player predictions")

    print("\nDone! Run 'streamlit run app.py' to see live predictions.")


if __name__ == "__main__":
    main()
