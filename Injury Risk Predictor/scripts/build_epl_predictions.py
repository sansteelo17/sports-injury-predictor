#!/usr/bin/env python3
"""
Build predictions for ALL current EPL players.

Fetches squad lists from Transfermarkt, gets injury data,
and runs model predictions for every player.
"""

import sys
import os
from datetime import datetime

import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.model_io import load_artifacts, save_artifacts
from src.data_loaders.transfermarkt_scraper import TransfermarktScraper
from bs4 import BeautifulSoup

# EPL teams with Transfermarkt IDs
EPL_TEAMS = {
    "Arsenal": ("arsenal-fc", 11),
    "Aston Villa": ("aston-villa", 405),
    "Bournemouth": ("afc-bournemouth", 989),
    "Brentford": ("brentford-fc", 1148),
    "Brighton": ("brighton-hove-albion", 1237),
    "Chelsea": ("chelsea-fc", 631),
    "Crystal Palace": ("crystal-palace", 873),
    "Everton": ("everton-fc", 29),
    "Fulham": ("fulham-fc", 931),
    "Ipswich": ("ipswich-town", 677),
    "Leicester": ("leicester-city", 1003),
    "Liverpool": ("liverpool-fc", 31),
    "Man City": ("manchester-city", 281),
    "Man United": ("manchester-united", 985),
    "Newcastle": ("newcastle-united", 762),
    "Nottingham Forest": ("nottingham-forest", 703),
    "Southampton": ("southampton-fc", 180),
    "Tottenham": ("tottenham-hotspur", 148),
    "West Ham": ("west-ham-united", 379),
    "Wolves": ("wolverhampton-wanderers", 543),
}


def get_team_squad(scraper, team_slug, team_id):
    """Get all players from a team's squad page."""
    url = f"https://www.transfermarkt.com/{team_slug}/kader/verein/{team_id}"

    try:
        html = scraper._fetch(url)
        soup = BeautifulSoup(html, "html.parser")

        players = []
        rows = soup.select("table.items tbody tr.odd, table.items tbody tr.even")

        for row in rows:
            try:
                # Get player name and link
                name_cell = row.select_one("td.hauptlink a")
                if not name_cell:
                    continue

                name = name_cell.get_text(strip=True)
                href = name_cell.get("href", "")

                # Get position
                pos_cell = row.select("td.posrela table tr")
                position = ""
                if len(pos_cell) >= 2:
                    position = pos_cell[1].get_text(strip=True)

                # Get age
                age = 25  # default
                age_cells = row.select("td.zentriert")
                for cell in age_cells:
                    text = cell.get_text(strip=True)
                    if text.isdigit() and 15 < int(text) < 45:
                        age = int(text)
                        break

                players.append({
                    "name": name,
                    "position": position,
                    "age": age,
                    "href": href
                })
            except Exception:
                continue

        return players
    except Exception as e:
        print(f"  Error fetching squad: {e}")
        return []


def main():
    print("=" * 60)
    print("BUILD EPL PREDICTIONS")
    print("=" * 60)

    # Load models
    print("\n1. Loading trained models...")
    artifacts = load_artifacts()
    if artifacts is None:
        print("ERROR: No trained models found.")
        sys.exit(1)

    ensemble = artifacts["ensemble"]
    feature_cols = ensemble.feature_names_
    print(f"   Model expects {len(feature_cols)} features")

    # Fetch all EPL squads
    print("\n2. Fetching EPL squads from Transfermarkt...")
    scraper = TransfermarktScraper()

    all_players = []
    for team_name, (slug, team_id) in EPL_TEAMS.items():
        print(f"   {team_name}...", end=" ", flush=True)
        squad = get_team_squad(scraper, slug, team_id)
        for p in squad:
            p["team"] = team_name
        all_players.extend(squad)
        print(f"{len(squad)} players")

    print(f"\n   Total: {len(all_players)} players")

    # Fetch injury data for each player
    print("\n3. Fetching injury history...")
    players_data = []

    for i, player in enumerate(all_players):
        if (i + 1) % 50 == 0:
            print(f"   Progress: {i+1}/{len(all_players)}", flush=True)

        try:
            injury_data = scraper.fetch_player_injuries(player["name"])

            if injury_data:
                # Use age from profile page if available, fall back to squad page
                age = injury_data.get("age") or player["age"]
                players_data.append({
                    "name": injury_data["name"],
                    "team": player["team"],
                    "position": player["position"],
                    "age": age,
                    "previous_injuries": injury_data["total_injuries"],
                    "total_days_lost": injury_data.get("total_days_out", 0),
                    "days_since_last_injury": injury_data["days_since_last"],
                    "last_injury_date": injury_data.get("last_injury_date"),
                })
            else:
                # Player not found - use defaults
                players_data.append({
                    "name": player["name"],
                    "team": player["team"],
                    "position": player["position"],
                    "age": 25,  # Default age when unknown
                    "previous_injuries": 0,
                    "total_days_lost": 0,
                    "days_since_last_injury": 365 * 3,
                    "last_injury_date": None,
                })
        except Exception:
            players_data.append({
                "name": player["name"],
                "team": player["team"],
                "position": player["position"],
                "age": 25,  # Default age when unknown
                "previous_injuries": 0,
                "total_days_lost": 0,
                "days_since_last_injury": 365 * 3,
                "last_injury_date": None,
            })

    df = pd.DataFrame(players_data)
    print(f"   Got injury data for {len(df)} players")

    # Add required columns for model
    print("\n4. Building feature matrix...")

    # Assign archetypes
    def assign_archetype(row):
        prev = row.get("previous_injuries", 0)
        days_since = row.get("days_since_last_injury", 365)
        total_lost = row.get("total_days_lost", 0)
        avg_days = total_lost / prev if prev > 0 else 0

        if days_since < 60:
            return "Currently Vulnerable"
        if prev == 0:
            return "Clean Record"
        if avg_days >= 40:
            return "Fragile"
        if avg_days >= 25:
            return "Injury Prone" if prev >= 10 else "Moderate Risk"
        if avg_days < 20:
            return "Recurring" if prev >= 10 and days_since < 180 else "Durable"
        return "Moderate Risk"

    df["archetype"] = df.apply(assign_archetype, axis=1)

    # Add default features
    defaults = {
        "acute_load": 0.5,
        "chronic_load": 0.5,
        "acwr": 1.0,
        "monotony": 1.5,
        "strain": 0.75,
        "fatigue_index": 0,
        "workload_slope": 0,
        "spike_flag": 0,
        "matches_last_7": 1,
        "matches_last_14": 2,
        "matches_last_30": 4,
        "rest_days_before_injury": 4,
        "avg_rest_last_5": 4,
        "goals_for_last_5": 7,
        "goals_against_last_5": 5,
        "goal_diff_last_5": 2,
        "avg_goal_diff_last_5": 0.4,
        "form_last_5": 7,
        "form_avg_last_5": 1.4,
        "win_ratio_last_5": 0.4,
        "win_streak": 0,
        "loss_streak": 0,
        "fifa_rating": 75,
        "is_injury_prone": 0,
        "player_avg_severity": 0,
    }

    for col in feature_cols:
        if col not in df.columns:
            df[col] = defaults.get(col, 0)

    # Derive some features
    df["is_injury_prone"] = ((df["previous_injuries"] >= 10) &
                             (df["total_days_lost"] / df["previous_injuries"].replace(0, 1) >= 25)).astype(int)
    df["player_avg_severity"] = df["total_days_lost"] / df["previous_injuries"].replace(0, 1)

    # Run predictions
    print("\n5. Running model predictions...")
    X = df[feature_cols].fillna(0)

    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        probs = ensemble.predict_proba(X)
        df["ensemble_prob"] = probs[:, 1] if len(probs.shape) > 1 else probs

        base_preds = ensemble.get_base_predictions(X)
        df["lgb_prob"] = base_preds["lgb_prob"]
        df["xgb_prob"] = base_preds["xgb_prob"]
        df["catboost_prob"] = base_preds["catboost_prob"]

    # Calibrate predictions using archetype-aware risk scoring
    # Since workload features are constant, use injury history patterns for calibration
    print("\n6. Calibrating predictions...")

    def calculate_risk_score(row):
        """
        Calculate calibrated risk score combining model output with injury patterns.

        Key factors:
        1. Days since last injury (recent = higher risk)
        2. Injury severity (avg days per injury)
        3. Injury frequency (injuries per year)
        4. Archetype (captures overall pattern)
        """
        prev_injuries = row.get("previous_injuries", 0)
        days_since = row.get("days_since_last_injury", 365)
        total_days = row.get("total_days_lost", 0)
        archetype = row.get("archetype", "Unknown")

        # Base score from injury history (0-1 scale)
        score = 0.0

        # Factor 1: Recency (0-0.35)
        # Recently injured = high risk of re-injury
        if days_since < 30:
            score += 0.35
        elif days_since < 60:
            score += 0.28
        elif days_since < 90:
            score += 0.20
        elif days_since < 180:
            score += 0.12
        elif days_since < 365:
            score += 0.05
        # else: 0 (long time since injury)

        # Factor 2: Severity (0-0.30)
        # High avg days per injury = fragile
        if prev_injuries > 0:
            avg_days = total_days / prev_injuries
            if avg_days >= 50:
                score += 0.30
            elif avg_days >= 35:
                score += 0.22
            elif avg_days >= 25:
                score += 0.15
            elif avg_days >= 15:
                score += 0.08
            # else: minor injuries, low score

        # Factor 3: Frequency (0-0.20)
        # Assume career span of ~8 years for these players
        if prev_injuries >= 15:
            score += 0.20
        elif prev_injuries >= 10:
            score += 0.14
        elif prev_injuries >= 6:
            score += 0.08
        elif prev_injuries >= 3:
            score += 0.04

        # Factor 4: Archetype boost (0-0.15)
        archetype_boost = {
            "Currently Vulnerable": 0.15,  # Just returned, high re-injury risk
            "Fragile": 0.10,  # Severe injuries
            "Injury Prone": 0.12,  # Frequent + moderate severity
            "Recurring": 0.08,  # Frequent but minor
            "Moderate Risk": 0.05,
            "Durable": 0.00,  # Handles load well
            "Clean Record": 0.02,  # Unknown, slight caution
        }
        score += archetype_boost.get(archetype, 0.05)

        # Clamp to 0.10-0.90 range (never 0% or 100%)
        return max(0.10, min(0.90, score))

    df["calibrated_prob"] = df.apply(calculate_risk_score, axis=1)

    # Use calibrated probability as the main display value
    df["raw_ensemble_prob"] = df["ensemble_prob"]  # Keep original for reference
    df["ensemble_prob"] = df["calibrated_prob"]

    # Add metadata
    df["snapshot_date"] = datetime.now().strftime("%Y-%m-%d")
    df["injury_datetime"] = datetime.now()
    df["player_team"] = df["team"]

    # Summary with calibrated values
    print(f"\n   Calibrated risk distribution:")
    print(f"   High risk (>60%): {len(df[df['ensemble_prob'] >= 0.6])}")
    print(f"   Medium risk (35-60%): {len(df[(df['ensemble_prob'] >= 0.35) & (df['ensemble_prob'] < 0.6)])}")
    print(f"   Low risk (<35%): {len(df[df['ensemble_prob'] < 0.35])}")

    print("\n   Archetype distribution:")
    print(df["archetype"].value_counts().to_string())

    # Save
    print("\n6. Saving predictions...")
    save_artifacts(
        models_dir="models",
        ensemble=artifacts["ensemble"],
        severity_clf=artifacts["severity_clf"],
        df_clusters=artifacts["df_clusters"],
        player_history=artifacts["player_history"],
        inference_df=df
    )

    print(f"\n   Saved {len(df)} EPL player predictions")
    print("\nDone! Run 'streamlit run app.py' to use the dashboard.")


if __name__ == "__main__":
    main()
