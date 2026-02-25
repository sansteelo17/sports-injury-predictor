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
from pathlib import Path

import pandas as pd

# Load .env if it exists (so API keys don't need to be passed on command line)
_env_path = Path(__file__).resolve().parents[1] / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _val = _line.split("=", 1)
                os.environ.setdefault(_key.strip(), _val.strip())
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

# Normalize team names between squad API and match API
_TEAM_NAME_MAP = {
    "Man City": "Manchester City",
    "Man United": "Manchester United",
    "Brighton Hove": "Brighton",
    "Leeds United": "Leeds",
    "Nottingham": "Nottingham Forest",
}


def _normalize_match_team(team: str) -> str:
    """Map squad team name to match data team name."""
    return _TEAM_NAME_MAP.get(team, team)


def compute_team_workload(matches_df, team, as_of_date):
    """
    Compute team-level workload metrics.

    Returns all workload features expected by the model:
    - acute_load, chronic_load, acwr (basic)
    - monotony, strain, fatigue_index (advanced)
    - workload_slope, spike_flag (trend indicators)
    """
    team = _normalize_match_team(team)
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

    # Workload slope: trend in match density over last 5 weeks
    # Positive = schedule getting busier, negative = easing off
    if len(team_matches) >= 3:
        weekly_loads = []
        for w in range(5):
            week_start = as_of_date - timedelta(days=7 * (w + 1))
            week_end = as_of_date - timedelta(days=7 * w)
            week_count = len(team_matches[
                (team_matches["Date"] >= week_start) &
                (team_matches["Date"] < week_end)
            ])
            weekly_loads.append(week_count)
        # Reverse so index 0 = oldest week, 4 = most recent
        weekly_loads = weekly_loads[::-1]
        x = np.arange(len(weekly_loads))
        workload_slope = np.polyfit(x, weekly_loads, 1)[0]
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
    Load player playing time ratios from FPL API (live minutes data).

    FPL provides actual minutes played this season for every PL player.
    Converts to a ratio: minutes / max_possible_minutes.

    Returns a dict mapping player name -> playing time ratio (0-1).
    """
    try:
        from src.data_loaders.fpl_api import FPLClient
        client = FPLClient()
        all_stats = client.get_all_player_stats()

        # Current GW determines max possible minutes
        current_gw = client.get_current_gameweek()
        gw_num = current_gw.get("id", 20) if current_gw else 20
        max_minutes = gw_num * 90  # Maximum possible minutes so far

        lookup = {}
        for p in all_stats:
            minutes = p.get("minutes", 0)
            if max_minutes > 0:
                ratio = min(minutes / max_minutes, 1.0)
            else:
                ratio = 0.5

            # Index by multiple name formats for matching
            name = p.get("name", "")
            full_name = p.get("full_name", "")
            if name:
                lookup[name] = ratio
            if full_name:
                lookup[full_name] = ratio

        logger.info(f"Loaded playing time ratios for {len(all_stats)} players from FPL API (GW{gw_num})")
        return lookup
    except Exception as e:
        logger.warning(f"Failed to load FPL minutes: {e}")
        return {}


def load_injury_history_lookup():
    """
    Load real injury history from scraped Transfermarkt data.

    CSVs are for training only — inference uses scraped pkl files exclusively.

    Sources (in priority order):
    1. player_injuries_detail.pkl — per-injury records with dates (best)
    2. player_history.pkl — summary stats (fallback)

    Returns a dict mapping player name -> {
        'previous_injuries': count,
        'days_since_last_injury': days since most recent injury,
        'last_injury_date': date of most recent injury,
        'total_days_lost': total days lost to injury,
        'avg_severity': average days per injury
    }
    """
    today = datetime.now()
    lookup = {}

    # --- Source 1: Scraped detail pkl (per-injury records with dates) ---
    detail_path = os.path.join(PROJECT_ROOT, "models", "player_injuries_detail.pkl")
    if os.path.exists(detail_path):
        try:
            detail_df = pd.read_pickle(detail_path)
        except (NotImplementedError, Exception) as e:
            # StringDtype compatibility: re-save as plain object dtype
            logger.warning(f"Pickle compat issue, re-loading with fix: {e}")
            import pickle
            with open(detail_path, "rb") as f:
                detail_df = pickle.load(f)

        # Fix StringDtype columns → plain object dtype
        for col in detail_df.select_dtypes(include=["string"]).columns:
            detail_df[col] = detail_df[col].astype(object)
        if hasattr(detail_df['name'].dtype, 'name') and 'String' in str(detail_df['name'].dtype):
            detail_df['name'] = detail_df['name'].astype(str)

        if 'name' in detail_df.columns:
            detail_df['injury_datetime'] = pd.to_datetime(detail_df['injury_datetime'], errors='coerce')
            detail_df = detail_df.dropna(subset=['injury_datetime'])

            for player, group in detail_df.groupby('name'):
                injuries = group.sort_values('injury_datetime', ascending=False)
                last_injury = injuries.iloc[0]['injury_datetime']
                total_lost = int(injuries['severity_days'].sum()) if 'severity_days' in injuries.columns else 0
                avg_sev = float(injuries['severity_days'].mean()) if 'severity_days' in injuries.columns else 0

                lookup[player] = {
                    'previous_injuries': len(injuries),
                    'days_since_last_injury': max(0, (today - last_injury).days),
                    'last_injury_date': last_injury.strftime('%Y-%m-%d'),
                    'total_days_lost': total_lost,
                    'avg_severity': avg_sev,
                }
            logger.info(f"Loaded injury history for {len(lookup)} players from detail pkl ({len(detail_df)} records)")
            return lookup

    # --- Source 2: Player history pkl (summary stats, no per-injury dates) ---
    history_path = os.path.join(PROJECT_ROOT, "models", "player_history.pkl")
    if os.path.exists(history_path):
        try:
            history_df = pd.read_pickle(history_path)
        except (NotImplementedError, Exception):
            import pickle
            with open(history_path, "rb") as f:
                history_df = pickle.load(f)

        if hasattr(history_df, 'columns') and 'name' in history_df.columns:
            for _, row in history_df.iterrows():
                count = int(row.get('player_injury_count', 0) or 0)
                avg_sev = float(row.get('player_avg_severity', 0) or 0)
                days_since = row.get('days_since_last_injury')
                last_date = row.get('last_injury_date')

                # Estimate days_since if not populated
                if pd.isna(days_since) or days_since is None:
                    days_since = 180 if count > 0 else 365 * 3

                lookup[row['name']] = {
                    'previous_injuries': count,
                    'days_since_last_injury': int(days_since),
                    'last_injury_date': last_date if pd.notna(last_date) else None,
                    'total_days_lost': int(count * avg_sev),
                    'avg_severity': avg_sev,
                }
            logger.info(f"Loaded injury history for {len(lookup)} players from history pkl (summary only)")
            return lookup

    logger.warning("No scraped injury data found. Run: python scripts/scrape_injuries.py")
    return {}


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


def compute_team_form(matches_df, team, as_of_date, n_matches=5):
    """
    Compute team form features from recent match results.

    Returns dict with: goals_for_last_5, goals_against_last_5, goal_diff_last_5,
    avg_goal_diff_last_5, form_last_5, form_avg_last_5, win_ratio_last_5,
    win_streak, loss_streak, rest_days_before_injury, avg_rest_last_5.
    """
    team = _normalize_match_team(team)
    team_matches = matches_df[
        ((matches_df["Home"] == team) | (matches_df["Away"] == team)) &
        (matches_df["Date"] < as_of_date)
    ].sort_values("Date", ascending=False)

    if len(team_matches) == 0:
        return {
            "goals_for_last_5": 7, "goals_against_last_5": 5,
            "goal_diff_last_5": 2, "avg_goal_diff_last_5": 0.4,
            "form_last_5": 7, "form_avg_last_5": 1.4,
            "win_ratio_last_5": 0.4, "win_streak": 0, "loss_streak": 0,
            "rest_days_before_injury": 4, "avg_rest_last_5": 4,
        }

    recent = team_matches.head(n_matches)

    goals_for = 0
    goals_against = 0
    points = 0
    wins = 0
    results = []  # W/D/L sequence for streak calculation

    for _, match in recent.iterrows():
        is_home = match["Home"] == team
        gf = match["HomeGoals"] if is_home else match["AwayGoals"]
        ga = match["AwayGoals"] if is_home else match["HomeGoals"]
        goals_for += gf
        goals_against += ga

        if gf > ga:
            points += 3
            wins += 1
            results.append("W")
        elif gf == ga:
            points += 1
            results.append("D")
        else:
            results.append("L")

    n = len(recent)
    goal_diff = goals_for - goals_against

    # Streaks (from most recent)
    win_streak = 0
    for r in results:
        if r == "W":
            win_streak += 1
        else:
            break

    loss_streak = 0
    for r in results:
        if r == "L":
            loss_streak += 1
        else:
            break

    # Rest days: days since last match
    last_match_date = team_matches.iloc[0]["Date"]
    rest_days = max(1, (as_of_date - last_match_date).days)

    # Average rest between last 5 matches
    if n >= 2:
        dates = recent["Date"].tolist()
        gaps = [(dates[i] - dates[i + 1]).days for i in range(len(dates) - 1)]
        avg_rest = np.mean(gaps)
    else:
        avg_rest = 4

    return {
        "goals_for_last_5": int(goals_for),
        "goals_against_last_5": int(goals_against),
        "goal_diff_last_5": int(goal_diff),
        "avg_goal_diff_last_5": round(goal_diff / n, 2) if n > 0 else 0,
        "form_last_5": int(points),
        "form_avg_last_5": round(points / n, 2) if n > 0 else 0,
        "win_ratio_last_5": round(wins / n, 2) if n > 0 else 0,
        "win_streak": win_streak,
        "loss_streak": loss_streak,
        "rest_days_before_injury": round(rest_days, 1),
        "avg_rest_last_5": round(avg_rest, 1),
    }


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

    # Fetch matches (PL + all other competitions for accurate workload)
    print("\n4. Fetching current season matches (PL + cups/European)...")
    now = datetime.now()
    season = now.year if now.month >= 8 else now.year - 1
    matches = client.get_all_matches_for_pl_teams(season=season)
    if "competition" in matches.columns:
        comp_counts = matches["competition"].value_counts().to_dict()
        comp_str = ", ".join(f"{v} {k}" for k, v in comp_counts.items())
        print(f"   Fetched {len(matches)} matches ({comp_str}) from {season}-{season+1}")
    else:
        print(f"   Fetched {len(matches)} matches from {season}-{season+1}")

    # Fetch squads
    print("\n5. Fetching current Premier League squads...")
    players = client.get_all_team_squads()
    print(f"   Fetched {len(players)} players from {players['team'].nunique()} teams")

    # Build snapshots with player-scaled workload + team form
    print("\n6. Computing player workloads (team schedule × playing time)...")
    snapshot_date = datetime.now()
    rows = []

    # Cache team-level computations (same for all players on a team)
    team_workload_cache = {}
    team_form_cache = {}

    players_with_ratio = 0
    for _, player in players.iterrows():
        team = player["team"]

        # Get team-level workload (cached per team)
        if team not in team_workload_cache:
            team_workload_cache[team] = compute_team_workload(matches, team, snapshot_date)
            team_form_cache[team] = compute_team_form(matches, team, snapshot_date)

        team_workload = team_workload_cache[team]
        team_form = team_form_cache[team]

        # Get player's playing time ratio (0-1), default to 0.5 if unknown
        player_name = player["name"]
        play_ratio = minutes_lookup.get(player_name)
        # Try last name if full name didn't match
        if play_ratio is None and " " in player_name:
            last_name = player_name.split()[-1]
            play_ratio = minutes_lookup.get(last_name)
        if play_ratio is None:
            play_ratio = 0.5

        # Use team-level acute/chronic loads (integer match counts)
        # Model was trained on integer rolling match counts, not scaled fractions
        player_acute = team_workload["acute_load"]
        player_chronic = team_workload["chronic_load"]

        # Compute player-level ACWR with workload variation
        # ACWR was designed for active players. For bench/youth players who
        # rarely play, use neutral values to avoid false extremes.
        team_acwr = team_workload["acwr"]
        if play_ratio >= 0.15:
            # Active players: modulate team ACWR by play pattern
            # High-minute starters have stable ACWR (~team level)
            # Rotation players (0.3-0.7) have slightly spikier loads
            if play_ratio >= 0.6:
                player_acwr = team_acwr  # Starter: matches team schedule
            else:
                # Rotation: slight spike factor (max 1.15x at play_ratio=0.4)
                spike_factor = 1.0 + 0.15 * (1.0 - abs(2 * play_ratio - 0.8))
                player_acwr = team_acwr * max(1.0, spike_factor)
        else:
            # Bench/youth player: neutral ACWR (not enough data for real calc)
            player_acwr = 0.9

        player_acwr = max(0.5, min(2.5, player_acwr))

        # Fatigue: acute - chronic (uses integer match counts, same as training)
        player_fatigue = player_acute - player_chronic

        scaled_workload = {
            # Workload metrics: scale by play_ratio for player-level estimate
            "acute_load": round(player_acute, 2),
            "chronic_load": round(player_chronic, 2),
            "acwr": round(player_acwr, 3),
            "monotony": team_workload["monotony"],
            "strain": team_workload["strain"],
            "fatigue_index": round(player_fatigue, 2),
            "workload_slope": team_workload["workload_slope"],
            "spike_flag": 1 if player_acwr > 1.5 else 0,
            # Match counts: keep as integers (model trained on integer counts)
            "matches_last_7": team_workload["matches_last_7"],
            "matches_last_14": team_workload["matches_last_14"],
            "matches_last_30": team_workload["matches_last_30"],
        }

        if player_name in minutes_lookup:
            players_with_ratio += 1

        # Pre-set days_since_last_match from real team schedule
        # (prevents temporal feature engineering from computing 0 via date diff)
        rest_days = team_form["rest_days_before_injury"]

        rows.append({
            "name": player_name,
            "player_team": player["team"],
            "team": player["team"],
            "position": player.get("position", "Unknown"),
            "age": player.get("age", 25),
            "injury_datetime": snapshot_date,
            "snapshot_date": snapshot_date.strftime("%Y-%m-%d"),
            "playing_time_ratio": play_ratio,
            "days_since_last_match": rest_days,
            **scaled_workload,
            **team_form,
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

    # Compute full injury history features from detail pkl (most accurate source)
    detail_path = os.path.join(PROJECT_ROOT, "models", "player_injuries_detail.pkl")
    detail_stats = {}
    if os.path.exists(detail_path):
        try:
            detail_df = pd.read_pickle(detail_path)
            if "severity_days" in detail_df.columns and "name" in detail_df.columns:
                for name, grp in detail_df.groupby("name"):
                    detail_stats[name] = {
                        "worst": grp["severity_days"].max(),
                        "std": grp["severity_days"].std() if len(grp) > 1 else 0.0,
                    }
        except Exception as e:
            logger.debug(f"Failed to load detail stats: {e}")

    # Merge real injury history (scraped data is the primary source)
    matched_injuries = 0
    for idx, row in df.iterrows():
        player_name = row.get("name", "")
        if player_name in injury_history:
            hist = injury_history[player_name]
            count = hist["previous_injuries"]
            avg_sev = hist.get("avg_severity", 0)
            total_lost = hist.get("total_days_lost", 0)
            ds = detail_stats.get(player_name, {})
            df.at[idx, "previous_injuries"] = count
            df.at[idx, "days_since_last_injury"] = hist["days_since_last_injury"]
            df.at[idx, "total_days_lost"] = total_lost
            df.at[idx, "player_injury_count"] = count
            df.at[idx, "player_avg_severity"] = avg_sev
            df.at[idx, "player_worst_injury"] = ds.get("worst", avg_sev)
            df.at[idx, "player_severity_std"] = ds.get("std", 0.0)
            df.at[idx, "is_injury_prone"] = 1 if count >= 3 else 0
            matched_injuries += 1

    print(f"   Matched {matched_injuries}/{len(df)} players with injury history")

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

    # Keep raw model probabilities — the API uses percentile-based normalization
    # (normalize_risk_score + get_risk_level) to display relative risk, so arbitrary
    # post-hoc calibration is unnecessary and can distort the model's rankings.
    print(f"   Raw probs: mean={df['ensemble_prob'].mean():.1%}, "
          f"range=[{df['ensemble_prob'].min():.1%}, {df['ensemble_prob'].max():.1%}]")

    # Agreement: how many models agree the player is above-median risk
    median_prob = df["ensemble_prob"].median()
    model_probs = np.column_stack([df["lgb_prob"], df["xgb_prob"], df["catboost_prob"]])
    df["agreement"] = (model_probs > median_prob).sum(axis=1)
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

def run_injury_scraper(max_age_days: int = 1):
    """Run the injury scraper to update player_injuries_detail.pkl.

    Only re-scrapes players whose data is older than max_age_days,
    so this is fast when run frequently (only new/stale players).
    """
    import subprocess
    script = os.path.join(os.path.dirname(__file__), "scrape_injuries.py")
    if not os.path.exists(script):
        logger.warning("scrape_injuries.py not found, skipping injury update")
        return False

    print(f"\n   Running injury scraper (max-age={max_age_days} days)...")
    try:
        result = subprocess.run(
            [sys.executable, script, "--max-age", str(max_age_days)],
            capture_output=True, text=True, timeout=1800  # 30 min max
        )
        # Print summary lines only
        for line in result.stdout.splitlines():
            if any(kw in line.lower() for kw in ["done!", "scraped", "final", "players to scrape", "nothing to do"]):
                print(f"   {line.strip()}")
        if result.returncode != 0:
            logger.warning(f"Scraper exited with code {result.returncode}")
            if result.stderr:
                logger.warning(result.stderr[-500:])
            return False
        return True
    except subprocess.TimeoutExpired:
        logger.warning("Injury scraper timed out after 30 minutes")
        return False
    except Exception as e:
        logger.warning(f"Injury scraper failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Refresh predictions with live data")
    parser.add_argument("--mode", choices=["fbref", "api"], default="fbref",
                        help="Data source: 'fbref' (accurate) or 'api' (fast)")
    parser.add_argument("--api-key", help="Football-data.org API key (for api mode)")
    parser.add_argument("--dry-run", action="store_true", help="Don't save, just preview")
    parser.add_argument("--players", help="Comma-separated player names to filter to")
    parser.add_argument("--transfermarkt", action="store_true",
                        help="Fetch injury history from Transfermarkt for missing players (slower)")
    parser.add_argument("--scrape-injuries", action="store_true",
                        help="Update injury history from Transfermarkt before refreshing predictions")
    parser.add_argument("--scrape-max-age", type=int, default=1,
                        help="Max age in days before re-scraping a player's injury data (default: 1)")
    args = parser.parse_args()

    print("=" * 60)
    print("INJURY RISK PREDICTOR - Live Data Refresh")
    print("=" * 60)
    print(f"Mode: {args.mode.upper()}")

    # Step 0: Update injury history if requested
    if args.scrape_injuries:
        print("\n0. Updating injury history from Transfermarkt...")
        run_injury_scraper(max_age_days=args.scrape_max_age)

    # Load models
    print("\n1. Loading trained models...")
    artifacts = load_artifacts()
    if artifacts is None or "ensemble" not in artifacts:
        print("ERROR: No trained models found. Run the notebook first.")
        sys.exit(1)

    ensemble = artifacts["ensemble"]
    severity_clf = artifacts.get("severity_clf")
    player_history = artifacts.get("player_history", {})
    archetype_df = artifacts.get("df_clusters")
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
