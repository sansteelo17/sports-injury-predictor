"""
Player statistics features for injury prediction.

Uses historical player performance data to extract injury-relevant features:
- Workload indicators (minutes, matches)
- Physical exertion metrics (distance, carries)
- Playing style indicators (tackles, shots)
- Career trajectory (experience, peak detection)
"""

import pandas as pd


def load_player_stats(filepath: str = "data/raw/All_Players_1992-2025.csv") -> pd.DataFrame:
    """Load and prepare player statistics data."""
    df = pd.read_csv(filepath)

    # Standardize column names
    df.columns = df.columns.str.strip()

    # Clean player names for matching
    df["player_name_clean"] = df["Player"].str.lower().str.strip()

    return df


def get_season_stats(player_stats: pd.DataFrame, season: str) -> pd.DataFrame:
    """
    Get player statistics for a specific season.

    Args:
        player_stats: Full player stats DataFrame
        season: Season string like "2023-2024"

    Returns:
        DataFrame filtered to that season
    """
    return player_stats[player_stats["Season"] == season].copy()


def calculate_workload_features(player_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate workload-related features from player stats.

    High workload = higher injury risk.
    """
    df = player_stats.copy()

    # Minutes per match (intensity of involvement)
    df["minutes_per_match"] = df["Min"] / df["MP"].replace(0, 1)

    # Percentage of possible minutes played (overuse indicator)
    # Assuming ~38 matches * 90 min = 3420 max minutes per season
    df["season_minutes_pct"] = df["Min"] / 3420

    # Starter ratio (regular starters may have different injury patterns)
    df["starter_ratio"] = df["Starts"] / df["MP"].replace(0, 1)

    # Sub usage (coming off bench cold can increase injury risk)
    df["sub_ratio"] = df["Subs"] / df["MP"].replace(0, 1)

    return df


def calculate_physical_features(player_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate physical exertion features.

    High physical output = higher muscle injury risk.
    """
    df = player_stats.copy()

    # Total distance per 90 (if available)
    if "TotDist" in df.columns:
        df["dist_per_90"] = df["TotDist"] / df["90s"].replace(0, 1)

    # Progressive distance per 90 (high-intensity running)
    if "PrgDist" in df.columns:
        df["prog_dist_per_90"] = df["PrgDist"] / df["90s"].replace(0, 1)

    # Carries per 90 (ball-carrying workload)
    if "Carries" in df.columns:
        df["carries_per_90"] = df["Carries"] / df["90s"].replace(0, 1)

    # Progressive carries per 90 (explosive movements)
    if "PrgC" in df.columns:
        df["prog_carries_per_90"] = df["PrgC"] / df["90s"].replace(0, 1)

    return df


def calculate_contact_features(player_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate contact-related features.

    High contact = higher contact injury risk.
    """
    df = player_stats.copy()

    # Tackles per 90 (contact frequency)
    if "Tkl" in df.columns:
        df["tackles_per_90"] = df["Tkl"] / df["90s"].replace(0, 1)

    # Tackle success rate (failed tackles = awkward contact)
    if "Tkl" in df.columns and "TklW" in df.columns:
        df["tackle_success_rate"] = df["TklW"] / df["Tkl"].replace(0, 1)

    # Times tackled per 90 (receiving contact)
    if "Tkld" in df.columns:
        df["tackled_per_90"] = df["Tkld"] / df["90s"].replace(0, 1)

    # Fouls committed per 90
    if "Fls" in df.columns:
        df["fouls_per_90"] = df["Fls"] / df["90s"].replace(0, 1)

    # Aerial duels (heading = neck/head injury risk)
    if "Won" in df.columns:
        df["aerials_won_per_90"] = df["Won"] / df["90s"].replace(0, 1)

    return df


def calculate_playing_style_features(player_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate playing style features.

    Different styles have different injury profiles.
    """
    df = player_stats.copy()

    # Shooting frequency (strikers shoot more, different injury profile)
    if "Sh" in df.columns:
        df["shots_per_90"] = df["Sh"] / df["90s"].replace(0, 1)

    # Passing volume (midfielders pass more)
    if "Pass" in df.columns:
        df["passes_per_90"] = df["Pass"] / df["90s"].replace(0, 1)

    # Key passes per 90 (creative players)
    if "KP" in df.columns:
        df["key_passes_per_90"] = df["KP"] / df["90s"].replace(0, 1)

    # Defensive actions per 90 (combined)
    if "Tkl" in df.columns and "Int" in df.columns:
        df["def_actions_per_90"] = (df["Tkl"] + df["Int"]) / df["90s"].replace(0, 1)

    # Recoveries per 90
    if "Recov" in df.columns:
        df["recoveries_per_90"] = df["Recov"] / df["90s"].replace(0, 1)

    return df


def calculate_career_features(player_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate career trajectory features.

    Career stage affects injury risk:
    - Young players: less injury history but adaptation phase
    - Peak years: high intensity, moderate risk
    - Veteran players: accumulated wear, higher risk
    """
    df = player_stats.copy()

    # Career seasons (experience)
    career_seasons = df.groupby("PlayerID")["Season"].transform("count")
    df["career_seasons"] = career_seasons

    # Is this their first season? (adaptation risk)
    first_season = df.groupby("PlayerID")["Season"].transform("min")
    df["is_first_season"] = (df["Season"] == first_season).astype(int)

    # Career minutes (accumulated wear)
    df["career_minutes"] = df.groupby("PlayerID")["Min"].transform("cumsum")

    # Minutes trajectory (increasing/decreasing involvement)
    df["prev_season_min"] = df.groupby("PlayerID")["Min"].shift(1)
    df["minutes_change"] = df["Min"] - df["prev_season_min"].fillna(df["Min"])
    df["minutes_change_pct"] = df["minutes_change"] / df["prev_season_min"].replace(0, 1)

    return df


def add_player_stats_features(injury_df: pd.DataFrame,
                              player_stats: pd.DataFrame,
                              season: str = None) -> pd.DataFrame:
    """
    Merge player statistics features with injury data.

    Args:
        injury_df: DataFrame with injury/match data (must have 'name' column)
        player_stats: DataFrame with player statistics
        season: Optional season filter (e.g., "2023-2024")

    Returns:
        DataFrame with added player statistics features
    """
    df = injury_df.copy()

    # Filter to season if specified
    if season:
        stats = player_stats[player_stats["Season"] == season].copy()
    else:
        stats = player_stats.copy()

    # Apply all feature calculations
    stats = calculate_workload_features(stats)
    stats = calculate_physical_features(stats)
    stats = calculate_contact_features(stats)
    stats = calculate_playing_style_features(stats)
    stats = calculate_career_features(stats)

    # Clean name for matching
    if "name" in df.columns:
        df["player_name_clean"] = df["name"].str.lower().str.strip()
    elif "Player" in df.columns:
        df["player_name_clean"] = df["Player"].str.lower().str.strip()

    # Select features to merge
    feature_cols = [
        "player_name_clean",
        # Workload
        "minutes_per_match", "season_minutes_pct", "starter_ratio", "sub_ratio",
        # Physical
        "dist_per_90", "prog_dist_per_90", "carries_per_90", "prog_carries_per_90",
        # Contact
        "tackles_per_90", "tackle_success_rate", "tackled_per_90", "fouls_per_90",
        "aerials_won_per_90",
        # Style
        "shots_per_90", "passes_per_90", "key_passes_per_90", "def_actions_per_90",
        "recoveries_per_90",
        # Career
        "career_seasons", "is_first_season", "career_minutes",
        "minutes_change", "minutes_change_pct"
    ]

    # Only include columns that exist
    available_cols = [c for c in feature_cols if c in stats.columns]
    stats_subset = stats[available_cols].copy()

    # Aggregate if multiple rows per player (take most recent or mean)
    stats_agg = stats_subset.groupby("player_name_clean").agg("mean").reset_index()

    # Merge
    df = df.merge(stats_agg, on="player_name_clean", how="left")

    # Fill missing values with median (player not in stats dataset)
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    for col in numeric_cols:
        if col in available_cols:
            df[col] = df[col].fillna(df[col].median())

    # Clean up
    if "player_name_clean" in df.columns:
        df = df.drop(columns=["player_name_clean"])

    return df


def get_high_risk_indicators(player_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Identify players with high-risk statistical profiles.

    Returns DataFrame with risk indicators based on:
    - High workload (top 25% minutes)
    - High contact (top 25% tackles)
    - Sudden workload increase (>50% more minutes than last season)
    """
    df = player_stats.copy()
    df = calculate_workload_features(df)
    df = calculate_contact_features(df)
    df = calculate_career_features(df)

    # High workload indicator
    min_75 = df["Min"].quantile(0.75)
    df["high_workload"] = (df["Min"] > min_75).astype(int)

    # High contact indicator
    if "Tkl" in df.columns:
        tkl_75 = df["Tkl"].quantile(0.75)
        df["high_contact"] = (df["Tkl"] > tkl_75).astype(int)

    # Sudden workload spike
    df["workload_spike"] = (df["minutes_change_pct"] > 0.5).astype(int)

    # Combined risk score
    risk_cols = ["high_workload", "high_contact", "workload_spike"]
    available_risk = [c for c in risk_cols if c in df.columns]
    df["stats_risk_score"] = df[available_risk].sum(axis=1) / len(available_risk)

    return df
