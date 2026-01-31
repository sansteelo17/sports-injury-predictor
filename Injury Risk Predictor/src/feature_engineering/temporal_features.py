"""
Temporal features for injury prediction.

Calendar-based features that capture when injuries are more likely to occur:
- Season phase (pre-season, mid-season, end-season)
- Month effects (winter = more injuries due to hard pitches)
- Fixture congestion periods (Christmas, end of season)
- Day of week effects
"""

import pandas as pd
import numpy as np


def add_temporal_features(df: pd.DataFrame, date_column: str = "injury_datetime") -> pd.DataFrame:
    """
    Add calendar-based temporal features that affect injury risk.

    Args:
        df: DataFrame with injury/match data
        date_column: Name of the datetime column

    Returns:
        DataFrame with additional temporal features
    """
    df = df.copy()

    if date_column not in df.columns:
        print(f"Warning: {date_column} not found. Skipping temporal features.")
        return df

    # Ensure datetime
    df[date_column] = pd.to_datetime(df[date_column], errors="coerce")

    # Extract basic components
    df["month"] = df[date_column].dt.month
    df["day_of_week"] = df[date_column].dt.dayofweek  # 0=Monday, 6=Sunday
    df["week_of_year"] = df[date_column].dt.isocalendar().week.astype(int)

    # ---------------------------
    # Season phase features
    # ---------------------------
    # Football season: Aug-May
    # Pre-season: Aug-Sep (high injury risk - players not match fit)
    # Mid-season: Oct-Feb (varies)
    # End-season: Mar-May (fatigue accumulation)

    df["is_preseason"] = df["month"].isin([8, 9]).astype(int)
    df["is_midseason"] = df["month"].isin([10, 11, 12, 1, 2]).astype(int)
    df["is_endseason"] = df["month"].isin([3, 4, 5]).astype(int)

    # ---------------------------
    # Weather/pitch conditions
    # ---------------------------
    # Winter months = harder pitches, more muscle injuries
    df["is_winter"] = df["month"].isin([12, 1, 2]).astype(int)

    # ---------------------------
    # Fixture congestion periods
    # ---------------------------
    # Christmas period (Dec 20 - Jan 5): Extremely high fixture density
    df["is_christmas_period"] = (
        ((df["month"] == 12) & (df[date_column].dt.day >= 20)) |
        ((df["month"] == 1) & (df[date_column].dt.day <= 5))
    ).astype(int)

    # End of season crunch (Apr 15 - May 31)
    df["is_season_crunch"] = (
        ((df["month"] == 4) & (df[date_column].dt.day >= 15)) |
        (df["month"] == 5)
    ).astype(int)

    # ---------------------------
    # Day of week effects
    # ---------------------------
    # Weekend matches (Sat/Sun) vs midweek
    df["is_weekend_match"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_midweek_match"] = df["day_of_week"].isin([1, 2, 3]).astype(int)  # Tue, Wed, Thu

    # ---------------------------
    # Season progress
    # ---------------------------
    # Normalize week of season (Aug=week 1, May=week ~40)
    # Higher values = more accumulated fatigue
    def get_season_week(row):
        month = row["month"]
        week = row["week_of_year"]
        # Adjust so Aug=1, Sep=5, etc.
        if month >= 8:  # Aug-Dec
            return (month - 8) * 4 + (week % 4) + 1
        else:  # Jan-May
            return (month + 4) * 4 + (week % 4) + 1

    df["season_week"] = df.apply(get_season_week, axis=1)

    # Fatigue accumulation proxy (later in season = more fatigue)
    df["season_fatigue_factor"] = df["season_week"] / 40.0  # Normalized 0-1

    return df


def add_fixture_density_features(df: pd.DataFrame,
                                  date_column: str = "match_date",
                                  team_column: str = "player_team") -> pd.DataFrame:
    """
    Add fixture density features based on match schedule.

    Args:
        df: DataFrame with match data
        date_column: Name of the match date column
        team_column: Name of the team column

    Returns:
        DataFrame with fixture density features
    """
    df = df.copy()

    if date_column not in df.columns or team_column not in df.columns:
        print(f"Warning: Required columns not found. Skipping fixture density.")
        return df

    df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
    df = df.sort_values([team_column, date_column])

    # Days since last match (per team)
    df["days_since_last_match"] = (
        df.groupby(team_column)[date_column]
        .diff()
        .dt.days
        .fillna(7)  # Default 7 days for first match
    )

    # Fixture density: matches in next 14 days
    # (Requires future knowledge, so only use for training data analysis)

    # Very short recovery (< 3 days)
    df["short_recovery"] = (df["days_since_last_match"] < 3).astype(int)

    # Minimal recovery (3-4 days)
    df["minimal_recovery"] = df["days_since_last_match"].between(3, 4).astype(int)

    # Normal recovery (5-7 days)
    df["normal_recovery"] = df["days_since_last_match"].between(5, 7).astype(int)

    # Extended rest (> 7 days) - could indicate returning from break
    df["extended_rest"] = (df["days_since_last_match"] > 7).astype(int)

    return df
