import pandas as pd

# ------------------------------------------------------------------
# TEAM NAME NORMALIZATION MAPPING
# ------------------------------------------------------------------
TEAM_MAPPING = {
    # Manchester United
    "man united": "manchester united",
    "man utd": "manchester united",
    "manchester utd": "manchester united",

    # Newcastle
    "newcastle": "newcastle united",
    "newcastle utd": "newcastle united",

    # Wolves formatting edge cases
    "wolverhampton": "wolves",
    "wolverhampton wanderers": "wolves",

    # Spurs formatting cases
    "tottenham hotspur": "tottenham",
    "spurs": "tottenham",
}


# ------------------------------------------------------------------
# GENERAL TEAM NORMALIZATION FOR ANY DATAFRAME COLUMN
# ------------------------------------------------------------------
def normalize_team_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Normalize team names for merging:
      - lowercase
      - strip whitespace
      - fix variants using TEAM_MAPPING
    Returns a NEW dataframe (safe, avoids SettingWithCopyWarning).
    """
    df = df.copy()
    df.loc[:, column] = (
        df[column]
        .astype(str)
        .str.lower()
        .str.strip()
        .replace(TEAM_MAPPING)
    )
    return df


# ------------------------------------------------------------------
# MATCH DATA NORMALIZATION
# (home_team, away_team, season_year)
# ------------------------------------------------------------------
def convert_season_end_year(year):
    """
    EPL datasets store season by END YEAR.
    Example:
        Season_End_Year = 2020 → "2019-2020"
    """
    year = int(year)
    start = year - 1
    end = year
    return f"{start}-{end}"


def normalize_match_teams(match_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize:
      - home_team / away_team
      - unify naming styles
      - create match_date as datetime
      - normalize season using EPL convention
    """
    df = match_df.copy()

    # Cleanup team name columns
    df = normalize_team_column(df, "home")
    df = normalize_team_column(df, "away")

    df = df.rename(columns={
        "home": "home_team",
        "away": "away_team"
    })

    # Match date
    df["match_date"] = pd.to_datetime(df["date"], errors="coerce")

    # Normalize season
    df["season_year"] = df["season_end_year"].apply(convert_season_end_year)

    return df


# ------------------------------------------------------------------
# MISMATCH INSPECTION (OPTIONAL)
# ------------------------------------------------------------------
def inspect_team_mismatches(injury_df, stats_df, match_df):
    """
    Utility for exploration only — shows mismatches between datasets.
    Helps you update TEAM_MAPPING when needed.
    """
    injury_teams = set(injury_df["player_team"])
    stats_teams = set(stats_df["player_team"])
    match_teams = set(match_df["home_team"]) | set(match_df["away_team"])

    return {
        "injury_vs_stats": injury_teams - stats_teams,
        "injury_vs_matches": injury_teams - match_teams,
        "stats_vs_matches": stats_teams - match_teams,
    }