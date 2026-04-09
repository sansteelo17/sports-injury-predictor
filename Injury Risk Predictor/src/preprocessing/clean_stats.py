import pandas as pd
from src.preprocessing.team_normalization import normalize_team_column


def _normalize_season(season):
    season = str(season)
    if "/" in season:
        start, end = season.split("/")
        return f"{start}-{start[:2]}{end}"
    return season


def clean_stats_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean player stats for merging with injury data.
    """

    # ALWAYS work on a copy to avoid chained warnings
    df = df.copy()

    # Standardize columns
    df.columns = df.columns.str.lower().str.strip()

    # Keep the leagues we actively model today
    league_key = df["league"].astype(str).str.lower()
    df = df[league_key.str.contains("premier|la liga", na=False)].copy()
    df.loc[:, "league"] = df["league"].astype(str).map(
        lambda value: "La Liga" if "la liga" in value.lower() else "Premier League"
    )

    # Normalize strings safely using .loc
    df.loc[:, "player_name"] = (
        df["player"]
        .str.lower()
        .str.strip()
        .str.normalize("NFKD")
        .str.encode("ascii", errors="ignore")
        .str.decode("utf-8")
    )

    # Normalize season
    df.loc[:, "season_year"] = df["season"].apply(_normalize_season)

    # Normalize team names
    df = normalize_team_column(df, "squad")

    df = df.rename(columns={"squad": "player_team"})

    # A tiny number of historical rows are duplicated in the source dump.
    # Keep the richest season row so downstream merges stay many-to-one.
    if "min" in df.columns:
        df = df.sort_values("min", ascending=False)
    df = df.drop_duplicates(subset=["player_name", "player_team", "season_year"], keep="first")

    return df
