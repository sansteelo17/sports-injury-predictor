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

    # Filter Premier League rows
    df = df[df["league"].str.lower().str.contains("premier", na=False)].copy()

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

    return df