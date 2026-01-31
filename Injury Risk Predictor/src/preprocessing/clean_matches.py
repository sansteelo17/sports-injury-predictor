import pandas as pd
from src.preprocessing.team_normalization import normalize_team_column


def clean_match_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans EPL match data.
    Supports BOTH:
      - raw CSV format (home/away/date/homegoals/awaygoals)
      - already-cleaned format (home_team/away_team/match_date)

    Steps:
      - standardize column names
      - normalize team names
      - ensure match_date exists
      - ensure home_team & away_team exist
      - normalize season_year
      - return sorted dataframe
    """

    df = df.copy()
    df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")

    # --------------------------------------------------
    # 1. DATE COLUMN (raw or cleaned)
    # --------------------------------------------------
    if "match_date" not in df.columns:
        if "date" in df.columns:
            df["match_date"] = pd.to_datetime(df["date"], errors="coerce")
        else:
            raise ValueError("Match dataset must contain 'date' or 'match_date'.")

    # --------------------------------------------------
    # 2. TEAM COLUMNS (raw or cleaned)
    # --------------------------------------------------
    # Raw format: home / away
    # Cleaned format: home_team / away_team
    if "home" in df.columns and "away" in df.columns:
        df = df.rename(columns={
            "home": "home_team",
            "away": "away_team"
        })

    # Validate existence after rename
    if "home_team" not in df.columns or "away_team" not in df.columns:
        raise ValueError(
            f"Match dataframe must have 'home_team' and 'away_team'. Found: {df.columns.tolist()}"
        )

    # Normalize team names
    df = normalize_team_column(df, "home_team")
    df = normalize_team_column(df, "away_team")

    # --------------------------------------------------
    # 3. GOALS COLUMNS (handle multiple naming variants)
    # --------------------------------------------------
    rename_map = {
        "homegoals": "homegoals",
        "home_goals": "homegoals",
        "home_goals_scored": "homegoals",
        "awaygoals": "awaygoals",
        "away_goals": "awaygoals",
        "away_goals_scored": "awaygoals",
    }

    for col in rename_map:
        if col in df.columns and rename_map[col] not in df.columns:
            df = df.rename(columns={col: rename_map[col]})

    # --------------------------------------------------
    # 4. SEASON YEAR (convert Season_End_Year → YYYY-YYYY)
    # --------------------------------------------------
    if "season_year" not in df.columns:
        if "season_end_year" not in df.columns:
            raise ValueError("Match dataset missing 'season_end_year' column.")
        
        df["season_year"] = df["season_end_year"].apply(
            lambda y: f"{int(y)-1}-{int(y)}"
        )

    # --------------------------------------------------
    # 5. Final sorting
    # --------------------------------------------------
    df = df.sort_values("match_date").reset_index(drop=True)

    return df