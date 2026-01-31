import pandas as pd
import numpy as np
import unidecode
import unicodedata

from src.preprocessing.team_normalization import normalize_team_column


def _clean_unicode(s):
    """Remove weird unicode artifacts (e.g., accents, special chars)."""
    if isinstance(s, str):
        return ''.join(c for c in unicodedata.normalize("NFKD", s) if ord(c) < 128)
    return s


def _normalize_season(season):
    """
    Convert formats like `2019/20` → `2019-2020`.
    """
    season = str(season)
    if "/" in season:
        start, end = season.split("/")
        return f"{start}-{start[:2]}{end}"
    return season


def clean_injury_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw injury dataset:
      - standardizes columns
      - fixes missing values and N.A variants
      - normalizes player names
      - removes future-info leakage (after_injury columns)
      - normalizes seasons and team names
      - cleans and parses injury/return dates
      - adds injury_datetime for time-aware merging
    """

    df = df.copy()

    # -----------------------------
    # 1. Standardize column names
    # -----------------------------
    df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")

    # -----------------------------
    # 2. Replace NA variants
    # -----------------------------
    na_values = ["N.A", "N.A.", "NA", "N/A", "n.a", "NaN", ""]
    df.replace(na_values, np.nan, inplace=True)

    # -----------------------------
    # 3. Remove leakage columns
    # -----------------------------
    leakage_cols = [c for c in df.columns if
                    "after_injury" in c or
                    "missed_match" in c]
    df.drop(columns=leakage_cols, inplace=True, errors="ignore")

    # -----------------------------
    # 4. Normalize player names
    # -----------------------------
    df["player_name"] = (
        df["name"]
        .astype(str)
        .str.lower()
        .str.strip()
        .apply(unidecode.unidecode)
    )

    # -----------------------------
    # 5. Normalize seasons
    # -----------------------------
    df["season_year"] = df["season"].apply(_normalize_season)

    # -----------------------------
    # 6. Normalize team names
    # -----------------------------
    df = normalize_team_column(df, "team_name")
    df = df.rename(columns={"team_name": "player_team"})

    # -----------------------------
    # 7. Clean unicode + comma formatting in dates
    # -----------------------------
    for col in ["date_of_injury", "date_of_return"]:
        df[col] = df[col].apply(_clean_unicode)
        df[col] = df[col].str.replace(",", ", ", regex=False)  # Fix "Aug 12,2020"

    # -----------------------------
    # 8. Convert to datetime
    # -----------------------------
    df["date_of_injury"] = pd.to_datetime(df["date_of_injury"], errors="coerce")
    df["date_of_return"] = pd.to_datetime(df["date_of_return"], errors="coerce")

    # -----------------------------
    # 9. Add final precise timestamp (23:59)
    # -----------------------------
    df["injury_datetime"] = df["date_of_injury"] + pd.Timedelta(hours=23, minutes=59)

    return df