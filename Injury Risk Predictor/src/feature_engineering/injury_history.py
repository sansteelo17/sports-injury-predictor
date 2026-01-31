import pandas as pd

def add_injury_history_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Count previous injuries per player
    df["previous_injuries"] = df.groupby("name").cumcount()

    # Time between injuries
    df["days_since_last_injury"] = (
        df.groupby("name")["injury_datetime"]
          .diff()
          .dt.days
    ).fillna(999)

    # Rename / add rest feature
    if "rest_days" in df.columns:
        df["rest_days_before_injury"] = df["rest_days"]

    return df