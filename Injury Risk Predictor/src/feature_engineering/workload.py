import pandas as pd
import numpy as np


def add_workload_metrics(team_matches: pd.DataFrame,
                         window_acute: int = 7,
                         window_chronic: int = 28) -> pd.DataFrame:
    """
    Adds workload science metrics used in elite sports:
      - Acute load (7-day rolling)
      - Chronic load (28-day rolling)
      - ACWR ratio
      - Monotony index
      - Training strain
      - Fatigue index
      - Workload slope (trend)
      - Spike indicator

    Here we use match frequency: dummy count per match day.

    Input must contain:
        ['team', 'match_date']

    Returns:
        DataFrame with new workload features appended.
    """

    df = team_matches.copy()

    # Ensure datetime and sorted
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df = df.sort_values(["team", "match_date"]).reset_index(drop=True)

    # Dummy load using matches (1 per match)
    df["dummy_load"] = 1

    # Set index to match_date (required for D-based windows)
    df = df.set_index("match_date")

    groups = df.groupby("team")

    # -----------------------------
    # 1️⃣ ACUTE LOAD (7 DAY WINDOW)
    # -----------------------------
    df["acute_load"] = (
        groups["dummy_load"]
        .rolling(f"{window_acute}D")
        .sum()
        .reset_index(level=0, drop=True)
    )

    # -----------------------------
    # 2️⃣ CHRONIC LOAD (28 DAY WINDOW)
    # -----------------------------
    df["chronic_load"] = (
        groups["dummy_load"]
        .rolling(f"{window_chronic}D")
        .sum()
        .reset_index(level=0, drop=True)
    )

    # -----------------------------
    # 3️⃣ ACWR
    # -----------------------------
    df["acwr"] = df["acute_load"] / df["chronic_load"].replace(0, np.nan)

    # -----------------------------
    # 4️⃣ MONOTONY = mean / std (14D window)
    # -----------------------------
    df["monotony"] = (
        groups["dummy_load"]
        .rolling("14D")
        .apply(lambda x: np.mean(x) / (np.std(x) + 1e-9))
        .reset_index(level=0, drop=True)
    )

    # -----------------------------
    # 5️⃣ STRAIN = load * monotony
    # -----------------------------
    df["strain"] = df["acute_load"] * df["monotony"]

    # -----------------------------
    # 6️⃣ FATIGUE INDEX
    # -----------------------------
    df["fatigue_index"] = df["acute_load"] - df["chronic_load"]

    # -----------------------------
    # 7️⃣ WORKLOAD SLOPE (trend of last 5 matches)
    # -----------------------------
    def slope(arr):
        if len(arr) < 2:
            return 0
        x = np.arange(len(arr))
        return np.polyfit(x, arr, 1)[0]

    df["workload_slope"] = (
        groups["acute_load"]
        .rolling(5, min_periods=2)
        .apply(slope)
        .reset_index(level=0, drop=True)
    )

    # -----------------------------
    # 8️⃣ SPIKE FLAG (ACWR danger zone > 1.5)
    # -----------------------------
    df["spike_flag"] = (df["acwr"] > 1.5).astype(int)

    # Reset index for merging with injuries later
    df = df.reset_index()

    return df