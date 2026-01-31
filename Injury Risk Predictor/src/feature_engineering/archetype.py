import pandas as pd
import numpy as np

# ===============================================================
# PLAYER ARCHETYPE FEATURE ENGINEERING
# ===============================================================
def safe_div(a, b):
    """Safe division for scalars OR arrays."""
    try:
        return np.where(b == 0, 0, a / b)
    except Exception:
        # b is a scalar
        return 0 if b == 0 else a / b


def build_player_archetype_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes full injury-level dataset with:
        - name
        - injury_datetime
        - severity_days
        - body_area
        - injury_type
        - matches_last_14_days (from earlier merge)
        - match_count_before_injury
    Returns player-level feature matrix.
    """

    df = df.copy()

    # Ensure datetime
    if "injury_datetime" in df.columns:
        df["injury_datetime"] = pd.to_datetime(df["injury_datetime"])

    # -----------------------------------------------------------
    # 1. Basic per-player aggregations
    # -----------------------------------------------------------
    grouped = df.groupby("name")

    features = pd.DataFrame()
    features["total_injuries"] = grouped.size()
    features["avg_severity"] = grouped["severity_days"].mean()
    features["median_severity"] = grouped["severity_days"].median()
    features["std_severity"] = grouped["severity_days"].std().fillna(0)
    features["max_severity"] = grouped["severity_days"].max()

    # High severity: > 28 days
    features["high_severity_rate"] = grouped.apply(
        lambda g: safe_div((g["severity_days"] > 28).sum(), len(g))
    )

    # -----------------------------------------------------------
    # 2. Frequency: recurrence & spacing
    # -----------------------------------------------------------
    def compute_gaps(g):
        g = g.sort_values("injury_datetime")
        gaps = g["injury_datetime"].diff().dt.days.dropna()
        return pd.Series({
            "avg_days_between_injuries": gaps.mean() if len(gaps) > 0 else 0,
            "min_gap": gaps.min() if len(gaps) > 0 else 0,
            "max_gap": gaps.max() if len(gaps) > 0 else 0,
        })

    gap_df = grouped.apply(compute_gaps)
    features = features.join(gap_df)

    # Re-injury: same body area within 60 days
    def reinjury_rate(g):
        g = g.sort_values("injury_datetime")
        count = 0
        for i in range(1, len(g)):
            if (
                g.iloc[i]["body_area"] == g.iloc[i-1]["body_area"]
                and (g.iloc[i]["injury_datetime"] - g.iloc[i-1]["injury_datetime"]).days <= 60
            ):
                count += 1
        return safe_div(count, len(g))

    features["reinjury_rate"] = grouped.apply(reinjury_rate)

    # -----------------------------------------------------------
    # 3. Body area composition (% per area)
    # -----------------------------------------------------------
    body_counts = pd.crosstab(df["name"], df["body_area"])
    body_pct = body_counts.div(body_counts.sum(axis=1), axis=0).fillna(0)

    body_pct.columns = [f"pct_area_{c}" for c in body_pct.columns]

    features = features.join(body_pct, how="left")

    # Diversity of injury sites
    features["body_area_entropy"] = body_pct.apply(
        lambda row: -(row * np.log(row + 1e-9)).sum(), axis=1
    )

    # -----------------------------------------------------------
    # 4. Injury type composition
    # -----------------------------------------------------------
    type_counts = pd.crosstab(df["name"], df["injury_type"])
    type_pct = type_counts.div(type_counts.sum(axis=1), axis=0).fillna(0)

    type_pct.columns = [f"pct_type_{c}" for c in type_pct.columns]
    features = features.join(type_pct, how="left")

    # strain-to-tear ratio
    features["strain_to_tear"] = features.apply(
        lambda r: safe_div(r.get("pct_type_strain", 0), r.get("pct_type_tear", 0)),
        axis=1
    )

    # -----------------------------------------------------------
    # 5. Load-response: match congestion behaviour
    # -----------------------------------------------------------
    if "matches_last_14_days" in df.columns:
        features["avg_match_congestion"] = grouped["matches_last_14_days"].mean()

    if "match_count_before_injury" in df.columns:
        features["avg_matches_before_injury"] = grouped["match_count_before_injury"].mean()

    # -----------------------------------------------------------
    # 6. Variability of response (how unpredictable severity is)
    # -----------------------------------------------------------
    features["severity_cv"] = safe_div(features["std_severity"], features["avg_severity"])
    # -----------------------------------------------------------
    # 7. Long-term trend: severity increasing or decreasing
    # -----------------------------------------------------------
    def severity_trend(g):
        g = g.sort_values("injury_datetime")
        if len(g) < 3:
            return 0
        x = np.arange(len(g))
        y = g["severity_days"].values
        slope = np.polyfit(x, y, 1)[0]
        return slope

    features["severity_trend"] = grouped.apply(severity_trend)

    # -----------------------------------------------------------
    return features.reset_index()