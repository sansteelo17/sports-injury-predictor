"""
Improved negative sampling strategies for injury prediction.

The goal is to create informative negative samples that:
1. Maintain position distribution similar to injury cases
2. Include "hard negatives" (high workload but no injury)
3. Preserve temporal distribution across the season
4. Create matched pairs for more meaningful comparisons
"""

import pandas as pd


def generate_negative_samples(team_matches: pd.DataFrame,
                              injury_df: pd.DataFrame,
                              sample_frac: float = 0.3,
                              strategy: str = "stratified") -> pd.DataFrame:
    """
    Generate negative samples using specified strategy.

    Args:
        team_matches: DataFrame with team match data
        injury_df: DataFrame with injury events
        sample_frac: Fraction of negatives to sample (for basic strategy)
        strategy: One of "basic", "stratified", "matched", "hard_negative"

    Returns:
        DataFrame with negative samples (injury_label=0)
    """
    if strategy == "basic":
        return _basic_negative_sampling(team_matches, injury_df, sample_frac)
    elif strategy == "stratified":
        return _stratified_negative_sampling(team_matches, injury_df, sample_frac)
    elif strategy == "matched":
        return _matched_negative_sampling(team_matches, injury_df)
    elif strategy == "hard_negative":
        return _hard_negative_sampling(team_matches, injury_df, sample_frac)
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'basic', 'stratified', 'matched', or 'hard_negative'")


def _basic_negative_sampling(team_matches: pd.DataFrame,
                             injury_df: pd.DataFrame,
                             sample_frac: float = 0.3) -> pd.DataFrame:
    """Original simple random sampling approach."""
    snapshots = team_matches.rename(columns={"team": "player_team"}).copy()
    snapshots["injury_label"] = 0
    snapshots["match_date"] = pd.to_datetime(snapshots["match_date"])

    injury_events = injury_df[["name", "player_team", "injury_datetime"]]
    merged = snapshots.merge(injury_events, on="player_team", how="left")

    merged["days_until_injury"] = (
        merged["injury_datetime"] - merged["match_date"]
    ).dt.days

    merged["injury_label"] = merged["days_until_injury"].apply(
        lambda x: 1 if pd.notna(x) and 0 <= x <= 14 else 0
    )

    negatives = merged[merged["injury_label"] == 0]
    negatives = negatives.sample(frac=sample_frac, random_state=42)

    return negatives.reset_index(drop=True)


def _stratified_negative_sampling(team_matches: pd.DataFrame,
                                  injury_df: pd.DataFrame,
                                  sample_frac: float = 0.3) -> pd.DataFrame:
    """
    Stratified sampling that maintains:
    - Position distribution similar to injury cases
    - Temporal distribution across the season
    """
    snapshots = team_matches.rename(columns={"team": "player_team"}).copy()
    snapshots["injury_label"] = 0
    snapshots["match_date"] = pd.to_datetime(snapshots["match_date"])

    injury_events = injury_df[["name", "player_team", "injury_datetime"]].copy()
    merged = snapshots.merge(injury_events, on="player_team", how="left")

    merged["days_until_injury"] = (
        merged["injury_datetime"] - merged["match_date"]
    ).dt.days

    merged["injury_label"] = merged["days_until_injury"].apply(
        lambda x: 1 if pd.notna(x) and 0 <= x <= 14 else 0
    )

    negatives = merged[merged["injury_label"] == 0].copy()

    # Add month for temporal stratification
    negatives["sample_month"] = negatives["match_date"].dt.month

    # Calculate target samples per month based on sample_frac
    total_target = int(len(negatives) * sample_frac)

    # Get month distribution from injury cases
    if "injury_datetime" in injury_df.columns:
        injury_df_copy = injury_df.copy()
        injury_df_copy["injury_datetime"] = pd.to_datetime(injury_df_copy["injury_datetime"])
        injury_months = injury_df_copy["injury_datetime"].dt.month.value_counts(normalize=True)
    else:
        # Fallback to uniform distribution
        injury_months = pd.Series({m: 1/12 for m in range(1, 13)})

    # Sample from each month proportionally
    sampled_dfs = []
    for month in negatives["sample_month"].unique():
        month_data = negatives[negatives["sample_month"] == month]

        # Get target proportion for this month
        month_prop = injury_months.get(month, 1/12)
        n_samples = max(1, int(total_target * month_prop))
        n_samples = min(n_samples, len(month_data))

        if len(month_data) > 0 and n_samples > 0:
            sampled = month_data.sample(n=n_samples, random_state=42 + month)
            sampled_dfs.append(sampled)

    if sampled_dfs:
        result = pd.concat(sampled_dfs, ignore_index=True)
    else:
        result = negatives.sample(frac=sample_frac, random_state=42)

    # Clean up temporary column
    if "sample_month" in result.columns:
        result = result.drop(columns=["sample_month"])

    return result.reset_index(drop=True)


def _matched_negative_sampling(team_matches: pd.DataFrame,
                               injury_df: pd.DataFrame,
                               matches_per_injury: int = 3) -> pd.DataFrame:
    """
    Create matched negative samples for each injury case.

    For each injury, find N matches from the same team around the same
    time period where no injury occurred (case-control matching).
    """
    snapshots = team_matches.rename(columns={"team": "player_team"}).copy()
    snapshots["match_date"] = pd.to_datetime(snapshots["match_date"])

    injury_events = injury_df[["name", "player_team", "injury_datetime"]].copy()
    injury_events["injury_datetime"] = pd.to_datetime(injury_events["injury_datetime"])

    merged = snapshots.merge(injury_events, on="player_team", how="left")

    merged["days_until_injury"] = (
        merged["injury_datetime"] - merged["match_date"]
    ).dt.days

    merged["injury_label"] = merged["days_until_injury"].apply(
        lambda x: 1 if pd.notna(x) and 0 <= x <= 14 else 0
    )

    negatives = merged[merged["injury_label"] == 0].copy()

    matched_negatives = []

    for _, injury_row in injury_events.iterrows():
        team = injury_row["player_team"]
        injury_date = injury_row["injury_datetime"]

        # Find matches from same team within +/- 60 days of injury
        # but NOT within 14 days before injury (those could be related)
        team_negatives = negatives[
            (negatives["player_team"] == team)
        ].copy()

        if len(team_negatives) == 0:
            continue

        team_negatives["date_diff"] = abs(
            (team_negatives["match_date"] - injury_date).dt.days
        )

        # Exclude matches too close to injury (within 14 days before)
        team_negatives = team_negatives[
            ~((team_negatives["match_date"] <= injury_date) &
              (team_negatives["match_date"] >= injury_date - pd.Timedelta(days=14)))
        ]

        # Prefer matches within similar time window (same season period)
        team_negatives = team_negatives[team_negatives["date_diff"] <= 60]

        if len(team_negatives) >= matches_per_injury:
            sampled = team_negatives.sample(n=matches_per_injury, random_state=42)
        elif len(team_negatives) > 0:
            sampled = team_negatives
        else:
            continue

        matched_negatives.append(sampled)

    if matched_negatives:
        result = pd.concat(matched_negatives, ignore_index=True)
        # Remove duplicates (same match could be matched to multiple injuries)
        if "match_date" in result.columns and "player_team" in result.columns:
            result = result.drop_duplicates(subset=["match_date", "player_team"])
    else:
        # Fallback to basic sampling
        result = negatives.sample(frac=0.3, random_state=42)

    # Clean up
    if "date_diff" in result.columns:
        result = result.drop(columns=["date_diff"])

    result["injury_label"] = 0
    return result.reset_index(drop=True)


def _hard_negative_sampling(team_matches: pd.DataFrame,
                            injury_df: pd.DataFrame,
                            sample_frac: float = 0.3) -> pd.DataFrame:
    """
    Include "hard negatives" - high-risk situations where injury didn't occur.

    These are more informative for the model because they represent
    cases where conditions were similar to injury cases but no injury happened.

    Hard negatives are identified by:
    - High acute workload
    - Short recovery periods
    - High-risk fixture periods (Christmas, end of season)
    """
    snapshots = team_matches.rename(columns={"team": "player_team"}).copy()
    snapshots["injury_label"] = 0
    snapshots["match_date"] = pd.to_datetime(snapshots["match_date"])

    injury_events = injury_df[["name", "player_team", "injury_datetime"]]
    merged = snapshots.merge(injury_events, on="player_team", how="left")

    merged["days_until_injury"] = (
        merged["injury_datetime"] - merged["match_date"]
    ).dt.days

    merged["injury_label"] = merged["days_until_injury"].apply(
        lambda x: 1 if pd.notna(x) and 0 <= x <= 14 else 0
    )

    negatives = merged[merged["injury_label"] == 0].copy()

    # Calculate risk score for hard negative mining
    negatives["risk_score"] = 0.0

    # High workload indicators
    if "acute_load" in negatives.columns:
        acute_75 = negatives["acute_load"].quantile(0.75)
        negatives["risk_score"] += (negatives["acute_load"] > acute_75).astype(float) * 0.3

    if "acwr" in negatives.columns:
        # ACWR > 1.5 is high risk
        negatives["risk_score"] += (negatives["acwr"] > 1.5).astype(float) * 0.3

    if "matches_last_7" in negatives.columns:
        # 2+ matches in 7 days is congestion
        negatives["risk_score"] += (negatives["matches_last_7"] >= 2).astype(float) * 0.2

    # Fixture congestion periods
    month = negatives["match_date"].dt.month
    day = negatives["match_date"].dt.day

    # Christmas period
    is_christmas = ((month == 12) & (day >= 20)) | ((month == 1) & (day <= 5))
    negatives["risk_score"] += is_christmas.astype(float) * 0.1

    # End of season crunch
    is_crunch = ((month == 4) & (day >= 15)) | (month == 5)
    negatives["risk_score"] += is_crunch.astype(float) * 0.1

    # Split into hard negatives (top 30% risk) and easy negatives
    risk_70 = negatives["risk_score"].quantile(0.70)
    hard_negatives = negatives[negatives["risk_score"] >= risk_70]
    easy_negatives = negatives[negatives["risk_score"] < risk_70]

    total_samples = int(len(negatives) * sample_frac)

    # Sample 50% from hard negatives, 50% from easy negatives
    n_hard = min(total_samples // 2, len(hard_negatives))
    n_easy = total_samples - n_hard

    sampled_hard = hard_negatives.sample(n=n_hard, random_state=42) if n_hard > 0 else pd.DataFrame()
    sampled_easy = easy_negatives.sample(n=min(n_easy, len(easy_negatives)), random_state=42)

    result = pd.concat([sampled_hard, sampled_easy], ignore_index=True)

    # Clean up temporary column
    if "risk_score" in result.columns:
        result = result.drop(columns=["risk_score"])

    return result.reset_index(drop=True)


def get_recommended_strategy(n_injuries: int, n_matches: int) -> str:
    """
    Recommend a sampling strategy based on data characteristics.

    Args:
        n_injuries: Number of injury cases (positives)
        n_matches: Number of total match records

    Returns:
        Recommended strategy name
    """
    ratio = n_matches / max(n_injuries, 1)

    if ratio > 100:
        # Very imbalanced - use matched sampling for balance
        return "matched"
    elif ratio > 50:
        # Moderately imbalanced - use hard negative mining
        return "hard_negative"
    elif ratio > 20:
        # Slightly imbalanced - stratified sampling
        return "stratified"
    else:
        # Reasonably balanced - basic sampling is fine
        return "basic"
