import pandas as pd


def build_classification_dataset(final_df: pd.DataFrame,
                                 negative_samples: pd.DataFrame,
                                 include_date: bool = True) -> pd.DataFrame:
    """
    Builds the final classification dataset by:
    - labeling positive samples (injuries)
    - merging player info onto negative samples
    - aligning features
    - concatenating positive + negative examples

    Args:
        final_df: DataFrame with injury cases and features
        negative_samples: DataFrame with non-injury samples
        include_date: If True, includes event_date column for temporal validation
    """

    # -------------------------
    # 1. Positive samples
    # -------------------------
    final_positive = final_df.copy()
    final_positive["injury_label"] = 1

    # Add event_date for temporal validation (from injury_datetime)
    if include_date and "injury_datetime" in final_positive.columns:
        final_positive["event_date"] = pd.to_datetime(final_positive["injury_datetime"])
    elif include_date and "date_of_injury" in final_positive.columns:
        final_positive["event_date"] = pd.to_datetime(final_positive["date_of_injury"])

    feature_cols = [
    # Player + static info
    "player_team", "position", "age", "fifa_rating",

    # Match workload + congestion
    "matches_last_7", "matches_last_14", "matches_last_30",
    "rest_days_before_injury", "avg_rest_last_5",

    # Performance rolling windows
    "goals_for_last_5", "goals_against_last_5",
    "goal_diff_last_5", "avg_goal_diff_last_5",
    "form_last_5", "form_avg_last_5",
    "win_ratio_last_5", "win_streak", "loss_streak",

    # Injury history
    "previous_injuries", "days_since_last_injury",

    # Workload analytics
    "acute_load", "chronic_load", "acwr",
    "monotony", "strain", "fatigue_index",
    "workload_slope", "spike_flag",

    # Target
    "injury_label"
    ]

    # Add event_date for temporal validation
    if include_date:
        feature_cols.append("event_date")

    # -------------------------
    # 2. Extract player info (needed for negatives)
    # -------------------------
    player_info = final_positive[[
        "name", "player_team", "position", "age", "fifa_rating",
        "previous_injuries", "days_since_last_injury"
    ]].drop_duplicates()

    valid_teams = final_positive["player_team"].unique()
    negative_samples = negative_samples[
        negative_samples["player_team"].isin(valid_teams)
    ].copy()

    # -------------------------
    # 3. Merge player info into negatives
    # -------------------------
    negative_samples = negative_samples.merge(
        player_info,
        on="player_team",
        how="left"
    )

    # one negative snapshot per match-date per team
    negative_samples = (
        negative_samples
        .groupby(["player_team", "match_date"])
        .apply(lambda g: g.sample(1, random_state=42))
        .reset_index(drop=True)
    )

    # fill injury history
    negative_samples["previous_injuries"] = \
        negative_samples["previous_injuries"].fillna(0)
    negative_samples["days_since_last_injury"] = \
        negative_samples["days_since_last_injury"].fillna(999)

    # rest days
    if "rest_days" in negative_samples.columns:
        negative_samples["rest_days_before_injury"] = negative_samples["rest_days"]

    # label negatives
    negative_samples["injury_label"] = 0

    # Add event_date for temporal validation (from match_date for negatives)
    if include_date and "match_date" in negative_samples.columns:
        negative_samples["event_date"] = pd.to_datetime(negative_samples["match_date"])

    # -------------------------
    # 4. Combine positive + negative datasets
    # -------------------------
    # Only include columns that exist in both
    cols_to_use = [c for c in feature_cols if c in final_positive.columns and c in negative_samples.columns]

    final_df_out = pd.concat([
        final_positive[cols_to_use],
        negative_samples[cols_to_use]
    ], ignore_index=True)

    # Sort by date for proper temporal ordering
    if "event_date" in final_df_out.columns:
        final_df_out = final_df_out.sort_values("event_date").reset_index(drop=True)

    return final_df_out