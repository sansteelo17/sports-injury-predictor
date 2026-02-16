import pandas as pd


def build_classification_dataset(final_df: pd.DataFrame,
                                 negative_samples: pd.DataFrame,
                                 include_date: bool = True) -> pd.DataFrame:
    """
    Builds the final classification dataset by:
    - labeling positive samples (injuries)
    - computing player injury history features from severity data
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

    # -------------------------
    # 1b. Compute player injury history features
    # -------------------------
    # Compute severity_days if not present (from date_of_return - date_of_injury)
    if "severity_days" not in final_positive.columns:
        if "date_of_return" in final_positive.columns and "date_of_injury" in final_positive.columns:
            final_positive["severity_days"] = (
                pd.to_datetime(final_positive["date_of_return"]) -
                pd.to_datetime(final_positive["date_of_injury"])
            ).dt.days

    if "severity_days" in final_positive.columns and "name" in final_positive.columns:
        # Compute per-player injury stats from severity data
        player_stats = final_positive.groupby("name").agg(
            player_injury_count=("severity_days", "count"),
            player_avg_severity=("severity_days", "mean"),
            player_worst_injury=("severity_days", "max"),
            player_severity_std=("severity_days", "std"),
            total_days_lost=("severity_days", "sum"),
        ).reset_index()

        player_stats["player_severity_std"] = player_stats["player_severity_std"].fillna(0)
        player_stats["is_injury_prone"] = (player_stats["player_injury_count"] >= 3).astype(int)

        # Merge back onto positives
        for col in ["player_injury_count", "player_avg_severity", "player_worst_injury",
                     "player_severity_std", "is_injury_prone", "total_days_lost"]:
            if col in final_positive.columns:
                final_positive = final_positive.drop(columns=[col])

        final_positive = final_positive.merge(player_stats, on="name", how="left")

        print(f"Added player injury history features to classification dataset")
        print(f"  Injury-prone players (3+ injuries): {player_stats['is_injury_prone'].sum()}")

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

    # Injury history (basic)
    "previous_injuries", "days_since_last_injury",

    # Injury history (rich - player-level stats)
    "player_injury_count", "player_avg_severity",
    "player_worst_injury", "player_severity_std",
    "is_injury_prone", "total_days_lost",

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
    # 2. Player-level negative sampling
    # -------------------------
    # Instead of assigning random player info to team-level negatives,
    # create player-specific negatives: for each known player, pair them
    # with their team's match dates where they didn't get injured.
    # This preserves the correct injury history for each negative sample.

    player_info_cols = [
        "name", "player_team", "position", "age", "fifa_rating",
        "previous_injuries", "days_since_last_injury",
        "player_injury_count", "player_avg_severity",
        "player_worst_injury", "player_severity_std",
        "is_injury_prone", "total_days_lost",
    ]
    player_info_cols = [c for c in player_info_cols if c in final_positive.columns]

    # Get unique player profiles (use last known state per player)
    player_info = (
        final_positive[player_info_cols]
        .drop_duplicates(subset=["name"], keep="last")
    )

    valid_teams = final_positive["player_team"].unique()
    negative_samples = negative_samples[
        negative_samples["player_team"].isin(valid_teams)
    ].copy()

    # Build set of (player, approximate_date) for positive samples to exclude
    injury_dates = set()
    if "injury_datetime" in final_positive.columns:
        for _, row in final_positive[["name", "injury_datetime"]].iterrows():
            injury_dates.add((row["name"], pd.to_datetime(row["injury_datetime"]).date()))

    # -------------------------
    # 3. Create player-level negatives
    # -------------------------
    player_negatives = []
    for _, player in player_info.iterrows():
        team = player["player_team"]
        name = player["name"]

        # Get all match dates for this player's team
        team_matches = negative_samples[negative_samples["player_team"] == team].copy()
        if team_matches.empty:
            continue

        # Exclude match dates within 14 days of this player's injuries
        team_matches["match_date_dt"] = pd.to_datetime(team_matches["match_date"])
        keep_mask = pd.Series(True, index=team_matches.index)
        for _, pos_row in final_positive[final_positive["name"] == name].iterrows():
            inj_dt = pd.to_datetime(pos_row.get("injury_datetime", pd.NaT))
            if pd.notna(inj_dt):
                too_close = abs((team_matches["match_date_dt"] - inj_dt).dt.days) <= 14
                keep_mask = keep_mask & ~too_close

        team_matches = team_matches[keep_mask]
        if team_matches.empty:
            continue

        # Sample up to 10 negatives per player (keeps dataset manageable)
        n_sample = min(10, len(team_matches))
        sampled = team_matches.sample(n=n_sample, random_state=hash(name) % 2**31)

        # Assign this player's real info to their negatives
        for col in player_info_cols:
            if col != "player_team":  # player_team already in team_matches
                sampled[col] = player[col]

        player_negatives.append(sampled)

    negative_samples = pd.concat(player_negatives, ignore_index=True)

    # Clean up temp column
    if "match_date_dt" in negative_samples.columns:
        negative_samples = negative_samples.drop(columns=["match_date_dt"])

    print(f"  Player-level negatives: {len(negative_samples)} samples for {len(player_negatives)} players")

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
