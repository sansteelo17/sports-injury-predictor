import pandas as pd

def impute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Fill rolling windows
    df[["matches_last_7", "matches_last_14", "matches_last_30"]] = \
        df[["matches_last_7", "matches_last_14", "matches_last_30"]].fillna(0)

    # Fill rest features using global mean
    df["rest_days_before_injury"] = df["rest_days_before_injury"].fillna(
        df["rest_days_before_injury"].mean()
    )
    df["avg_rest_last_5"] = df["avg_rest_last_5"].fillna(
        df["avg_rest_last_5"].mean()
    )

    # Team-based filling for performance stats
    perf_cols = [
        "goals_for_last_5", "goals_against_last_5", "goal_diff_last_5",
        "avg_goal_diff_last_5", "form_last_5", "form_avg_last_5",
        "win_ratio_last_5", "win_streak", "loss_streak"
    ]

    team_means = df.groupby("player_team")[perf_cols].transform("mean")
    df[perf_cols] = df[perf_cols].fillna(team_means)

    # Workload metrics
    df["acute_load"] = df["acute_load"].fillna(0)
    df["chronic_load"] = df["chronic_load"].fillna(0)
    df["acwr"] = df["acwr"].fillna(0)
    df["monotony"] = df["monotony"].fillna(1)
    df["strain"] = df["strain"].fillna(0)
    df["fatigue_index"] = df["fatigue_index"].fillna(0)

    # workload_slope → fill with team average or 0
    df["workload_slope"] = df["workload_slope"].fillna(
        df.groupby("player_team")["workload_slope"].transform("mean")
    )
    df["workload_slope"] = df["workload_slope"].fillna(0)

    # spike_flag → fill with 0
    df["spike_flag"] = df["spike_flag"].fillna(0)

    return df