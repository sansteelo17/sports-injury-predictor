import numpy as np
import pandas as pd


def _numeric(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce").fillna(default)


def _position_role_bucket(position: pd.Series, fallback: pd.Series) -> pd.Series:
    raw = (
        position.fillna("")
        .astype(str)
        .str.lower()
        .str.cat(fallback.fillna("").astype(str).str.lower(), sep=" ")
    )
    role = pd.Series("other", index=raw.index, dtype="object")
    role = role.mask(raw.str.contains(r"goalkeeper|keeper|\bgk\b", regex=True), "goalkeeper")
    role = role.mask(raw.str.contains(r"forward|winger|striker|centre-forward|center-forward|\bfw\b", regex=True), "attacker")
    role = role.mask(raw.str.contains(r"midfielder|midfield|playmaker|\bmf\b|\bdm\b|\bam\b|\bcm\b", regex=True), "midfielder")
    role = role.mask(raw.str.contains(r"defender|centre-back|center-back|full-back|wing-back|\bdf\b|back", regex=True), "defender")
    return role


def add_contextual_classification_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add football-context features shared by training and live inference.

    The goal is to make the model look beyond pure injury recency:
    - player usage / role importance
    - attacking involvement
    - opponent strength and fixture edge
    - smoother injury-recency pressure instead of raw days dominating everything
    """
    out = df.copy()

    minutes = _numeric(out, "min", 0.0)
    if minutes.eq(0).all():
        minutes = _numeric(out, "minutes_played", 0.0)

    appearances = _numeric(out, "mp", 0.0)
    if appearances.eq(0).all():
        appearances = _numeric(out, "appearances", 0.0)

    starts = _numeric(out, "starts", 0.0)
    season_90s = _numeric(out, "90s", 0.0)
    goals = _numeric(out, "gls", 0.0)
    assists = _numeric(out, "ast", 0.0)
    if goals.eq(0).all():
        goals = _numeric(out, "goals", 0.0)
    if assists.eq(0).all():
        assists = _numeric(out, "assists", 0.0)

    min_pct = _numeric(out, "min%", np.nan)
    goals_per_90_live = _numeric(out, "goals_per_90", 0.0)
    assists_per_90_live = _numeric(out, "assists_per_90", 0.0)
    shots_per_90_live = _numeric(out, "shots_per_90", 0.0)
    saves_per_90_live = _numeric(out, "saves_per_90", 0.0)

    role_bucket = _position_role_bucket(out.get("position", pd.Series(index=out.index, dtype="object")), out.get("pos", pd.Series(index=out.index, dtype="object")))

    minutes_share = (min_pct / 100.0).clip(lower=0.0, upper=1.0)
    minutes_share = minutes_share.fillna((minutes / 3000.0).clip(lower=0.0, upper=1.0))

    starter_ratio = (starts / appearances.replace(0, np.nan)).fillna(0.0).clip(lower=0.0, upper=1.0)
    inferred_starter_ratio = (minutes / (appearances.replace(0, np.nan) * 70.0)).fillna(0.0).clip(lower=0.0, upper=1.0)
    starter_ratio = starter_ratio.mask(starter_ratio.eq(0.0), inferred_starter_ratio)

    involvement_per90_train = ((goals + assists) / season_90s.replace(0, np.nan)).fillna(0.0)
    goal_involvement_per90 = involvement_per90_train.mask(
        involvement_per90_train.eq(0.0),
        goals_per_90_live + assists_per_90_live,
    ).clip(lower=0.0, upper=2.5)

    shot_volume_per90 = _numeric(out, "sh/90", 0.0).mask(_numeric(out, "sh/90", 0.0).eq(0.0), shots_per_90_live).clip(lower=0.0, upper=6.0)
    creative_actions_per90 = _numeric(out, "sca90", 0.0).clip(lower=0.0, upper=10.0)
    save_volume_per90 = saves_per_90_live.clip(lower=0.0, upper=8.0)

    role_boost = pd.Series(0.95, index=out.index, dtype="float64")
    role_boost = role_boost.mask(role_bucket.eq("attacker"), 1.06)
    role_boost = role_boost.mask(role_bucket.eq("midfielder"), 1.03)
    role_boost = role_boost.mask(role_bucket.eq("defender"), 0.98)
    role_boost = role_boost.mask(role_bucket.eq("goalkeeper"), 0.94)

    output_signal = (
        goal_involvement_per90.clip(upper=1.0) * 0.70
        + shot_volume_per90.div(4.0).clip(upper=1.0) * 0.20
        + creative_actions_per90.div(5.0).clip(upper=1.0) * 0.10
    )

    defensive_signal = (
        _numeric(out, "tkl+int", 0.0).div(season_90s.replace(0, np.nan)).fillna(0.0).clip(upper=8.0).div(8.0)
        + _numeric(out, "blocks", 0.0).div(season_90s.replace(0, np.nan)).fillna(0.0).clip(upper=4.0).div(8.0)
    )

    player_importance_score = (
        minutes_share * 0.48
        + starter_ratio * 0.24
        + output_signal * 0.18
        + defensive_signal.clip(lower=0.0, upper=1.0) * 0.10
    ) * role_boost
    player_importance_score = player_importance_score.clip(lower=0.0, upper=1.0)

    days_since = _numeric(out, "days_since_last_injury", 365.0).clip(lower=0.0, upper=365.0)
    previous_injuries = _numeric(out, "previous_injuries", 0.0).clip(lower=0.0, upper=20.0)
    total_days_lost = _numeric(out, "total_days_lost", 0.0).clip(lower=0.0, upper=1200.0)
    injury_prone = _numeric(out, "is_injury_prone", 0.0).clip(lower=0.0, upper=1.0)

    recent_injury_pressure = (1.0 - (days_since.clip(upper=180.0) / 180.0)).clip(lower=0.0, upper=1.0)
    injury_burden_index = (
        previous_injuries.div(10.0).clip(upper=1.0) * 0.42
        + total_days_lost.div(500.0).clip(upper=1.0) * 0.20
        + injury_prone * 0.18
        + recent_injury_pressure * 0.20
    ).clip(lower=0.0, upper=1.0)

    out["minutes_share"] = minutes_share.round(4)
    out["starter_ratio"] = starter_ratio.round(4)
    out["goal_involvement_per90"] = goal_involvement_per90.round(4)
    out["shot_volume_per90"] = shot_volume_per90.round(4)
    out["creative_actions_per90"] = creative_actions_per90.round(4)
    out["save_volume_per90"] = save_volume_per90.round(4)
    out["player_importance_score"] = player_importance_score.round(4)
    out["days_since_last_injury_capped"] = days_since.round(1)
    out["recent_injury_pressure"] = recent_injury_pressure.round(4)
    out["injury_burden_index"] = injury_burden_index.round(4)

    # Safe defaults for live inference rows that do not carry matchup data yet.
    for col in [
        "opp_form_avg_last_5",
        "opp_goal_diff_last_5",
        "opp_win_ratio_last_5",
        "h2h_matches_played",
        "h2h_win_ratio",
        "h2h_points_per_match",
        "fixture_edge_score",
    ]:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    return out


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
    """

    # -------------------------
    # 1. Positive samples
    # -------------------------
    final_positive = final_df.copy()
    final_positive["injury_label"] = 1

    if include_date and "injury_datetime" in final_positive.columns:
        final_positive["event_date"] = pd.to_datetime(final_positive["injury_datetime"])
    elif include_date and "date_of_injury" in final_positive.columns:
        final_positive["event_date"] = pd.to_datetime(final_positive["date_of_injury"])

    # -------------------------
    # 1b. Compute player injury history features
    # -------------------------
    if "severity_days" not in final_positive.columns:
        if "date_of_return" in final_positive.columns and "date_of_injury" in final_positive.columns:
            final_positive["severity_days"] = (
                pd.to_datetime(final_positive["date_of_return"]) -
                pd.to_datetime(final_positive["date_of_injury"])
            ).dt.days

    if "severity_days" in final_positive.columns and "name" in final_positive.columns:
        player_stats = final_positive.groupby("name").agg(
            player_injury_count=("severity_days", "count"),
            player_avg_severity=("severity_days", "mean"),
            player_worst_injury=("severity_days", "max"),
            player_severity_std=("severity_days", "std"),
            total_days_lost=("severity_days", "sum"),
        ).reset_index()

        player_stats["player_severity_std"] = player_stats["player_severity_std"].fillna(0)
        player_stats["is_injury_prone"] = (player_stats["player_injury_count"] >= 3).astype(int)

        for col in [
            "player_injury_count",
            "player_avg_severity",
            "player_worst_injury",
            "player_severity_std",
            "is_injury_prone",
            "total_days_lost",
        ]:
            if col in final_positive.columns:
                final_positive = final_positive.drop(columns=[col])

        final_positive = final_positive.merge(player_stats, on="name", how="left")

        print("Added player injury history features to classification dataset")
        print(f"  Injury-prone players (3+ injuries): {player_stats['is_injury_prone'].sum()}")

    feature_cols = [
        # Player + static info
        "player_team", "position", "age", "fifa_rating", "league",

        # Match workload + congestion
        "matches_last_7", "matches_last_14", "matches_last_30",
        "rest_days_before_injury", "avg_rest_last_5",

        # Performance rolling windows
        "goals_for_last_5", "goals_against_last_5",
        "goal_diff_last_5", "avg_goal_diff_last_5",
        "form_last_5", "form_avg_last_5",
        "win_ratio_last_5", "win_streak", "loss_streak",

        # Opponent / fixture context
        "opp_form_avg_last_5", "opp_goal_diff_last_5", "opp_win_ratio_last_5",
        "h2h_matches_played", "h2h_win_ratio", "h2h_points_per_match",
        "fixture_edge_score",

        # Injury history
        "previous_injuries", "days_since_last_injury_capped", "recent_injury_pressure",
        "injury_burden_index",
        "player_injury_count", "player_avg_severity",
        "player_worst_injury", "player_severity_std",
        "is_injury_prone", "total_days_lost",

        # Player usage / importance
        "minutes_share", "starter_ratio", "goal_involvement_per90",
        "shot_volume_per90", "creative_actions_per90", "player_importance_score",

        # Workload analytics
        "acute_load", "chronic_load", "acwr",
        "monotony", "strain", "fatigue_index",
        "workload_slope", "spike_flag",

        # Target
        "injury_label"
    ]

    if include_date:
        feature_cols.append("event_date")

    # -------------------------
    # 2. Player-level negative sampling
    # -------------------------
    player_info_cols = [
        "name", "player_team", "position", "age", "fifa_rating", "league",
        "previous_injuries", "days_since_last_injury",
        "player_injury_count", "player_avg_severity",
        "player_worst_injury", "player_severity_std",
        "is_injury_prone", "total_days_lost",
        "min", "mp", "starts", "90s", "gls", "ast", "sh/90", "sca90", "pos",
    ]
    player_info_cols = [c for c in player_info_cols if c in final_positive.columns]

    player_info = (
        final_positive[player_info_cols]
        .drop_duplicates(subset=["name"], keep="last")
    )

    valid_teams = final_positive["player_team"].unique()
    negative_samples = negative_samples[
        negative_samples["player_team"].isin(valid_teams)
    ].copy()

    # -------------------------
    # 3. Create player-level negatives
    # -------------------------
    player_negatives = []
    for _, player in player_info.iterrows():
        team = player["player_team"]
        name = player["name"]

        team_matches = negative_samples[negative_samples["player_team"] == team].copy()
        if team_matches.empty:
            continue

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

        n_sample = min(20, len(team_matches))
        if len(team_matches) > n_sample:
            team_matches = team_matches.copy()
            team_matches["_year"] = team_matches["match_date_dt"].dt.year
            years = sorted(team_matches["_year"].unique())
            per_year = max(1, n_sample // len(years))
            parts = []
            for yr in years:
                yr_rows = team_matches[team_matches["_year"] == yr]
                take = min(per_year, len(yr_rows))
                parts.append(yr_rows.sample(n=take, random_state=hash(name) % 2**31))
            sampled = pd.concat(parts).head(n_sample).drop(columns=["_year"])
            team_matches = team_matches.drop(columns=["_year"])
        else:
            sampled = team_matches

        for col in player_info_cols:
            if col != "player_team":
                sampled[col] = player[col]

        player_negatives.append(sampled)

    negative_samples = pd.concat(player_negatives, ignore_index=True)

    if "match_date_dt" in negative_samples.columns:
        negative_samples = negative_samples.drop(columns=["match_date_dt"])

    print(f"  Player-level negatives: {len(negative_samples)} samples for {len(player_negatives)} players")

    if "rest_days" in negative_samples.columns:
        negative_samples["rest_days_before_injury"] = negative_samples["rest_days"]

    negative_samples["injury_label"] = 0

    if include_date and "match_date" in negative_samples.columns:
        negative_samples["event_date"] = pd.to_datetime(negative_samples["match_date"])

    # -------------------------
    # 4. Combine positive + negative datasets
    # -------------------------
    support_cols = [
        "min", "mp", "starts", "90s", "gls", "ast", "sh/90", "sca90", "pos",
        "goals", "assists", "goals_per_90", "assists_per_90", "shots_per_90", "saves_per_90",
        "minutes_played", "appearances",
        "opp_form_avg_last_5", "opp_goal_diff_last_5", "opp_win_ratio_last_5",
        "h2h_matches_played", "h2h_win_ratio", "h2h_points_per_match", "fixture_edge_score",
        "days_since_last_injury", "previous_injuries", "total_days_lost", "is_injury_prone",
    ]
    raw_cols_to_use = [
        c for c in set(feature_cols + support_cols)
        if c in final_positive.columns and c in negative_samples.columns
    ]

    final_df_out = pd.concat([
        final_positive[raw_cols_to_use],
        negative_samples[raw_cols_to_use]
    ], ignore_index=True)

    final_df_out = add_contextual_classification_features(final_df_out)
    final_df_out = final_df_out[[c for c in feature_cols if c in final_df_out.columns]]

    if "event_date" in final_df_out.columns:
        final_df_out = final_df_out.sort_values("event_date").reset_index(drop=True)

    return final_df_out
