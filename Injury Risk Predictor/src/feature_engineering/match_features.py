import pandas as pd

# -------------------------------------------------------------
# 1. Convert home/away structure into unified team-level rows
# -------------------------------------------------------------
def build_team_match_frame(match_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert match dataset into a unified format where each row
    represents a team-game (home or away).
    """

    df = match_df.copy()
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")

    # Home rows
    home_df = df[[
        "match_date", "season_year",
        "home_team", "away_team", "homegoals", "awaygoals"
    ]].rename(columns={
        "home_team": "team",
        "away_team": "opp_team",
        "homegoals": "goals_for",
        "awaygoals": "goals_against",
    })
    home_df["venue"] = "home"

    # Away rows
    away_df = df[[
        "match_date", "season_year",
        "away_team", "home_team", "awaygoals", "homegoals"
    ]].rename(columns={
        "away_team": "team",
        "home_team": "opp_team",
        "awaygoals": "goals_for",
        "homegoals": "goals_against",
    })
    away_df["venue"] = "away"

    # Combine
    combined = pd.concat([home_df, away_df], ignore_index=True)

    # Ensure sorted for rolling operations
    combined = combined.sort_values(["team", "match_date"])

    return combined.reset_index(drop=True)


# -------------------------------------------------------------
# 2. Create rolling windows and match-derived performance features
# -------------------------------------------------------------
def add_match_features(team_matches_df: pd.DataFrame) -> pd.DataFrame:
    df = team_matches_df.copy()

    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df = df.sort_values(["team", "match_date"]).reset_index(drop=True)

    groups = df.groupby("team")
    df["dummy"] = 1

    # Rolling match counts — FIXED (always aligned to df index)
    df["matches_last_7"] = (
        groups.apply(lambda g: g.rolling("7D", on="match_date")["dummy"].count() - 1)
              .reset_index(level=0, drop=True)
              .reset_index(drop=True)
    )

    df["matches_last_14"] = (
        groups.apply(lambda g: g.rolling("14D", on="match_date")["dummy"].count() - 1)
              .reset_index(level=0, drop=True)
              .reset_index(drop=True)
    )

    df["matches_last_30"] = (
        groups.apply(lambda g: g.rolling("30D", on="match_date")["dummy"].count() - 1)
              .reset_index(level=0, drop=True)
              .reset_index(drop=True)
    )

    # Rest days — FIXED
    df["rest_days"] = (
        groups["match_date"].diff().dt.days
              .reset_index(level=0, drop=True)
              .reset_index(drop=True)
    )

    # Avg rest last 5 — FIXED
    df["avg_rest_last_5"] = (
        groups["rest_days"]
        .rolling(5, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
        .reset_index(drop=True)
    )

    # Points
    df["points"] = df.apply(
        lambda r: 3 if r["goals_for"] > r["goals_against"]
        else (1 if r["goals_for"] == r["goals_against"] else 0),
        axis=1
    )

    # Form
    df["form_last_5"] = (
        groups["points"].rolling(5, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
        .reset_index(drop=True)
    )

    df["form_avg_last_5"] = (
        groups["points"].rolling(5, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
        .reset_index(drop=True)
    )

    # Goals rolling
    df["goals_for_last_5"] = (
        groups["goals_for"].rolling(5, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
        .reset_index(drop=True)
    )

    df["goals_against_last_5"] = (
        groups["goals_against"].rolling(5, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
        .reset_index(drop=True)
    )

    df["goal_diff_last_5"] = df["goals_for_last_5"] - df["goals_against_last_5"]

    df["avg_goal_diff_last_5"] = (
        groups["goal_diff_last_5"].rolling(5, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
        .reset_index(drop=True)
    )

    # Streaks
    df["is_win"] = (df["points"] == 3).astype(int)
    df["is_loss"] = (df["points"] == 0).astype(int)

    df["win_ratio_last_5"] = (
        groups["is_win"]
        .rolling(5, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True))

    df["win_streak"] = (
        groups["is_win"].apply(lambda g: g.groupby((g != g.shift()).cumsum()).cumsum() * g)
        .reset_index(level=0, drop=True)
        .reset_index(drop=True)
    )

    df["loss_streak"] = (
        groups["is_loss"].apply(lambda g: g.groupby((g != g.shift()).cumsum()).cumsum() * g)
        .reset_index(level=0, drop=True)
        .reset_index(drop=True)
    )

    return df