import pandas as pd


def merge_injuries_with_matches(injury_df: pd.DataFrame, match_df: pd.DataFrame) -> pd.DataFrame:
    """
    Safe merge_asof between injury records and team match features.
    Drops invalid rows, sorts keys, and handles missing team matches.
    """

    # Remove rows with no valid injury datetime
    injury_df = injury_df.dropna(subset=["injury_datetime"]).copy()

    # Ensure sorted for merge_asof (requirement)
    injury_df = injury_df.sort_values(["player_team", "injury_datetime"])
    match_df = match_df.sort_values(["team", "match_date"])

    results = []

    for team in injury_df["player_team"].unique():
        left_team = injury_df[injury_df["player_team"] == team].copy()
        right_team = match_df[match_df["team"] == team].copy()

        # If a team has no recorded matches, skip to avoid empty merge errors
        if right_team.empty:
            left_team["match_date"] = pd.NaT
            results.append(left_team)
            continue

        merged_team = pd.merge_asof(
            left_team.sort_values("injury_datetime"),
            right_team.sort_values("match_date"),
            left_on="injury_datetime",
            right_on="match_date",
            direction="backward",
            allow_exact_matches=True
        )

        results.append(merged_team)

    return pd.concat(results, ignore_index=True)