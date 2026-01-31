import pandas as pd


def merge_injury_and_stats(injury_df: pd.DataFrame, stats_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge injury + stats on:
        player_name, player_team, season_year
    """
    return injury_df.merge(
        stats_df,
        on=["player_name", "player_team", "season_year"],
        how="left",
        validate="many_to_one",
    )