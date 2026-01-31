import pandas as pd

def rename_final_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename columns in the final dataframe for clarity.
    """
    return df.rename(
        columns={
            "age_x": "age",
            "season_x": "season",
            "season_year_x": "season_year",
        }
    )