import pandas as pd


def rename_final_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean up merge artifact columns in the final dataframe.

    When merging DataFrames with overlapping column names, pandas creates
    suffixes like _x and _y. This function:
    - Renames _x columns to clean names (age_x → age)
    - Drops _y columns (they're duplicates)

    Args:
        df: DataFrame with potential merge artifacts

    Returns:
        DataFrame with clean column names
    """
    df = df.copy()

    # Rename _x columns to clean names
    rename_map = {
        "age_x": "age",
        "season_x": "season",
        "season_year_x": "season_year",
    }

    # Only rename columns that exist
    rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
    if rename_map:
        df = df.rename(columns=rename_map)

    # Drop _y columns (duplicates from merge)
    y_cols = [c for c in df.columns if c.endswith("_y")]
    if y_cols:
        df = df.drop(columns=y_cols)

    return df