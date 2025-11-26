###
## cluster_maker
## James Foadi - University of Bath
## November 2025
###

from __future__ import annotations

import pandas as pd


def calculate_descriptive_statistics(data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute descriptive statistics for each numeric column in the DataFrame.

    Parameters
    ----------
    data : pandas.DataFrame

    Returns
    -------
    stats : pandas.DataFrame
        Result of `data.describe()` including count, mean, std, etc.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    return data.describe()


def calculate_correlation(data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the correlation matrix for numeric columns in the DataFrame.

    Parameters
    ----------
    data : pandas.DataFrame

    Returns
    -------
    corr : pandas.DataFrame
        Correlation matrix.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    return data.corr(numeric_only=True)

def numeric_summary(data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a numeric summary for all numeric columns in a DataFrame.
    For each numeric column, return:
        - mean
        - standard deviation
        - minimum
        - maximum
        - number of missing values

    Non-numeric columns are ignored.

    Parameters
    ----------
    data : pandas.DataFrame

    Returns
    -------
    summary_df : pandas.DataFrame
        DataFrame with one row per numeric column.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    numeric_cols = [
        col for col in data.columns
        if pd.api.types.is_numeric_dtype(data[col])
    ]

    if not numeric_cols:
        raise ValueError("DataFrame contains no numeric columns.")

    summary_records = []

    for col in numeric_cols:
        col_data = data[col]
        summary_records.append({
            "column": col,
            "mean": col_data.mean(),
            "std": col_data.std(),
            "min": col_data.min(),
            "max": col_data.max(),
            "missing_values": col_data.isna().sum(),
        })

    summary_df = pd.DataFrame(summary_records)
    return summary_df
