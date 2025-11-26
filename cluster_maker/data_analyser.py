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


def column_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute summary statistics for numeric columns in a DataFrame.
    Returns a new DataFrame with one row per numeric column, containing:
    mean, std, min, max, and number of missing values.

    Non-numeric columns are ignored.
    """

    # Select numeric columns only
    numeric_df = df.select_dtypes(include="number")

    summary = {
        "mean": numeric_df.mean(),
        "std": numeric_df.std(),
        "min": numeric_df.min(),
        "max": numeric_df.max(),
        "n_missing": numeric_df.isna().sum(),
    }

    # Convert dict of Series to DataFrame
    summary_df = pd.DataFrame(summary)

    return summary_df