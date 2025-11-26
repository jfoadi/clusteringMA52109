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

#added this function to calculate numeric summary
def calculate_numeric_summary(data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute summary statistics for numeric columns in the DataFrame.
    Includes mean, std, min, max, and number of missing values.

    Parameters
    ----------
    data : pandas.DataFrame

    Returns
    -------
    summary : pandas.DataFrame
        DataFrame with statistics as columns and feature names as index.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    # Select numeric columns
    numeric_df = data.select_dtypes(include=["number"])

    if numeric_df.empty:
        return pd.DataFrame()

    # Calculate statistics
    summary = numeric_df.agg(["mean", "std", "min", "max"]).T
    summary["missing_count"] = numeric_df.isnull().sum()

    return summary