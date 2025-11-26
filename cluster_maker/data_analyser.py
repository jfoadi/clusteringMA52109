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

def column_summary(data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a numeric summary for each numeric column in a DataFrame.

    For each numeric column, the summary includes:
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
        DataFrame where rows correspond to numeric columns and columns
        contain the calculated summary statistics.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    numeric_cols = data.select_dtypes(include="number").columns

    summary = {
        "mean": data[numeric_cols].mean(),
        "std": data[numeric_cols].std(),
        "min": data[numeric_cols].min(),
        "max": data[numeric_cols].max(),
        "n_missing": data[numeric_cols].isna().sum(),
    }

    summary_df = pd.DataFrame(summary)
    return summary_df
