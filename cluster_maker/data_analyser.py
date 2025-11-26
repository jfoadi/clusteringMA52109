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

def summarise_numeric_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Return per-numeric-column summary statistics.

    For each numeric column returns:
      - mean
      - std (sample standard deviation)
      - min
      - max
      - n_missing (number of NA values)

    Non-numeric columns are ignored. If no numeric columns exist an empty
    DataFrame is returned.

    Parameters
    ----------
    data : pandas.DataFrame

    Returns
    -------
    pd.DataFrame
        Index: numeric column names
        Columns: ['mean','std','min','max','n_missing']
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    numeric = data.select_dtypes(include="number")
    if numeric.shape[1] == 0:
        # No numeric columns -- return empty DataFrame with expected columns
        return pd.DataFrame(columns=["mean", "std", "min", "max", "n_missing"])

    mean = numeric.mean(skipna=True)
    std = numeric.std(ddof=1, skipna=True)
    min_ = numeric.min(skipna=True)
    max_ = numeric.max(skipna=True)
    n_missing = numeric.isna().sum()

    stats = pd.DataFrame(
        {
            "mean": mean,
            "std": std,
            "min": min_,
            "max": max_,
            "n_missing": n_missing,
        }
    )

    return stats