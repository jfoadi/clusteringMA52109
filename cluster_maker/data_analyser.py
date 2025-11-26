###
## cluster_maker
## James Foadi - University of Bath
## November 2025
###

from __future__ import annotations

import pandas as pd
import numpy as np

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

def calculate_extended_statistics(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate specific stats (mean, std, min, max, n_missing) for numeric columns.
    Ignores non-numeric columns.

    Parameters
    ----------
    data : pandas.DataFrame

    Returns
    -------
    summary : pandas.DataFrame
        Index corresponds to column names.
        Columns are ['mean', 'std', 'min', 'max', 'n_missing'].
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    # Select only numeric columns
    numeric_df = data.select_dtypes(include=[np.number])

    # If no numeric columns exist, return an empty DataFrame with the correct columns
    if numeric_df.empty:
        return pd.DataFrame(columns=["mean", "std", "min", "max", "n_missing"])

    # Calculate standard stats
    stats = numeric_df.agg(["mean", "std", "min", "max"]).transpose()
    
    # Calculate missing values
    missing = numeric_df.isna().sum()
    
    # Add missing count to the stats DataFrame
    stats["n_missing"] = missing

    return stats