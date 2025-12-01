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


# This function takes a pandas DataFrame and computes specified statistics
# for each numeric column, while ignoring non-numeric columns.

def column_statistics(data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute statistics for each numeric column in the DataFrame.

    Parameters
    ----------
    data : pandas.DataFrame

    Returns
    -------
    stats_df : pandas.DataFrame
        DataFrame with statistics: mean, std, min, max, missing_count.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    stats = {}
    for col in data.select_dtypes(include='number').columns:
        col_data = data[col]
        stats[col] = {
            'mean': col_data.mean(),
            'std': col_data.std(),
            'min': col_data.min(),
            'max': col_data.max(),
            'missing_count': col_data.isna().sum()
        }

    stats_df = pd.DataFrame.from_dict(stats, orient='index')
    return stats_df