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
    Compute summary statistics for numeric columns in a DataFrame.

    Parameters
    ----------
    data : pandas.DataFrame
        Input DataFrame that may contain numeric and non-numeric columns.

    Returns
    -------
    summary : pandas.DataFrame
        DataFrame with one row per numeric column containing:
        - 'mean': arithmetic mean
        - 'std': standard deviation
        - 'min': minimum value
        - 'max': maximum value
        - 'missing_count': number of missing values (NaN)

    Raises
    ------
    TypeError
        If data is not a pandas DataFrame.

    Notes
    -----
    Non-numeric columns are ignored.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    # Identify numeric columns
    numeric_cols = [
        col for col in data.columns
        if pd.api.types.is_numeric_dtype(data[col])
    ]

    # Compute statistics for each numeric column
    summary_data = []
    for col in numeric_cols:
        summary_data.append({
            'column': col,
            'mean': data[col].mean(),
            'std': data[col].std(),
            'min': data[col].min(),
            'max': data[col].max(),
            'missing_count': data[col].isna().sum(),
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df.set_index('column', inplace=True)
    return summary_df