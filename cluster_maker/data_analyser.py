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

def numeric_statistics(data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute basic numeric statistics (mean, sd, min, max, no. of missing values) for each numeric column.

    Parameters
    ----------
    data : pandas.DataFrame

    Returns
    -------
    stats : pandas.DataFrame
        DataFrame with mean, sd, min, max, and no. of missing values for each numeric column.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    

    # Select only numeric columns (preserve order)
    numeric_cols = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])]

    cols = ["mean", "sd", "min", "max", "missing_values"]
    if not numeric_cols:
        return pd.DataFrame(columns=cols)

    subset = data[numeric_cols].apply(pd.to_numeric, errors="coerce")

    mean = subset.mean(skipna=True)
    sd = subset.std(skipna=True)
    minv = subset.min(skipna=True)
    maxv = subset.max(skipna=True)
    missing = subset.isna().sum()

    stats = pd.DataFrame(
        {
            "mean": mean,
            "sd": sd,
            "min": minv,
            "max": maxv,
            "missing_values": missing,
        }
    )

    # Ensure rows follow the original numeric column order and columns order
    stats = stats.reindex(index=numeric_cols)[cols]
    return stats
