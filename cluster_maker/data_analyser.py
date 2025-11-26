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


def summarize_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return summary stats for numeric columns: mean, std, min, max, missing count.
    Non-numeric columns are ignored.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=["column", "mean", "std", "min", "max", "missing"])

    rows = []
    for col in numeric_df.columns:
        s = numeric_df[col]
        rows.append({
            "column": col,
            "mean": float(s.mean()),
            "std": float(s.std(ddof=1)),
            "min": float(s.min()),
            "max": float(s.max()),
            "missing": int(s.isna().sum()),
        })

    return pd.DataFrame(rows, columns=["column", "mean", "std", "min", "max", "missing"])