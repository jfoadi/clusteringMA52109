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
     # --- Added code to ignore non-numeric columns ---
    numeric_cols = data.select_dtypes(include="number").columns
    non_numeric_cols = data.select_dtypes(exclude="number").columns
    if len(non_numeric_cols) > 0:
        print(f"Warning: Non-numeric columns ignored: {list(non_numeric_cols)}")
    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found in the input DataFrame.")

    data_numeric = data[numeric_cols]
    # --- End added code ---
    return data_numeric.describe()


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