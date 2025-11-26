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

def summarise_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise numeric columns of a DataFrame by computing:
      - mean
      - standard deviation
      - minimum
      - maximum
      - number of missing values

    Non-numeric columns are ignored.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.

    Returns
    -------
    summary : pandas.DataFrame
        Summary statistics with one row per numeric column and
        columns: ['mean', 'std', 'min', 'max', 'n_missing'].
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    # Extract only numeric columns
    numeric_df = df.select_dtypes(include="number")

    # Handle case where there are no numeric columns
    if numeric_df.empty:
        return pd.DataFrame(
            columns=["mean", "std", "min", "max", "n_missing"]
        )

    summary = pd.DataFrame(index=numeric_df.columns)

    summary["mean"] = numeric_df.mean()
    summary["std"] = numeric_df.std()
    summary["min"] = numeric_df.min()
    summary["max"] = numeric_df.max()
    summary["n_missing"] = numeric_df.isna().sum()

    return summary
