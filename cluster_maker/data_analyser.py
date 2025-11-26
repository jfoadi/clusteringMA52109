###
## cluster_maker
## James Foadi - University of Bath
## November 2025
###

from __future__ import annotations

import pandas as pd
import numpy as np # <-- NEW IMPORT REQUIRED for .select_dtypes and np.number

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

def summarise_numeric_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Computes a summary of key statistics and missing values for all numeric columns.

    Parameters
    ----------
    data : pandas.DataFrame
        The input data. Non-numeric columns are ignored.

    Returns
    -------
    summary_df : pandas.DataFrame
        DataFrame with statistics (mean, std, min, max, count_missing) as index
        and numeric column names as columns.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    # Select only numeric columns (robustness to non-numeric)
    numeric_data = data.select_dtypes(include=[np.number])
    if numeric_data.empty:
        # Return an empty DataFrame if no numeric columns are found
        return pd.DataFrame()

    # 1. Compute standard descriptive statistics (index includes mean, std, min, max)
    stats_df = numeric_data.describe()

    # 2. Compute the count of missing values
    missing_counts = numeric_data.isna().sum()
    missing_series = pd.Series(missing_counts, name='count_missing')

    # 3. Select required stats and append the missing count row
    required_stats = stats_df.loc[['mean', 'std', 'min', 'max']]

    # Transpose the missing series and concatenate it as a new row
    summary_df = pd.concat([required_stats, missing_series.to_frame().T])

    # Ensure the final order of statistics is as implied in the prompt
    return summary_df.loc[['mean', 'std', 'min', 'max', 'count_missing']]