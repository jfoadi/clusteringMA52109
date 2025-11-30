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

def analyse_numeric_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Computes descriptive statistics for all numeric columns in a DataFrame,
    returning a summary including mean, standard deviation, minimum, maximum,
    and number of missing values (NaN count).

    Parameters
    ----------
    data : pandas.DataFrame
        The input DataFrame potentially containing numeric and non-numeric columns.

    Returns
    -------
    summary_df : pandas.DataFrame
        A DataFrame containing 'mean', 'std', 'min', 'max', and 'missing'
        for each numeric column. Non-numeric columns are ignored.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    # 1. Select only numeric columns
    numeric_df = data.select_dtypes(include=np.number)

    # 2. Handle case where no numeric columns exist
    if numeric_df.empty:
        return pd.DataFrame(columns=['mean', 'std', 'min', 'max', 'missing'])

    # 3. Compute required statistics directly
    print("Computing descriptive statistics for numeric features...")

    summary_df = pd.DataFrame({
        'mean': numeric_df.mean(),
        'std': numeric_df.std(),
        'min': numeric_df.min(),
        'max': numeric_df.max(),
        'missing': numeric_df.isna().sum()
    })

    print("Descriptive statistics computed successfully.")
    print("Summary: ")
    print(summary_df)

    return summary_df.round(4)