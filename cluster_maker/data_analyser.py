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


def get_numeric_summary(data: pd.DataFrame) -> pd.DataFrame:
    """
    Get a summary of numeric columns in the DataFrame.
    Summary includes mean, min, max, standard deviation, and number of missing values
    Non-numeric columns are ignored.

    Parameters
    ----------
    data : pandas.DataFrame

    Returns
    -------
    summary : pandas.DataFrame
        A DataFrame with one row per numeric column.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    
    # Select only numeric columns
    numeric_data = data.select_dtypes(include='number')

    summary = pd.DataFrame({
        'mean': numeric_data.mean(),
        'min': numeric_data.min(),
        'max': numeric_data.max(), 
        'std': numeric_data.std(),
        'n_missing': numeric_data.isna().sum()
    })
    return summary
    
    