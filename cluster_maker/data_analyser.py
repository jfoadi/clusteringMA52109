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

def calculate_summary_stats(data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute key summary statistics (mean, std, min, max, missing count)
    for each numeric column in the DataFrame.

    Parameters
    ----------
    data : pandas.DataFrame

    Returns
    -------
    stats_df : pandas.DataFrame
        DataFrame with statistics as columns and numeric features as index.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    # Select only numeric columns
    numeric_data = data.select_dtypes(include=np.number)

    if numeric_data.empty:
        # Robustness: return an empty DataFrame or report clearly if no numeric columns found
        print("Warning: No numeric columns found for summary statistics.")
        return pd.DataFrame()

    # Calculate basic descriptive statistics (includes mean, std, min, max)
    stats = numeric_data.agg([
        'mean', 
        'std', 
        'min', 
        'max'
    ])

    # Calculate the number of missing values (NaN count)
    missing_count = numeric_data.isnull().sum()
    
    # Add missing count as a new row (transpose needed for alignment)
    stats_df = stats.T
    stats_df['missing_count'] = missing_count.values
    
    # Rename index to 'feature' for clarity
    stats_df.index.name = 'feature'
    
    # Optional: Reorder columns for presentation
    stats_df = stats_df[['mean', 'std', 'min', 'max', 'missing_count']]

    print("Descriptive statistics calculated successfully.")
    print(f"Mean values:\n{stats_df['mean']}")
    print(f"Standard deviations:\n{stats_df['std']}")
    print(f"Missing value counts:\n{stats_df['missing_count']}")
    print(f"Minimum values:\n{stats_df['min']}")
    print(f"Maximum values:\n{stats_df['max']}")
    
    return stats_df