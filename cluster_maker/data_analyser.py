###
## cluster_maker
## James Foadi - University of Bath
## November 2025
###

from __future__ import annotations

import numpy as np
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

def calculate_comprehensive_statistics(data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute comprehensive descriptive statistics for numeric columns.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Input DataFrame with mixed column types
    
    Returns
    -------
    summary_df : pandas.DataFrame
        DataFrame containing for each numeric column:
        - mean, standard deviation, minimum, maximum, and count of missing values
        Non-numeric columns are ignored.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    
    # Select only numeric columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        # Return empty DataFrame with expected columns if no numeric data
        return pd.DataFrame(columns=['mean', 'std', 'min', 'max', 'missing_count'])
    
    # Calculate statistics for each numeric column
    stats_dict = {}
    for col in numeric_cols:
        col_data = data[col]
        stats_dict[col] = {
            'mean': col_data.mean(),
            'std': col_data.std(),
            'min': col_data.min(),
            'max': col_data.max(),
            'missing_count': col_data.isnull().sum()
        }
    
    # Convert to DataFrame and transpose so columns become rows
    summary_df = pd.DataFrame(stats_dict).T
    summary_df = summary_df[['mean', 'std', 'min', 'max', 'missing_count']]
    
    return summary_df    