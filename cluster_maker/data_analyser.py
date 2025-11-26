###
## cluster_maker
## James Foadi - University of Bath
## November 2025
###

from __future__ import annotations
import pandas as pd

def calculate_descriptive_statistics(data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute descriptive statistics for numeric columns, including
    mean, std, min, max, and missing value counts.

    Parameters
    ----------
    data : pandas.DataFrame
        Input dataframe.

    Returns
    -------
    stats : pandas.DataFrame
        A dataframe with statistics as columns and features as rows.
    """
    # Robustness check: Ensure input is a DataFrame
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    
    # Robustness check: Select only numeric columns
    numeric_df = data.select_dtypes(include=['number'])
    
    # Handle case where no numeric columns exist
    if numeric_df.empty:
        return pd.DataFrame()

    # Calculate standard stats (mean, std, min, max)
    # Transpose (.T) so features are rows, stats are columns
    stats = numeric_df.agg(['mean', 'std', 'min', 'max']).T
    
    # Calculate missing values count and add it as a new column
    missing = numeric_df.isna().sum()
    stats['missing_values'] = missing
    
    return stats

def calculate_correlation(data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the correlation matrix for numeric columns in the DataFrame.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    
    # numeric_only=True ensures robustness against string columns
    return data.corr(numeric_only=True)