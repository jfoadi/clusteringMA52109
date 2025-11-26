###
## cluster_maker
## James Foadi - University of Bath
## November 2025
###

from __future__ import annotations

import pandas as pd

def summarize_numeric_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes summary statistics for all numeric columns in a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing data to be summarized.

    Returns
    -------
    summary_df : pandas.DataFrame
        A new DataFrame where each row corresponds to a numeric column
        from the input, and columns contain the mean, standard deviation,
        minimum, maximum, and count of missing values.
    """
    
    # 1. Select only numeric columns
    numeric_df = df.select_dtypes(include=['number'])

    # Check if any numeric columns were found
    if numeric_df.empty:
        # Return an empty DataFrame with the expected columns if no numeric data is found
        return pd.DataFrame(columns=['mean', 'std', 'min', 'max', 'missing_count'])

    # 2. Compute the required statistics using built-in Pandas methods
    
    # Mean and standard deviation (std)
    stats_mean = numeric_df.mean()
    stats_std = numeric_df.std()
    
    # Minimum (min) and Maximum (max)
    stats_min = numeric_df.min()
    stats_max = numeric_df.max()
    
    # Number of missing values (isna().sum())
    stats_missing = numeric_df.isna().sum()

    # 3. Combine the results into a single DataFrame
    
    # Create a dictionary where keys are the statistic names and values are the Series
    summary_data = {
        'mean': stats_mean,
        'std': stats_std,
        'min': stats_min,
        'max': stats_max,
        'missing_count': stats_missing,
    }
    
    # Combine the Series into a DataFrame. 
    # Pandas aligns them automatically by the column names (index of the Series).
    summary_df = pd.DataFrame(summary_data)
    
    # Rename the index to reflect that each row corresponds to a feature/column
    summary_df.index.name = 'feature'
    
    return summary_df






















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