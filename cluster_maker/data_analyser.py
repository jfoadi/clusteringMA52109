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

# Task 3a) Added new function
def calculate_column_statistics(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate comprehensive statistics for each numeric column in the DataFrame.

    Parameters
    ----------
    data : pandas.DataFrame
        Input DataFrame containing numeric columns to analyze.

    Returns
    -------
    stats_df : pandas.DataFrame
        DataFrame with statistics for each numeric column:
        - mean: Arithmetic mean of the column
        - std: Standard deviation
        - min: Minimum value
        - max: Maximum value
        - missing: Number of missing (NaN) values
        Non-numeric columns are automatically excluded.
    """
    print("Starting column statistics analysis...")
    
    if not isinstance(data, pd.DataFrame):
        print("ERROR: Input is not a pandas DataFrame.")
        raise TypeError("data must be a pandas DataFrame.")
    
    print(f"Input DataFrame shape: {data.shape}")
    print(f"Columns available: {list(data.columns)}")
    
    # Identify numeric columns only
    numeric_cols = [
        col for col in data.columns
        if pd.api.types.is_numeric_dtype(data[col])
    ]
    
    non_numeric_cols = [col for col in data.columns if col not in numeric_cols]
    
    if non_numeric_cols:
        print(f"Non-numeric columns detected and excluded: {non_numeric_cols}")
    
    if not numeric_cols:
        print("ERROR: No numeric columns found for analysis.")
        raise ValueError("No numeric columns found in the input DataFrame.")
    
    print(f"Found {len(numeric_cols)} numeric column(s) for analysis: {numeric_cols}")
    print("Computing statistics for each numeric column...")
    
    # Initialize dictionary to store statistics
    stats_data = {}
    
    for col in numeric_cols:
        column_data = data[col]
        print(f"   Analyzing column: '{col}'")
        
        stats_data[col] = {
            'mean': column_data.mean(),
            'std': column_data.std(),
            'min': column_data.min(),
            'max': column_data.max(),
            'missing': column_data.isna().sum()
        }
    
    # Create DataFrame with statistics
    stats_df = pd.DataFrame.from_dict(stats_data, orient='index')
    
    # Reorder columns for better presentation
    stats_df = stats_df[['mean', 'std', 'min', 'max', 'missing']]
    
    print("Statistics computation completed successfully!")
    print(f"Generated statistics for {len(numeric_cols)} column(s)")
    print("Summary of results:")
    print(f"   - Columns analyzed: {numeric_cols}")
    print(f"   - Statistics computed: mean, standard deviation, min, max, missing values")
    print(f"   - Output DataFrame shape: {stats_df.shape}")
    print(" Task 3a: Column statistics function completed successfully!")
    print("-" * 60)
    
    return stats_df