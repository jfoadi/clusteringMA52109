###
## cluster_maker
## James Foadi - University of Bath
## November 2025
###

from __future__ import annotations

import pandas as pd
import numpy as np


######## Task 3a   ######################

def get_numeric_column_summary(data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean, standard deviation, min, max, and missing value count
    for all numeric columns in the input DataFrame.

    Non-numeric columns are ignored. Includes print statements for tracing.

    Parameters
    ----------
    data : pandas.DataFrame

    Returns
    -------
    summary_df : pandas.DataFrame
        A DataFrame where the index consists of the statistic names
        and columns are the original numeric features.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
        
    print("\n--- Starting Numeric Column Summary Calculation ---")
    print(f"Input DataFrame shape: {data.shape}")
    
    # 1. Filter for only numeric columns for robustness
    print("Step 1: Selecting only numeric columns...")
    numeric_data = data.select_dtypes(include=np.number)
    
    numeric_cols_found = list(numeric_data.columns)
    print(f"  -> Found {len(numeric_cols_found)} numeric column(s): {numeric_cols_found}")

    # Handle case where no numeric columns are found
    if numeric_data.empty:
        print("Warning: The input DataFrame contains no numeric columns. Returning empty summary.")
        return pd.DataFrame(index=['mean', 'std', 'min', 'max', 'missing_values'])

    # 2. Calculate standard descriptive statistics (excluding count)
    print("Step 2: Calculating mean, standard deviation, min, and max")
    stats_df = numeric_data.agg(
        ['mean', 'std', 'min', 'max']
    ).T
    
    # 3. Calculate the number of missing values (NaN count)
    print("Step 3: Calculating the count of missing (NaN) values for each column...")
    missing_values = numeric_data.isnull().sum()
    print(f"  -> Missing values calculated.")

    # 4. Add the 'missing_values' row to the statistics DataFrame
    stats_df['missing_values'] = missing_values

    # 5. Transpose the final result to meet the requirement:
    #    index = stats, columns = features
    summary_df = stats_df.T
    
    # Optional: Rename the index for clarity
    summary_df.index.name = "statistic"
    
    print(f"  -> Final Summary DataFrame shape: {summary_df.shape}")        
    print("\n--- Final Numeric Summary Results (Returned DataFrame) ---")
    print(summary_df.to_string())
    print("----------------------------------------------------------")
    
    print("--- Numeric Column Summary Calculation Complete ---")

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