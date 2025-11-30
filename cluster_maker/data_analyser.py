###
## cluster_maker
## James Foadi - University of Bath
## November 2025
###

from __future__ import annotations

import pandas as pd

# function inputs a pandas DataFrame and outputs a pandas DataFrame
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
    return data.describe() # pandas function

# The function checks that the input is a DataFrame, and if it is, it returns the standard 
# descriptive statistics for each numeric column (count, mean, standard deviation, quartiles, 
# and min/max). It’s essentially a safe wrapper around data.describe() with type checking and 
# documentation.

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
# The function checks that the input is a pandas DataFrame, 
# and then computes and returns the correlation matrix for only numeric columns. 
# It serves as a safe and clean wrapper around data.corr() with type checking to ensure valid input.


# Created for Task 3a

def summarise_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a summary table for all numeric columns in the DataFrame.

    For each numeric column, compute:
    - mean
    - standard deviation
    - minimum
    - maximum
    - number of missing values

    Non-numeric columns are ignored but clearly reported in a warning.

    Returns
    -------
    summary_df : pandas.DataFrame
        A DataFrame where each row corresponds to a numeric column and
        each column contains one of the summary statistics.

    Notes
    -----
    - Robust to non-numeric columns.
    - Produces human-readable statistics.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    # Identify numeric and non-numeric columns
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    non_numeric_cols = [col for col in df.columns if col not in numeric_cols]

    if non_numeric_cols:
        # This is a gentle, user-friendly warning instead of raising an error.
        print(f"Warning: Non-numeric columns ignored: {non_numeric_cols}")

    # Compute requested statistics
    summary_data = {
        "mean": df[numeric_cols].mean(),
        "std": df[numeric_cols].std(),
        "min": df[numeric_cols].min(),
        "max": df[numeric_cols].max(),
        "missing_values": df[numeric_cols].isna().sum(),
    } # summary data dictionary

    # Construct a new DataFrame with these metrics
    summary_df = pd.DataFrame(summary_data)
    # turn dictionary into DataFrame

    return summary_df
# provides a clean, human-readable summary of all numeric columns within a DataFrame. 
# It automatically identifies numeric features, computes key descriptive statistics 
# (mean, standard deviation, minimum, maximum, and the number of missing values), 
# and organises these results into a tidy summary table. Non-numeric columns are safely ignored, 
# with a gentle warning to the user. This makes the function a convenient tool for quickly 
# understanding the structure and quality of numerical data in any dataset.