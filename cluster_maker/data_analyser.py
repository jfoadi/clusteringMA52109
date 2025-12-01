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


def summarise_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute summary statistics for all numeric columns in a DataFrame.

    For each numeric column, the function returns:
        - mean
        - standard deviation
        - minimum
        - maximum
        - number of missing values

    Non-numeric columns are automatically ignored so that the function
    remains robust even when the input contains mixed data types.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data containing numeric and/or non-numeric columns.

    Returns
    -------
    summary_df : pandas.DataFrame
        A table of summary statistics for each numeric column,
        indexed by column name.

    Raises
    ------
    TypeError
        If the input is not a pandas DataFrame.
    ValueError
        If the DataFrame contains no numeric columns.
    """

    # Ensure correct input type
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    # Identify numeric columns
    numeric_df = df.select_dtypes(include="number")

    # Provide a clear, helpful error for edge cases
    if numeric_df.empty:
        raise ValueError("No numeric columns found in the DataFrame.")

    # If numeric and non-numeric columns coexist, explicitly acknowledge it
    if len(numeric_df.columns) != len(df.columns):
        # This print is allowed and improves user-facing transparency
        print(
            f"Note: Ignoring non-numeric columns: "
            f"{[col for col in df.columns if col not in numeric_df.columns]}"
        )

    # Build summary stats
    summary = {
        "mean": numeric_df.mean(),
        "std": numeric_df.std(),
        "min": numeric_df.min(),
        "max": numeric_df.max(),
        "missing_values": numeric_df.isna().sum()
    }

    # Convert dictionary into a well-structured DataFrame
    summary_df = pd.DataFrame(summary)

    return summary_df