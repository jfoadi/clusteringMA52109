###
## cluster_maker
## James Foadi - University of Bath
## November 2025
###

from __future__ import annotations

from typing import Union, TextIO

import pandas as pd


def export_to_csv(
    data: pd.DataFrame,
    filename: str,
    delimiter: str = ",",
    include_index: bool = False,
) -> None:
    """
    Export a DataFrame to CSV.

    Parameters
    ----------
    data : pandas.DataFrame
    filename : str
        Output filename.
    delimiter : str, default ","
    include_index : bool, default False
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    data.to_csv(filename, sep=delimiter, index=include_index)


def export_formatted(
    data: pd.DataFrame,
    file: Union[str, TextIO],
    include_index: bool = False,
) -> None:
    """
    Export a DataFrame as a formatted text table.

    Parameters
    ----------
    data : pandas.DataFrame
    file : str or file-like
        Filename or open file handle.
    include_index : bool, default False
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    table_str = data.to_string(index=include_index)

    if isinstance(file, str):
        with open(file, "w", encoding="utf-8") as f:
            f.write(table_str)
    else:
        file.write(table_str)


def export_summary_csv(
    summary_df: pd.DataFrame,
    filename: str,
) -> None:
    """
    Export a summary DataFrame to a CSV file.

    Parameters
    ----------
    summary_df : pandas.DataFrame
        Summary DataFrame (e.g., from numeric_summary).
    filename : str
        Output CSV filename.

    Raises
    ------
    TypeError
        If summary_df is not a pandas DataFrame.
    """
    if not isinstance(summary_df, pd.DataFrame):
        raise TypeError("summary_df must be a pandas DataFrame.")
    summary_df.to_csv(filename, index=True)


def export_summary_text(
    summary_df: pd.DataFrame,
    filename: str,
) -> None:
    """
    Export a summary DataFrame as a formatted human-readable text file.

    Parameters
    ----------
    summary_df : pandas.DataFrame
        Summary DataFrame (e.g., from numeric_summary).
    filename : str
        Output text filename.

    Raises
    ------
    TypeError
        If summary_df is not a pandas DataFrame.
    """
    if not isinstance(summary_df, pd.DataFrame):
        raise TypeError("summary_df must be a pandas DataFrame.")

    # Format the summary as a nicely formatted text table
    formatted_text = "Numeric Summary Statistics\n"
    formatted_text += "=" * 70 + "\n\n"

    for col_name in summary_df.index:
        formatted_text += f"Column: {col_name}\n"
        formatted_text += "-" * 70 + "\n"
        for stat_name in summary_df.columns:
            value = summary_df.loc[col_name, stat_name]
            formatted_text += f"  {stat_name:.<30} {value}\n"
        formatted_text += "\n"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(formatted_text)