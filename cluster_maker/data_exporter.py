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


def export_summary(summary_df: pd.DataFrame, csv_path: str, txt_path: str) -> None:
    """
    Export the summary DataFrame produced by column_summary() to:
        - a CSV file
        - a human-readable text file (one column summary per line)

    Parameters
    ----------
    summary_df : pandas.DataFrame
        Summary produced by data_analyser.column_summary()
    csv_path : str
        Path to save the CSV summary file
    txt_path : str
        Path to save the human-readable text summary
    """
    if not isinstance(summary_df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    # Save CSV version
    summary_df.to_csv(csv_path)

    # Save human-readable text version
    with open(txt_path, "w", encoding="utf-8") as f:
        for col in summary_df.index:
            stats = summary_df.loc[col]
            f.write(f"Column '{col}':\n")
            f.write(f"  mean       = {stats['mean']}\n")
            f.write(f"  std        = {stats['std']}\n")
            f.write(f"  min        = {stats['min']}\n")
            f.write(f"  max        = {stats['max']}\n")
            f.write(f"  missing    = {stats['n_missing']}\n")
            f.write("\n")
