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

# File: cluster_maker/data_exporter.py

# ... (existing functions)

def export_summary_report(
    summary_df: pd.DataFrame,
    csv_path: str,
    text_path: str,
) -> None:
    """
    Exports the numeric summary DataFrame to both a CSV file and a neatly
    formatted human-readable text file.

    Parameters
    ----------
    summary_df : pandas.DataFrame
        The summary DataFrame (stats as index, columns as features).
    csv_path : str
        Path for the CSV output file.
    text_path : str
        Path for the human-readable text output file.
    """
    if not isinstance(summary_df, pd.DataFrame):
        raise TypeError("summary_df must be a pandas DataFrame.")

    # 1. Write to CSV file (reusing export_to_csv)
    # The index (stat names) must be included.
    export_to_csv(summary_df, csv_path, delimiter=",", include_index=True) 

    # 2. Write to human-readable text file (reusing export_formatted)
    # Transpose the data to get one feature per row for a neat summary (one line per column)
    formatted_data = summary_df.transpose()
    export_formatted(formatted_data, text_path, include_index=True) 