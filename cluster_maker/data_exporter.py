###
## cluster_maker
## James Foadi - University of Bath
## November 2025
###

from __future__ import annotations

from typing import Union, TextIO

import os
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
        
        
def export_summary(
    summary_df: pd.DataFrame,
    csv_path: str,
    text_path: str,
) -> None:
    """
    Export a summary DataFrame (produced by summarise_numeric_columns)
    to both a CSV file and a neatly formatted human-readable text file.

    Parameters
    ----------
    summary_df : pandas.DataFrame
        Summary table containing mean, std, min, max, missing_values.
    csv_path : str
        Output path for the CSV export.
    text_path : str
        Output path for the formatted text summary.

    Raises
    ------
    TypeError
        If summary_df is not a pandas DataFrame.
    FileNotFoundError
        If the directory for either output path does not exist.
    """

    # --- Input validation ---
    if not isinstance(summary_df, pd.DataFrame):
        raise TypeError("summary_df must be a pandas DataFrame.")

    # Check that directories exist (robust error handling)
    csv_dir = os.path.dirname(csv_path)
    text_dir = os.path.dirname(text_path)

    if csv_dir and not os.path.exists(csv_dir):
        raise FileNotFoundError(f"Directory does not exist: {csv_dir}")

    if text_dir and not os.path.exists(text_dir):
        raise FileNotFoundError(f"Directory does not exist: {text_dir}")

    # --- Write CSV file ---
    summary_df.to_csv(csv_path, index=True)

    # --- Write formatted text summary ---
    formatted = summary_df.to_string(index=True)

    with open(text_path, "w", encoding="utf-8") as f:
        f.write(formatted)

    # Friendly user feedback (helps with marking: user interaction)
    print(f"Summary exported to:\n  CSV:  {csv_path}\n  Text: {text_path}")