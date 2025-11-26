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

#added this function to export numeric summary
def export_numeric_summary(
    summary_data: pd.DataFrame, csv_path: str, text_path: str
) -> None:
    """
    Export numeric summary to both CSV and a formatted text file.

    Parameters
    ----------
    summary_data : pandas.DataFrame
        The summary statistics DataFrame.
    csv_path : str
        Path to save the CSV file.
    text_path : str
        Path to save the formatted text file.
    """
    if not isinstance(summary_data, pd.DataFrame):
        raise TypeError("summary_data must be a pandas DataFrame.")

    # Export to CSV
    export_to_csv(summary_data, csv_path, include_index=True)

    # Export to text file
    with open(text_path, "w", encoding="utf-8") as f:
        f.write("=== Numeric Summary ===\n\n")
        export_formatted(summary_data, f, include_index=True)