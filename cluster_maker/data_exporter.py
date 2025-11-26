###
## cluster_maker
## James Foadi - University of Bath
## November 2025
###

from __future__ import annotations

from typing import Union, TextIO

import pandas as pd
import io


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

def export_summary_report(
    summary_df: pd.DataFrame,
    csv_filename: str,
    text_filename: str,
) -> None:
    """
    Exports a summary DataFrame to both a CSV file and a human-readable text file.

    Parameters
    ----------
    summary_df : pandas.DataFrame
        Summary statistics DataFrame created in part (a).
    csv_filename : str
        Output filename for the CSV report.
    text_filename : str
        Output filename for the formatted text report.
    """
    if not isinstance(summary_df, pd.DataFrame):
        raise TypeError("summary_df must be a pandas DataFrame.")

    # 1. Write to CSV file
    export_to_csv(summary_df, csv_filename, include_index=True)

    # 2. Write to formatted text file
    # Ensure the feature index is part of the string output for the human-readable format
    text_summary = io.StringIO()
    summary_df.to_string(buf=text_summary, header=True, index=True)
    
    with open(text_filename, "w", encoding="utf-8") as f:
        f.write("--- Summary Statistics Report ---\n\n")
        f.write(text_summary.getvalue())