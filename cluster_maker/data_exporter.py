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

def export_summary(summary_df: pd.DataFrame, csv_file: str, txt_file: str) -> None:
    """
    Export a summary DataFrame to both CSV and a human-readable text file.

    Parameters
    ----------
    summary_df : pandas.DataFrame
        The summary DataFrame from calculate_descriptive_statistics.
    csv_file : str
        Path to CSV output file.
    txt_file : str
        Path to human-readable text output file.
    """
    if not isinstance(summary_df, pd.DataFrame):
        raise TypeError("summary_df must be a pandas DataFrame.")

    # Export to CSV
    summary_df.to_csv(csv_file, sep=",", index=True)

    # Export to human-readable text
    lines = []
    for col in summary_df.columns:
        values = ", ".join(f"{v}" for v in summary_df[col])
        lines.append(f"{col}: {values}\n")

    with open(txt_file, "w", encoding="utf-8") as f:
        f.writelines(lines)
