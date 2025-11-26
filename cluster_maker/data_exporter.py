###
## cluster_maker
## James Foadi - University of Bath
## November 2025
###

from __future__ import annotations

from typing import Union, TextIO

import pandas as pd
import os

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


def export_summary(summary_df: pd.DataFrame, csv_path: str, text_path: str) -> None:
    """
    Write the summary DataFrame to CSV and a human-readable text file.
    """
    if not isinstance(summary_df, pd.DataFrame):
        raise TypeError("summary_df must be a pandas DataFrame.")
    if not isinstance(csv_path, str) or not isinstance(text_path, str):
        raise TypeError("csv_path and text_path must be strings.")

    out_dir_csv = os.path.dirname(csv_path) or "."
    out_dir_txt = os.path.dirname(text_path) or "."
    if not os.path.exists(out_dir_csv) or not os.path.exists(out_dir_txt):
        raise FileNotFoundError("Output directory does not exist.")

    summary_df.to_csv(csv_path, index=False)

    lines = []
    for _, row in summary_df.iterrows():
        lines.append(
            f"{row['column']}: mean={row['mean']:.6g}, "
            f"std={row['std']:.6g}, min={row['min']:.6g}, "
            f"max={row['max']:.6g}, missing={int(row['missing'])}"
        )

    with open(text_path, "w", encoding="utf-8") as f:
        f.write("Numeric column summary\n")
        f.write("----------------------\n")
        for line in lines:
            f.write(line + "\n")
