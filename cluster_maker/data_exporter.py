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
    if not isinstance(filename, str):
        raise ValueError("filename must be a string.")

    try:
        data.to_csv(filename, sep=delimiter, index=include_index)
    except Exception as e:
        raise IOError(f"Failed to write CSV file '{filename}': {e}")


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

    try:
        if isinstance(file, str):
            with open(file, "w", encoding="utf-8") as f:
                f.write(table_str)
        else:
            file.write(table_str)
    except Exception as e:
        raise IOError(f"Failed to write formatted text to '{file}': {e}")


def export_summary_files(
    data: pd.DataFrame,
    csv_path: str,
    txt_path: str
) -> None:
    """
    Export a summary DataFrame to both CSV and formatted text.

    Parameters
    ----------
    data : pandas.DataFrame
    csv_path : str
        Output CSV filename.
    txt_path : str
        Output text filename.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    
    if not isinstance(csv_path, str) or not isinstance(txt_path, str):
        raise ValueError("csv_path and txt_path must be strings.")
    
    # Export to CSV
    data.to_csv(csv_path, index=True)

    # Export to formatted text
    formatted = data.to_string()
    
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(formatted)