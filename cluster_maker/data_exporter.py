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


def export_summary(
    summary: pd.DataFrame,
    csv_path: str,
    text_path: str,
) -> None:
    """
    Export a numeric summary DataFrame to CSV and a human-readable text file.

    Parameters
    ----------
    summary : pandas.DataFrame
        Summary DataFrame (for example, with rows indexed by column name and
        columns like 'mean', 'std', 'min', 'max', 'n_missing').
    csv_path : str
        Path to write the CSV file to. The containing directory must exist.
    text_path : str
        Path to write a human-readable text summary. The containing directory
        must exist.

    Raises
    ------
    TypeError
        If `summary` is not a pandas DataFrame.
    FileNotFoundError
        If the directory for `csv_path` or `text_path` does not exist.
    """
    if not isinstance(summary, pd.DataFrame):
        raise TypeError("summary must be a pandas DataFrame.")

    # Ensure target directories exist
    csv_dir = os.path.dirname(os.path.abspath(csv_path)) or "."
    text_dir = os.path.dirname(os.path.abspath(text_path)) or "."
    if csv_dir and not os.path.exists(csv_dir):
        raise FileNotFoundError(f"Directory for csv_path does not exist: {csv_dir}")
    if text_dir and not os.path.exists(text_dir):
        raise FileNotFoundError(f"Directory for text_path does not exist: {text_dir}")

    # Write CSV using existing helper
    export_to_csv(summary, csv_path, delimiter=",", include_index=True)

    # Build a neat human-readable summary: one line per index (row)
    lines = []
    for idx, row in summary.iterrows():
        # Format numeric values to reasonable precision where possible
        parts = []
        for col in summary.columns:
            val = row[col]
            # Prefer concise formatting for floats
            if pd.api.types.is_float_dtype(type(val)):
                try:
                    part = f"{col}={float(val):.6g}"
                except Exception:
                    part = f"{col}={val}"
            else:
                part = f"{col}={val}"
            parts.append(part) 
        lines.append(f"{idx}: " + ", ".join(parts))

    with open(text_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

 