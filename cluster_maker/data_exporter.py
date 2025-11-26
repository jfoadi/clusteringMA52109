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

def export_numeric_summary(summary_df: pd.DataFrame, csv_path: str, txt_path: str) -> None:
    """
    Export a numeric summary (created by numeric_summary) to:
      - a CSV file
      - a human-readable formatted text file

    Parameters
    ----------
    summary_df : pandas.DataFrame
        Summary table from numeric_summary().
    csv_path : str
        Path to save CSV file.
    txt_path : str
        Path to save text summary file.
    """

    if not isinstance(summary_df, pd.DataFrame):
        raise TypeError("summary_df must be a pandas DataFrame.")

    # Export CSV
    summary_df.to_csv(csv_path, index=False)

    # Export formatted text
    lines = []
    for _, row in summary_df.iterrows():
        line = (
            f"Column '{row['column']}': "
            f"mean={row['mean']:.3f}, std={row['std']:.3f}, "
            f"min={row['min']}, max={row['max']}, "
            f"missing={row['missing_values']}"
        )
        lines.append(line)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
