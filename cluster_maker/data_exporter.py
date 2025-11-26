###
## cluster_maker
## James Foadi - University of Bath
## November 2025
###

from __future__ import annotations

from typing import Union, TextIO

import pandas as pd
from .data_analyser import column_statistics


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

##      b) In module "data_exporter.py", add a function that:
##         - takes the summary DataFrame created in part (a);
##         - writes it to a CSV file;
##         - writes, to a separate human-readable text file, a neatly
##           formatted summary (for example, one line per column).

def export_column_statistics(
    data: pd.DataFrame,
    csv_filename: str,
    text_filename: str,
) -> None:
    """
    Compute column statistics and export to CSV and formatted text file.

    Parameters
    ----------
    data : pandas.DataFrame
    csv_filename : str
        Output CSV filename.
    text_filename : str
        Output text filename.
    """
    stats_df = column_statistics(data)

    # Export to CSV
    export_to_csv(stats_df, csv_filename, delimiter=",", include_index=True)

    # Export to formatted text file
    export_formatted(stats_df, text_filename, include_index=True)