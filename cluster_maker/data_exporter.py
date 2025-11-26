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
    include_index: bool = True,
) -> None:
    """
    Export a DataFrame to CSV.

    Parameters
    ----------
    data : pandas.DataFrame
    filename : str
    delimiter : str, default ","
    include_index : bool, default True
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    
    try:
        data.to_csv(filename, sep=delimiter, index=include_index)
    except OSError as e:
        print(f"Error exporting to CSV: {e}")


def export_formatted(
    data: pd.DataFrame,
    file: Union[str, TextIO],
    include_index: bool = True,
) -> None:
    """
    Export a DataFrame as a formatted text table.

    Parameters
    ----------
    data : pandas.DataFrame
    file : str or file-like object
        Filename string or open file handle.
    include_index : bool, default True
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    # Convert DataFrame to a pretty string format
    table_str = data.to_string(index=include_index)
    
    header = "=== Data Analysis Summary ===\n\n"
    footer = "\n\n============================="

    # Logic to handle both string filenames and open file objects
    if isinstance(file, str):
        try:
            with open(file, 'w') as f:
                f.write(header)
                f.write(table_str)
                f.write(footer)
        except OSError as e:
            print(f"Error writing formatted output: {e}")
    else:
        # Assume it is a file object
        file.write(header)
        file.write(table_str)
        file.write(footer)