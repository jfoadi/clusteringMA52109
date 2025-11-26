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
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    
    # No try/except here -> Let the error happen so the test can see it!
    data.to_csv(filename, sep=delimiter, index=include_index)


def export_formatted(
    data: pd.DataFrame,
    file: Union[str, TextIO],
    include_index: bool = True,
) -> None:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    table_str = data.to_string(index=include_index)
    header = "=== Data Analysis Summary ===\n\n"
    footer = "\n\n============================="

    if isinstance(file, str):
        # No try/except here
        with open(file, 'w') as f:
            f.write(header + table_str + footer)
    else:
        file.write(header + table_str + footer)