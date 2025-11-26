###
## cluster_maker
## James Foadi - University of Bath
## November 2025
###

from __future__ import annotations

from typing import Union, TextIO, List 

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
        
        
def export_summary_data(
    summary_df: pd.DataFrame, 
    csv_path: str, 
    text_path: str
) -> None:
    """
    Exports the numeric summary DataFrame to both a CSV file and a neatly
    formatted human-readable text file.

    Parameters
    ----------
    summary_df : pandas.DataFrame
        The DataFrame of summary statistics (features as index, stats as columns).
    csv_path : str
        The file path for the output CSV file.
    text_path : str
        The file path for the output human-readable text file.
    """
    
    # 1. Write the summary DataFrame to a CSV file
    # index=True ensures the 'feature' names (from the index) are included as the first column.
    summary_df.to_csv(csv_path, index=True)

    # 2. Prepare the content for the human-readable text file
    summary_lines: List[str] = ["--- Numeric Feature Summary ---"]
    
    # Iterate over the DataFrame rows (one row per feature)
    # Using itertuples for efficient row-wise iteration
    for row in summary_df.itertuples(index=True, name='FeatureSummary'):
        feature_name = row.Index
        
        # Create a neatly formatted line with aligned statistics
        line = (
            f"{feature_name:<15}: "  # Left-align feature name (15 spaces)
            f"Mean={row.mean:>8.3f}, "  # Right-align float (8 spaces, 3 decimal places)
            f"Std={row.std:>8.3f}, "
            f"Min={row.min:>8.3f}, "
            f"Max={row.max:>8.3f}, "
            f"Missing={int(row.missing_count):>4}" # Right-align integer
        )
        summary_lines.append(line)
        
    summary_text = "\n".join(summary_lines)

    # 3. Write the formatted content to a separate text file
    with open(text_path, 'w') as f:
        f.write(summary_text)