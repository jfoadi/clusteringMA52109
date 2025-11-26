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

def export_summary_report(
    summary_df: pd.DataFrame,
    csv_path: str,
    txt_path: str
) -> None:
    """
    Export the summary DataFrame to CSV and a formatted text file.

    Parameters
    ----------
    summary_df : pandas.DataFrame
        The DataFrame produced by calculate_extended_statistics.
    csv_path : str
        Destination for the CSV file.
    txt_path : str
        Destination for the text file.
    """
    # 1. Export to CSV
    summary_df.to_csv(csv_path)

    # 2. Export to formatted text file
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=== Data Summary Report ===\n\n")
        f.write(f"{'Column':<20} | {'Mean':<10} | {'Std':<10} | {'Min':<10} | {'Max':<10} | {'Missing':<8}\n")
        f.write("-" * 85 + "\n")
        
        for col_name, row in summary_df.iterrows():
            # Handle potential NaNs in stats (e.g., if a column is all NaN)
            mean_val = f"{row['mean']:.2f}" if pd.notna(row['mean']) else "NaN"
            std_val = f"{row['std']:.2f}" if pd.notna(row['std']) else "NaN"
            min_val = f"{row['min']:.2f}" if pd.notna(row['min']) else "NaN"
            max_val = f"{row['max']:.2f}" if pd.notna(row['max']) else "NaN"
            missing_val = str(int(row['n_missing']))

            f.write(f"{col_name:<20} | {mean_val:<10} | {std_val:<10} | {min_val:<10} | {max_val:<10} | {missing_val:<8}\n")