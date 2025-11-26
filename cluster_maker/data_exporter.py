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


def export_summary_statistics(
    summary_df: pd.DataFrame,
    csv_filename: str,
    text_filename: str,
) -> None:
    """
    Export summary statistics to both CSV and human-readable text files.
    
    Parameters
    ----------
    summary_df : pandas.DataFrame
        Summary statistics DataFrame from calculate_comprehensive_statistics
    csv_filename : str
        Output CSV filename
    text_filename : str
        Output human-readable text filename
    """
    if not isinstance(summary_df, pd.DataFrame):
        raise TypeError("summary_df must be a pandas DataFrame.")
    
    # Export to CSV
    export_to_csv(summary_df, csv_filename, include_index=True)
    
    # Export to human-readable text file
    with open(text_filename, "w", encoding="utf-8") as f:
        f.write("COMPREHENSIVE STATISTICS SUMMARY\n")
        f.write("=" * 40 + "\n\n")
        
        if summary_df.empty:
            f.write("No numeric columns found in the data.\n")
            return
        
        for col_name, row in summary_df.iterrows():
            f.write(f"Column: {col_name}\n")
            f.write(f"  Mean: {row['mean']:.4f}\n")
            f.write(f"  Standard Deviation: {row['std']:.4f}\n")
            f.write(f"  Minimum: {row['min']:.4f}\n")
            f.write(f"  Maximum: {row['max']:.4f}\n")
            f.write(f"  Missing Values: {int(row['missing_count'])}\n")
            f.write("-" * 30 + "\n")        



         