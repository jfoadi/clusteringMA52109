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

def export_summary(summary_df: pd.DataFrame, base_filename: str, output_dir: str = '.') -> None:
    """
    Exports a summary DataFrame (mean, std, min, max, missing) to a 
    CSV file and a neatly formatted human-readable text file.

    Parameters
    ----------
    summary_df : pandas.DataFrame
        The summary DataFrame with feature names as index and statistics 
        as columns ('mean', 'std', 'min', 'max', 'missing').
    base_filename : str
        The base name for the output files (e.g., 'analysis_report').
    output_dir : str, default='.'
        The directory where the files should be saved.
    """
    if not isinstance(summary_df, pd.DataFrame):
        raise TypeError("summary_df must be a pandas DataFrame.")
        
    # Ensure the output directory exists using os
    os.makedirs(output_dir, exist_ok=True)
    
    # Define file paths using os.path.join
    csv_path = os.path.join(output_dir, f"{base_filename}.csv")
    txt_path = os.path.join(output_dir, f"{base_filename}.txt")

    # 1. Write to CSV file 
    summary_df.to_csv(csv_path, index=True, float_format='%.4f')
    print(f"Summary DataFrame successfully exported to CSV: {os.path.abspath(csv_path)}")

    # 2. Write to human-readable text file (formatted summary)
    with open(txt_path, 'w') as f:
        f.write("--- Feature Analysis Summary Report ---\n\n")
        f.write(f"Timestamp: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("-" * 50 + "\n")
        
        # Format the summary, one line per column/feature
        for col_name, row in summary_df.iterrows():
            f.write(f"Feature: {col_name}\n")
            f.write(f"  > Descriptive Statistics\n")
            f.write(f"    Mean: {row['mean']:.4f}  |  Std Dev: {row['std']:.4f}\n")
            f.write(f"    Min:  {row['min']:.4f}  |  Max: {row['max']:.4f}\n")
            f.write(f"  > Data Quality\n")
            f.write(f"    Missing Values: {int(row['missing'])}\n") 
            f.write("-" * 50 + "\n")
            
    print(f"Summary report successfully exported to text file: {os.path.abspath(txt_path)}")