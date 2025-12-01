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


# Task 3b) New function added
def export_statistics_summary(
    stats_df: pd.DataFrame,
    csv_filename: str,
    text_filename: str
) -> None:
    """
    Export statistics summary to both CSV and human-readable text formats.

    Parameters
    ----------
    stats_df : pandas.DataFrame
        Statistics DataFrame from calculate_column_statistics function.
    csv_filename : str
        Path for the CSV output file.
    text_filename : str
        Path for the human-readable text summary file.
    """
    print("Starting statistics export process...")
    
    # Input validation
    if not isinstance(stats_df, pd.DataFrame):
        print("ERROR: Input must be a pandas DataFrame.")
        raise TypeError("stats_df must be a pandas DataFrame.")
    
    if stats_df.empty:
        print("ERROR: Input DataFrame is empty.")
        raise ValueError("stats_df must not be empty.")
    
    required_columns = ['mean', 'std', 'min', 'max', 'missing']
    missing_columns = [col for col in required_columns if col not in stats_df.columns]
    
    if missing_columns:
        print(f"ERROR: Missing required columns: {missing_columns}")
        raise ValueError(f"stats_df must contain columns: {required_columns}")
    
    print(f"Input statistics DataFrame shape: {stats_df.shape}")
    print(f"Columns in stats DataFrame: {list(stats_df.columns)}")
    print(f"Rows (features) to export: {len(stats_df)}")
    
    # Export to CSV
    print(f"Exporting to CSV file: {csv_filename}")
    try:
        export_to_csv(stats_df, csv_filename, include_index=True)
        print("CSV export completed successfully!")
    except Exception as e:
        print(f"CSV export failed: {e}")
        raise
    
    # Export to human-readable text file
    print(f"Creating human-readable summary: {text_filename}")
    try:
        with open(text_filename, 'w', encoding='utf-8') as f:
            # Write header
            f.write("=" * 60 + "\n")
            f.write("COLUMN STATISTICS SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            # Write summary for each column
            for column_name in stats_df.index:
                stats = stats_df.loc[column_name]
                
                f.write(f"COLUMN: {column_name}\n")
                f.write("-" * 40 + "\n")
                f.write(f"  Mean:          {stats['mean']:>12.6f}\n")
                f.write(f"  Std Dev:       {stats['std']:>12.6f}\n")
                f.write(f"  Minimum:       {stats['min']:>12.6f}\n")
                f.write(f"  Maximum:       {stats['max']:>12.6f}\n")
                f.write(f"  Missing Values: {int(stats['missing']):>12d}\n") # Fixed line so now demo file in task 4) should run correctly
                
                # Add some derived statistics for better insights
                f.write(f"  Range:         {(stats['max'] - stats['min']):>12.6f}\n")
                f.write(f"  Coef of Variation: {(stats['std'] / stats['mean'] if stats['mean'] != 0 else float('inf')):>12.6f}\n")
                
                f.write("\n")
            
            # Write overall summary
            f.write("=" * 60 + "\n")
            f.write("OVERALL SUMMARY\n")
            f.write("=" * 60 + "\n")
            f.write(f"Total columns analyzed: {len(stats_df)}\n")
            f.write(f"Total missing values: {stats_df['missing'].sum()}\n")
            f.write(f"Data range across all columns:\n")
            f.write(f"  Overall min: {stats_df['min'].min():.6f}\n")
            f.write(f"  Overall max: {stats_df['max'].max():.6f}\n")
            f.write("=" * 60 + "\n")
        
        print("Human-readable summary created successfully!")
        
    except Exception as e:
        print(f"Text export failed: {e}")
        raise
    
    # Final success message
    print("Statistics export completed successfully!")
    print("Output files created:")
    print(f"CSV file: {csv_filename}")
    print(f"Text summary: {text_filename}")
    print(f"Summary details:")
    print(f"   - Exported statistics for {len(stats_df)} columns")
    print(f"   - CSV format: Machine-readable data")
    print(f"   - Text format: Human-readable analysis")
    print("Task 3b: Statistics export function completed successfully!")
    print("-" * 60)        