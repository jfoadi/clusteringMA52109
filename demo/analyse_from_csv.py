###
## cluster_maker: CSV Analysis Demo Script
## Mock Practical MA52109
## November 2025
###

from __future__ import annotations

import os
import sys

import pandas as pd

# Import functions from our package
from cluster_maker import calculate_column_statistics, export_statistics_summary


def main(args: list[str]) -> None:
    """
    Main function to analyze CSV data and generate statistical summaries.
    
    Parameters
    ----------
    args : list of str
        Command line arguments
    """
    print("=" * 60)
    print("CLUSTER_MAKER: CSV DATA ANALYSIS TOOL")
    print("=" * 60)
    
    # Task 4a): Command line argument validation
    print("\nChecking command line arguments...")
    if len(args) != 2:
        print("ERROR: Incorrect number of arguments provided.")
        print("\nUsage:")
        print("  python demo/analyse_from_csv.py path/to/input.csv")
        print("\nExample:")
        print("  python demo/analyse_from_csv.py data/demo_data.csv")
        sys.exit(1)
    
    input_path = args[1]
    print(f"Command line arguments validated successfully")
    print(f"Input file specified: {input_path}")
    
    # Task 4a): Check if file exists
    print(f"\nVerifying input file existence...")
    if not os.path.exists(input_path):
        print(f"ERROR: The file '{input_path}' does not exist.")
        print("Please check the file path and try again.")
        sys.exit(1)
    print("Input file exists and is accessible")
    
    # Task 4b): Read the input CSV into a pandas DataFrame
    print(f"\nReading data from CSV file...")
    try:
        df = pd.read_csv(input_path)
        print(f"CSV file read successfully!")
        print(f"Data shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"Available columns: {list(df.columns)}")
    except Exception as e:
        print(f"ERROR: Failed to read CSV file: {e}")
        print("Please ensure the file is a valid CSV format.")
        sys.exit(1)
    
    # Check if we have any data to analyze
    if df.empty:
        print("ERROR: The CSV file is empty.")
        sys.exit(1)
    
    # Task 4c): Compute numeric summary using our function from Task 3a)
    print(f"\n" + "=" * 50)
    print("TASK 4c: COMPUTING NUMERIC SUMMARY STATISTICS")
    print("=" * 50)
    print("Using calculate_column_statistics function from Task 3a...")
    
    try:
        stats_df = calculate_column_statistics(df)
        print(f"Numeric summary computation completed!")
        print(f"Summary covers {len(stats_df)} numeric column(s)")
    except Exception as e:
        print(f"ERROR: Failed to compute statistics: {e}")
        sys.exit(1)
    
    # Task 4c): Export results using our function from Task 3b)
    print(f"\n" + "=" * 50)
    print("TASK 4c: EXPORTING RESULTS TO FILES")
    print("=" * 50)
    print("Using export_statistics_summary function from Task 3b...")
    
    # Create output directory if it doesn't exist
    output_dir = "demo_output"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}/")
    
    # Generate output filenames based on input filename
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    csv_output = os.path.join(output_dir, f"{base_name}_statistics.csv")
    text_output = os.path.join(output_dir, f"{base_name}_summary.txt")
    
    print(f"CSV output file: {csv_output}")
    print(f"Text summary file: {text_output}")
    
    try:
        export_statistics_summary(stats_df, csv_output, text_output)
        print("File export completed successfully!")
    except Exception as e:
        print(f"ERROR: Failed to export results: {e}")
        sys.exit(1)
    
    # Final summary
    print(f"\n" + "=" * 60)
    print("TASK 4: ANALYSIS COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("ANALYSIS SUMMARY:")
    print(f"   Input file: {input_path}")
    print(f"   Data analyzed: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"   Numeric columns processed: {len(stats_df)}")
    print(f"   Output files created:")
    print(f"        {csv_output} (Machine-readable CSV)")
    print(f"        {text_output} (Human-readable summary)")
    print(f"\nFUNCTIONS USED FROM CLUSTER_MAKER:")
    print(f"   calculate_column_statistics() - From Task 3a")
    print(f"   export_statistics_summary() - From Task 3b")
    print(f"\nAll Task 4 requirements completed!")
    print("=" * 60)


if __name__ == "__main__":
    main(sys.argv)