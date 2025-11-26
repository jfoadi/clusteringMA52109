from __future__ import annotations

import os
import sys
import pandas as pd

# Import the core package functions
from cluster_maker.data_analyser import summarize_numeric_data
from cluster_maker.data_exporter import export_summary_data # Assuming this is your function

OUTPUT_DIR = "demo_output_analysis"
SUMMARY_CSV = "numeric_summary.csv"
SUMMARY_TXT = "numeric_summary.txt"


def main(args: list[str]) -> None:
    """
    Analyzes an input CSV file and exports the numeric summary statistics.
    """
    
    # --- a) Command Line Argument Handling ---
    print("=== Data Analysis Demo: Summarize Numeric Data ===\n")
    
    # The length of sys.argv must be 2: [script_name, input_path]
    if len(args) != 2:
        print("ERROR: Incorrect number of arguments provided.")
        print("Usage: python demo/analyse_from_csv.py path/to/input.csv")
        sys.exit(1)
        
    # Get the input path (at index 1, as index 0 is the script name)
    input_path = args[1]
    
    # --- b) Read Input CSV ---
    print(f"Reading input file: {input_path}")
    
    if not os.path.exists(input_path):
        print(f"\nERROR: The file '{input_path}' does not exist.")
        sys.exit(1)
        
    try:
        df = pd.read_csv(input_path)
        print(f"Input data loaded successfully. Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}\n")
    except Exception as exc:
        print(f"\nERROR reading CSV file: {exc}")
        sys.exit(1)

    # --- c) Compute Numeric Summary ---
    print("Running numeric data analysis...")
    
    # Use the function from data_analyser.py
    summary_df = summarize_numeric_data(df)
    
    if summary_df.empty:
        print("\nWARNING: No numeric columns found in the input data. Aborting export.")
        sys.exit(0)

    print(f"Analysis complete. Summarized {summary_df.shape[0]} numeric columns.")
    
    # --- c) & d) Export and Progress Messages ---
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    csv_path = os.path.join(OUTPUT_DIR, SUMMARY_CSV)
    text_path = os.path.join(OUTPUT_DIR, SUMMARY_TXT)
    
    print("\nExporting results...")
    
    # Use the function from data_exporter.py
    try:
        export_summary_data(
            summary_df=summary_df,
            csv_path=csv_path,
            text_path=text_path
        )
        print("Export successful.")
        print(f"  - CSV file saved to: {csv_path}")
        print(f"  - Text file saved to: {text_path}")
        
    except Exception as exc:
        print(f"\nERROR during data export:\n{exc}")
        sys.exit(1)

    print("\n=== Analysis Demo Complete ===")


if __name__ == "__main__":
    # Note: Using sys.argv is correct here for command line arguments
    main(sys.argv)