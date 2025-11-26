###
## cluster_maker: demo for analysis and export from CSV
## MA52109 Mock Exam - Task 4
###

from __future__ import annotations

import os
import sys
import pandas as pd

# Import the new analysis and export functions from Task 3
from cluster_maker.data_analyser import summarise_numeric_data
from cluster_maker.data_exporter import export_summary_report

# Define output directory based on Task 4c
OUTPUT_DIR = "demo_output" 

def main(args: list[str]) -> None:
    # ----------------------------------------------------
    # a) Command-line argument handling
    # ----------------------------------------------------
    # Check if the number of command line arguments is exactly 2 (script path + 1 input file) [cite: 21]
    if len(args) != 2:
        # Print a clear error message [cite: 21]
        print("ERROR: This script requires exactly one argument (the input CSV file path).")
        # Print a short "Usage:" line [cite: 22]
        print("Usage: python demo/analyse_from_csv.py [path/to/input.csv]")
        # Exit without raising a traceback [cite: 23]
        sys.exit(1)

    # The input path is the first argument after the script name (args[1])
    input_path = args[1] 

    # d) Print informative progress messages [cite: 28]
    print(f"--- Starting Numeric Data Analysis Workflow ---")
    print(f"Input file specified: {input_path}")

    # Check file existence
    if not os.path.exists(input_path):
        print(f"\nERROR: The input file '{input_path}' was not found.")
        print("Exiting script.")
        sys.exit(1)
    
    # Ensure output directory exists for saving files (Task 4c)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory confirmed: {OUTPUT_DIR}")
    print("-" * 40)

    # ----------------------------------------------------
    # b) Read the input CSV into a pandas DataFrame [cite: 24]
    # ----------------------------------------------------
    try:
        print("1. Reading input CSV file into a pandas DataFrame...") # [cite: 28]
        df = pd.read_csv(input_path)
        print(f"   Success: Data loaded with {len(df)} rows and {len(df.columns)} columns.")
    except Exception as e:
        print(f"\nFATAL ERROR: Could not read CSV file at {input_path}. Reason: {e}")
        sys.exit(1)
        
    print("-" * 40)

    # ----------------------------------------------------
    # c) Use new analysis and export functions [cite: 25, 26]
    # ----------------------------------------------------
    
    # Compute the numeric summary [cite: 25]
    print("2. Computing numeric summary...") # [cite: 28]
    summary_df = summarise_numeric_data(df)
    
    if summary_df.empty:
        print("   Warning: No numeric columns were found to summarise. Exiting.")
        sys.exit(0)
        
    print(f"   Success: Summary created for {len(summary_df.columns)} numeric columns.")

    # Define export paths (CSV and text file in demo_output directory) [cite: 26, 27]
    csv_output = os.path.join(OUTPUT_DIR, "numeric_summary.csv")
    text_output = os.path.join(OUTPUT_DIR, "numeric_summary_report.txt")

    # Export results [cite: 26]
    print("3. Exporting summary results...") # [cite: 28]
    export_summary_report(
        summary_df=summary_df,
        csv_path=csv_output, # To a CSV file in the "demo_output" directory [cite: 26]
        text_path=text_output, # To a human-readable text file in the same directory [cite: 27]
    )

    print(f"   * Summary CSV saved to: {csv_output}")
    print(f"   * Report TXT saved to: {text_output}")
    
    print("-" * 40)
    print("--- Workflow Complete ---")


if __name__ == "__main__":
    # Pass all command-line arguments (including script path at args[0])
    main(sys.argv)