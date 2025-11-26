from __future__ import annotations

import os
import sys
import pandas as pd

# Import the required functions from the package
from cluster_maker.data_analyser import calculate_summary_stats
from cluster_maker.data_exporter import export_summary_report

# Define the output directory and filenames
OUTPUT_DIR = "demo_output"
CSV_REPORT_FILENAME = os.path.join(OUTPUT_DIR, "summary_report.csv")
TEXT_REPORT_FILENAME = os.path.join(OUTPUT_DIR, "summary_report.txt")


def main(args: list[str]) -> None:
    """
    Runs the summary statistics analysis on an input CSV file specified
    via the command line.
    """
    print("=== cluster_maker demo: analysis from CSV ===")

    # a) Argument handling and error checking
    if len(args) != 2:
        print("\nERROR: Incorrect number of arguments provided.")
        print("Usage: python demo/analyse_from_csv.py path/to/input.csv")
        sys.exit(1)

    input_path = args[1] # The CSV file is the second argument (index 1)

    print(f"Input CSV file: {input_path}")

    if not os.path.exists(input_path):
        print(f"\nERROR: The file '{input_path}' does not exist.")
        sys.exit(1)

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory confirmed: {OUTPUT_DIR}")
    print("-" * 50)

    # b) Read the input CSV
    print(f"Reading data from {input_path}...")
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"\nERROR: Could not read CSV file. Details:\n{e}")
        sys.exit(1)
    
    print(f"Data loaded successfully. {len(df)} rows and {len(df.columns)} columns found.")
    
    # c) Compute the numeric summary
    print("\nRunning summary statistics calculation...")
    try:
        summary_df = calculate_summary_stats(df)
    except Exception as e:
        print(f"\nERROR: Analysis failed. Details:\n{e}")
        sys.exit(1)
        
    if summary_df.empty:
        print("WARNING: Analysis returned an empty report (no numeric features found).")
    else:
        print(f"Analysis completed successfully for {len(summary_df)} numeric feature(s).")
    
    # c) Export the summary
    print("\nExporting analysis report...")
    try:
        export_summary_report(
            summary_df=summary_df,
            csv_filename=CSV_REPORT_FILENAME,
            text_filename=TEXT_REPORT_FILENAME,
        )
        print(f"Report saved to CSV:\n  - {CSV_REPORT_FILENAME}")
        print(f"Report saved to formatted text:\n  - {TEXT_REPORT_FILENAME}")
    except Exception as e:
        print(f"\nERROR: Export failed. Details:\n{e}")
        sys.exit(1)

    # d) Print final messages
    print("\n=== Demo finished successfully ===")


if __name__ == "__main__":
    # sys.argv includes the script name as the first element
    main(sys.argv)