###
## cluster_maker: demo for analysis from CSV
## James Foadi - University of Bath
## November 2025
###

"""
Demo script for cluster_maker package: CSV data analysis and export.

This script demonstrates the data analysis functionality including:
- CSV file loading
- Numeric summary computation (mean, std, min, max, missing values)
- Export to CSV and formatted text files

Usage:
    python demo/analyse_from_csv.py <input_csv_file>

Example:
    python demo/analyse_from_csv.py demo/sample_data.csv

Output:
    - demo_output/analysis_summary.csv (machine-readable)
    - demo_output/analysis_summary.txt (human-readable)
"""

from __future__ import annotations

import sys
import os
import pandas as pd
from cluster_maker.data_analyser import calculate_numeric_summary
from cluster_maker.data_exporter import export_numeric_summary

OUTPUT_DIR = "demo_output"


def main(args: list[str]) -> None:
    # Check command line arguments
    if len(args) != 2:
        print("ERROR: Incorrect number of arguments provided.")
        print("Usage: python demo/analyse_from_csv.py [input_csv_file]")
        sys.exit(0)  # Exit without traceback

    input_path = args[1]
    print(f"Input CSV file: {input_path}")

    # Check if file exists
    if not os.path.exists(input_path):
        print(f"ERROR: The file '{input_path}' does not exist.")
        sys.exit(0)

    try:
        # Read CSV
        print("Reading CSV file...")
        df = pd.read_csv(input_path)
        print("CSV loaded successfully.")

        # Calculate summary
        print("Calculating numeric summary...")
        summary = calculate_numeric_summary(df)
        
        if summary.empty:
            print("WARNING: No numeric columns found in the input file.")
        else:
            print("Summary calculated.")

        # Prepare output paths
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        csv_output_path = os.path.join(OUTPUT_DIR, "analysis_summary.csv")
        text_output_path = os.path.join(OUTPUT_DIR, "analysis_summary.txt")

        # Export results
        print(f"Exporting results to {OUTPUT_DIR}...")
        export_numeric_summary(summary, csv_output_path, text_output_path)
        
        print("Done.")
        print(f"CSV summary saved to: {csv_output_path}")
        print(f"Text summary saved to: {text_output_path}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv)
