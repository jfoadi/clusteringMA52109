###
## cluster_maker: analyse_from_csv demo
## James Foadi - University of Bath
## November 2025
###
from __future__ import annotations

import os
import sys

import pandas as pd

from cluster_maker.data_analyser import summarise_numeric
from cluster_maker.data_exporter import export_summary

OUTPUT_DIR = "demo_output"


def main(args: list[str]) -> None:
    """
    Command-line demo script to:
      - read a CSV file,
      - compute numeric summary statistics,
      - export the summary to CSV and text files in demo_output/.
    """
    print("=== cluster_maker demo: analyse numeric columns from CSV ===\n")
    print("This demo will:")
    print("  - read a CSV file you provide,")
    print("  - compute basic descriptive statistics for all numeric columns")
    print("    (mean, std, min, max, number of missing values),")
    print("  - save the summary to:")
    print("      * a CSV file, and")
    print("      * a human-readable text file in the 'demo_output' directory.\n")
    print("Note: this script does not fit a clustering model;")
    print("      it focuses on a simple descriptive statistics 'model' for your data.\n")

    # We expect exactly 2 command-line arguments:
    #   args[0] = script name
    #   args[1] = path/to/input.csv
    if len(args) != 2:
        print("ERROR: Incorrect number of arguments.")
        print("Usage: python demo/analyse_from_csv.py path/to/input.csv")
        # Exit cleanly without a traceback
        sys.exit(1)

    input_path = args[1]
    print(f"Input CSV file: {input_path}")

    # Check that the file exists
    if not os.path.exists(input_path):
        print(f"\nERROR: The file '{input_path}' does not exist.")
        sys.exit(1)

    # Load the CSV into a DataFrame
    print("\nReading CSV file into a pandas DataFrame...")
    try:
        df = pd.read_csv(input_path)
    except Exception as exc:
        print("\nERROR: Failed to read the CSV file.")
        print(f"Details: {exc}")
        sys.exit(1)

    print("Data loaded successfully.")
    print(f"Number of rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    # Show a sample of the dataset
    print("\nHere is a preview of the dataset (first 5 rows):")
    print(df.head())

    # Compute numeric summary
    print("\nComputing numeric summary (mean, std, min, max, n_missing) "
          "for all numeric columns...")
    summary = summarise_numeric(df)

    if summary.empty:
        print("\nWARNING: No numeric columns were found in this dataset.")
        print("No summary files will be created.")
        sys.exit(0)

    print("Summary computed successfully.")
    print("\nSummary preview:")
    print(summary)

    # Ensure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Define output paths
    csv_output_path = os.path.join(OUTPUT_DIR, "numeric_summary.csv")
    txt_output_path = os.path.join(OUTPUT_DIR, "numeric_summary.txt")

    print(f"\nSaving summary to:")
    print(f"  - CSV: {csv_output_path}")
    print(f"  - Text: {txt_output_path}")

    try:
        export_summary(summary, csv_output_path, txt_output_path)
    except Exception as exc:
        print("\nERROR: Failed to export summary.")
        print(f"Details: {exc}")
        sys.exit(1)

    print("\nSummary export completed successfully.")
    print("You can now inspect the files in the 'demo_output' directory.")
    print("\n=== End of analyse_from_csv demo ===")


if __name__ == "__main__":
    main(sys.argv)
