###
## cluster_maker: analyse_from_csv demo script
## Georgie Paterson - University of Bath
## November 2025
###

from __future__ import annotations

import os
import sys
import pandas as pd

# Add parent directory (clusteringMA52109) to module search path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PARENT_DIR)

from cluster_maker.data_analyser import column_summary
from cluster_maker.data_exporter import export_summary


OUTPUT_DIR = "demo_output"


def main(args: list[str]) -> None:
    print("\n=== cluster_maker: CSV Analysis Tool ===\n")

    # ----------------------------------------------------------------------
    # (a) Check command-line arguments
    # ----------------------------------------------------------------------
    if len(args) != 2:
        print("ERROR: Incorrect number of arguments.")
        print("Usage: python demo/analyse_from_csv.py path/to/input.csv")
        return  # exit cleanly without crashing

    input_path = args[1]
    print(f"> Input CSV file provided: {input_path}\n")

    # ----------------------------------------------------------------------
    # (b) Read CSV file
    # ----------------------------------------------------------------------
    print("Step 1: Loading the CSV file...")
    if not os.path.exists(input_path):
        print(f"ERROR: The file '{input_path}' does not exist.")
        return

    try:
        df = pd.read_csv(input_path)
    except Exception as exc:
        print("ERROR: Could not read the CSV file.")
        print(f"Details: {exc}")
        return

    print("✓ CSV file loaded successfully.")
    print(f"  - Number of rows: {len(df)}")
    print(f"  - Number of columns: {len(df.columns)}")
    print("  - Column names:", list(df.columns))

    print("\nPreview of the first 5 rows:")
    print(df.head())
    print("\n")

    # ----------------------------------------------------------------------
    # (c) Explain what we are about to compute
    # ----------------------------------------------------------------------
    print("Step 2: Preparing to compute column summaries...")
    print("We will analyse *all* columns in the file.")
    print("Numeric columns will include:")
    print("  - mean")
    print("  - standard deviation")
    print("  - minimum value")
    print("  - maximum value")
    print("  - number of missing (NaN) values")
    print("Non-numeric columns will still be listed and clearly labelled.")
    print("\nRunning analysis...\n")

    # ----------------------------------------------------------------------
    # (c) Compute summary statistics
    # ----------------------------------------------------------------------
    summary_df = column_summary(df)

    print("✓ Summary computed successfully.\n")

    print("Here is the start of the summary table:")
    print(summary_df.head())
    print("\n")

    # ----------------------------------------------------------------------
    # (c) Export results
    # ----------------------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_output_path = os.path.join(OUTPUT_DIR, "column_summary.csv")
    txt_output_path = os.path.join(OUTPUT_DIR, "column_summary.txt")

    print("Step 3: Saving output files...")
    export_summary(summary_df, csv_output_path, txt_output_path)

    print("✓ Summary files saved.")
    print(f"  - CSV summary: {csv_output_path}")
    print(f"  - Text summary: {txt_output_path}\n")

    # ----------------------------------------------------------------------
    # (d) Final message
    # ----------------------------------------------------------------------
    print("=== Analysis complete ===")
    print("You can now open the summary files to review the results.")
    print("Thank you for using cluster_maker.\n")


if __name__ == "__main__":
    main(sys.argv)