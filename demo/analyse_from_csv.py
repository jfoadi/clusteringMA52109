###
## cluster_maker: demo for numeric summary and export
## Yas Akilakulasingam - University of Bath
## November 2025
###



from __future__ import annotations

import os
import sys
import pandas as pd

# Path of this script: demo/analyse_from_csv.py
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Path of parent directory: clusteringMA52109/
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Add PROJECT_ROOT to Python path
sys.path.insert(0, PROJECT_ROOT)

from cluster_maker import summarise_numeric_columns, export_summary

OUTPUT_DIR = "demo_output"



def main(args: list[str]) -> None:
    print("=== cluster_maker demo: analyse_from_csv ===\n")

    # ------------------------------------------------------------
    # Validate command-line arguments
    # ------------------------------------------------------------
    if len(args) != 2:
        print("ERROR: Incorrect number of arguments.")
        print("Usage: python demo/analyse_from_csv.py <input_csv_file>")
        sys.exit(1)

    # Use args[-1] so it works whether run directly or via python -m
    input_path = os.path.abspath(args[-1])
    print(f"Input file detected: {input_path}")

    # Check file exists before trying to read it
    if not os.path.exists(input_path):
        print(f"ERROR: The file '{input_path}' does not exist.")
        sys.exit(1)

    # ------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------
    print("\nLoading CSV file...")
    try:
        df = pd.read_csv(input_path)
    except Exception as exc:
        print(f"ERROR: Unable to read the file:\n{exc}")
        sys.exit(1)

    print("File loaded successfully.")
    print(f"Rows: {len(df)}   Columns: {list(df.columns)}")

    # ------------------------------------------------------------
    # Compute numeric summary using the function from part (a)
    # ------------------------------------------------------------
    print("\nComputing summary statistics for numeric columns...")
    try:
        summary_df = summarise_numeric_columns(df)
    except Exception as exc:
        print(f"ERROR while computing summary:\n{exc}")
        sys.exit(1)

    print("Summary created successfully.")
    print(summary_df)
    print("-" * 60)

    # ------------------------------------------------------------
    # Ensure output directory exists
    # ------------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    csv_out   = os.path.join(OUTPUT_DIR, "summary.csv")
    text_out  = os.path.join(OUTPUT_DIR, "summary.txt")

    print(f"Saving results to:\n  {csv_out}\n  {text_out}")

    # ------------------------------------------------------------
    # Export results using the new function from part (b)
    # ------------------------------------------------------------
    try:
        export_summary(summary_df, csv_out, text_out)
    except Exception as exc:
        print(f"ERROR while exporting summary:\n{exc}")
        sys.exit(1)

    print("\nSummary successfully exported.")
    print("=== End of analyse_from_csv demo ===")


if __name__ == "__main__":
    main(sys.argv)