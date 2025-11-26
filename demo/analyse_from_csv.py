###
## analyse_from_csv.py
## Demo script for numeric summary analysis
## MA52109 - Programming for Data Science
## Task 4
###

from __future__ import annotations

import os
import sys
import pandas as pd

from cluster_maker.data_analyser import numeric_summary
from cluster_maker.data_exporter import export_numeric_summary


OUTPUT_DIR = "demo_output"


def main(args: list[str]) -> None:
    print("=== cluster_maker: Numeric Summary Analysis ===\n")

    # --- Check command-line arguments ---
    if len(args) != 2:
        print("ERROR: Incorrect number of arguments.")
        print("Usage: python demo/analyse_from_csv.py path/to/input.csv")
        sys.exit(1)

    input_path = args[1]

    print(f"Input CSV: {input_path}")

    if not os.path.exists(input_path):
        print(f"\nERROR: File '{input_path}' does not exist.")
        sys.exit(1)

    # --- Load the CSV file ---
    print("\nReading CSV file...")
    try:
        df = pd.read_csv(input_path)
    except Exception as exc:
        print(f"ERROR: Could not read CSV file:\n{exc}")
        sys.exit(1)

    print("CSV loaded successfully.")
    print(f"Columns found: {list(df.columns)}")

    # --- Run numeric summary analysis ---
    print("\nComputing numeric summary...")
    try:
        summary_df = numeric_summary(df)
    except Exception as exc:
        print(f"ERROR while computing numeric summary:\n{exc}")
        sys.exit(1)

    print("Numeric summary computed:")
    print(summary_df)

    # --- Ensure output directory exists ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    csv_out = os.path.join(OUTPUT_DIR, "numeric_summary.csv")
    txt_out = os.path.join(OUTPUT_DIR, "numeric_summary.txt")

    # --- Export results ---
    print("\nSaving summary output files...")
    try:
        export_numeric_summary(summary_df, csv_out, txt_out)
    except Exception as exc:
        print(f"ERROR while exporting summary:\n{exc}")
        sys.exit(1)

    print(f"Summary CSV saved to: {csv_out}")
    print(f"Summary text file saved to: {txt_out}")

    print("\n=== Analysis complete ===")


if __name__ == "__main__":
    main(sys.argv)
