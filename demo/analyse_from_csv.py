###
## cluster_maker: demo for numeric analysis from CSV
## James Foadi - University of Bath
## November 2025
###

from __future__ import annotations

import os
import sys
import pandas as pd

from cluster_maker import numeric_summary, export_summary_csv, export_summary_text

OUTPUT_DIR = "demo_output"


def main(args: list[str]) -> None:
    print("=== cluster_maker demo: numeric analysis from CSV ===\n")

    # 4a) Check for exactly 2 arguments (script name + CSV file)
    if len(args) != 2:
        print("ERROR: Incorrect number of arguments provided.")
        print("Usage: python demo/analyse_from_csv.py [input_csv_file]")
        sys.exit(0)

    # Input CSV file
    input_path = args[1]
    print(f"Input CSV file: {input_path}")

    # Check file exists
    if not os.path.exists(input_path):
        print(f"ERROR: The file '{input_path}' does not exist.")
        sys.exit(0)

    # 4b) Read the input CSV into a pandas DataFrame
    print("\nReading CSV file...")
    try:
        df = pd.read_csv(input_path)
        print(f"✓ CSV file loaded successfully.")
        print(f"  - Rows: {len(df)}")
        print(f"  - Columns: {list(df.columns)}")
    except Exception as exc:
        print(f"ERROR reading CSV file:\n{exc}")
        sys.exit(0)

    # 4c) Compute numeric summary
    print("\nComputing numeric summary...")
    try:
        summary = numeric_summary(df)
        if summary.empty:
            print("WARNING: No numeric columns found in the DataFrame.")
            sys.exit(0)
        print(f"✓ Numeric summary computed for {len(summary)} numeric column(s).")
    except Exception as exc:
        print(f"ERROR computing summary:\n{exc}")
        sys.exit(0)

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nOutput directory '{OUTPUT_DIR}' ready.")

    # Export to CSV file
    csv_filename = os.path.join(OUTPUT_DIR, "numeric_summary.csv")
    print(f"\nExporting summary to CSV file...")
    try:
        export_summary_csv(summary, csv_filename)
        print(f"✓ CSV file saved to: {csv_filename}")
    except Exception as exc:
        print(f"ERROR exporting to CSV:\n{exc}")
        sys.exit(0)

    # Export to human-readable text file
    text_filename = os.path.join(OUTPUT_DIR, "numeric_summary.txt")
    print(f"Exporting summary to human-readable text file...")
    try:
        export_summary_text(summary, text_filename)
        print(f"✓ Text file saved to: {text_filename}")
    except Exception as exc:
        print(f"ERROR exporting to text:\n{exc}")
        sys.exit(0)

    # Summary
    print("\n" + "=" * 60)
    print("Analysis complete. Output files:")
    print(f"  - {csv_filename}")
    print(f"  - {text_filename}")
    print("=" * 60)
    print("\n=== End of demo ===")


if __name__ == "__main__":
    main(sys.argv)
