## 4) Create a NEW demo script in the "demo" directory, called 
## "analyse_from_csv.py". This script must:
##      a) Be executable from the command line as:
##             python demo/analyse_from_csv.py path/to/input.csv
##         If the number of command line arguments is not exactly 2, the
##         script must:
##          - print a clear error message;
##          - print a short "Usage:" line showing the correct syntax;
##          - exit without raising a traceback.
##      b) Read the input CSV into a pandas DataFrame.
##      c) Use the functions you wrote or fixed in "data_analyser.py" and
##         "data_exporter.py" to:
##          - compute the numeric summary;
##          - export it to:
##           * a CSV file in the "demo_output" directory; and
##           * a human-readable text file in the same directory.
##      d) Print informative progress messages to the screen (for example,
##         when reading the CSV, running the analysis, and saving files),
##         so that a non-expert user can follow what is happening.

from __future__ import annotations
import os
import sys
import pandas as pd
from cluster_maker.data_analyser import column_statistics
from cluster_maker.data_exporter import export_to_csv, export_formatted

OUTPUT_DIR = "demo_output"

def main(args: list[str]) -> None:
    print("=== cluster_maker demo: data analysis from CSV ===\n")

    # Require exactly one argument: the CSV file path
    if len(args) != 2:
        print("ERROR: Incorrect number of arguments provided.")
        print("Usage: python demo/analyse_from_csv.py [input_csv_file]")
        sys.exit(1)

    # Input CSV file
    input_path = args[1]
    print(f"Input CSV file: {input_path}")

    # Check file exists
    if not os.path.exists(input_path):
        print(f"\nERROR: The file '{input_path}' does not exist.")
        sys.exit(1)

    # Load data
    print("\nLoading data from CSV...")
    df = pd.read_csv(input_path)
    print("Data loaded successfully.")
    print(f"Number of rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    # Compute numeric summary
    print("\nComputing numeric summary statistics...")
    stats_df = column_statistics(df)
    print("Summary statistics computed.")

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Export to CSV
    csv_output_path = os.path.join(OUTPUT_DIR, "numeric_summary.csv")
    print(f"\nExporting summary statistics to CSV: {csv_output_path}")
    export_to_csv(stats_df, csv_output_path)
    print("CSV export completed.")

    # Export to formatted text file
    text_output_path = os.path.join(OUTPUT_DIR, "numeric_summary.txt")
    print(f"\nExporting summary statistics to text file: {text_output_path}")
    export_formatted(stats_df, text_output_path)
    print("Text file export completed.")

    print("\nData analysis complete.")

    print("\n=== End of demo ===")

if __name__ == "__main__":
    main(sys.argv)