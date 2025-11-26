#!/usr/bin/env python3

import sys
import os
import pandas as pd

from cluster_maker.data_analyser import calculate_descriptive_statistics
from cluster_maker.data_exporter import export_to_csv, export_formatted

def main():
    # Check command-line arguments
    if len(sys.argv) != 2:
        print("ERROR: Incorrect number of arguments provided.")
        print("Usage: python demo/analyse_from_csv.py path/to/input.csv")
        sys.exit(1)

    input_csv = sys.argv[1]

    if not os.path.isfile(input_csv):
        print(f"ERROR: File '{input_csv}' does not exist.")
        sys.exit(1)

    print(f"Reading input CSV file: {input_csv}")
    df = pd.read_csv(input_csv)

    print("Calculating descriptive statistics...")
    summary_df = calculate_descriptive_statistics(df)

    # Ensure demo_output directory exists
    output_dir = "demo_output"
    os.makedirs(output_dir, exist_ok=True)

    csv_file = os.path.join(output_dir, "summary.csv")
    txt_file = os.path.join(output_dir, "summary.txt")

    print(f"Exporting summary to CSV: {csv_file}")
    export_to_csv(summary_df, csv_file)

    print(f"Exporting summary to human-readable text file: {txt_file}")
    export_formatted(summary_df, txt_file)

    print("Analysis complete. Files saved in 'demo_output' directory.")

if __name__ == "__main__":
    main()
