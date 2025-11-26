

import sys
import os
import pandas as pd

# Import the functions we created in Task 3
try:
    from cluster_maker.data_analyser import calculate_descriptive_statistics
    from cluster_maker.data_exporter import export_to_csv, export_formatted
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import cluster_maker modules.\n{e}")
    sys.exit(1)

OUTPUT_DIR = "demo_output"

def main():
    # --- Requirement 4a: Command Line Arguments ---
    # We require exactly 2 arguments: script name and csv file path
    if len(sys.argv) != 2:
        print("Error: Incorrect number of arguments.")
        print("Usage: python demo/analyse_from_csv.py <path/to/input.csv>")
        sys.exit(1)

    input_path = sys.argv[1]

    # Check if file exists
    if not os.path.exists(input_path):
        print(f"Error: The file '{input_path}' does not exist.")
        sys.exit(1)

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # --- Requirement 4b: Read CSV ---
    print(f"-> Reading data from: {input_path}")
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)

    # --- Requirement 4c: Compute Statistics ---
    print("-> Calculating descriptive statistics...")
    try:
        stats_df = calculate_descriptive_statistics(df)
        if stats_df.empty:
            print("Warning: No numeric data found to analyse.")
            sys.exit(0)
    except Exception as e:
        print(f"Error during analysis: {e}")
        sys.exit(1)

    # --- Requirement 4c: Export Results ---
    # Create filenames based on input name (e.g. data.csv -> data_summary.csv)
    base_name = os.path.basename(input_path).split('.')[0]
    csv_out_path = os.path.join(OUTPUT_DIR, f"{base_name}_summary.csv")
    txt_out_path = os.path.join(OUTPUT_DIR, f"{base_name}_summary.txt")

    print(f"-> Exporting results to {OUTPUT_DIR}...")
    try:
        # Export CSV
        export_to_csv(stats_df, csv_out_path, include_index=True)
        print(f"   [Saved] {csv_out_path}")

        # Export Text
        export_formatted(stats_df, txt_out_path, include_index=True)
        print(f"   [Saved] {txt_out_path}")

    except Exception as e:
        print(f"Error exporting files: {e}")
        sys.exit(1)

    print("Done. Analysis complete.")

if __name__ == "__main__":
    main()