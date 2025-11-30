from __future__ import annotations

import os
import sys
import pandas as pd

from cluster_maker.data_analyser import get_numeric_column_summary
from cluster_maker.data_exporter import export_summary_report

# Configuration
OUTPUT_DIR = "demo_output"


def main(args: list[str]) -> None:
    """
    Main execution logic for running descriptive analysis on a CSV file.
    """
    print("=== cluster_maker demo: Data Analysis from CSV ===")

    # 4a) Command Line Argument Validation and Exit
    # We expect args[0] (script name) and args[1] (input file path)
    if len(args) != 2:
        print("\nERROR: Incorrect number of arguments provided.")
        print("Usage: python demo/analyse_from_csv.py [path/to/input.csv]")
        sys.exit(1)

    input_path = args[1] # The input file path is the second argument

    # Check file existence
    if not os.path.exists(input_path):
        print(f"\nERROR: The file '{input_path}' does not exist.")
        sys.exit(1)

    print(f"\nInput file specified: {input_path}")
    print(f"Output directory: {OUTPUT_DIR}")

    # Ensure output directory exists
    print("\nPreparing output directory...")
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"  -> Directory '{OUTPUT_DIR}' ensured.")
    except Exception as e:
        print(f"  -> ERROR creating output directory: {e}")
        sys.exit(1)
        
    # 4b) Read the input CSV into a pandas DataFrame.
    print("\nReading input CSV file into a Pandas DataFrame...")
    try:
        df = pd.read_csv(input_path)
        print(f"  -> Data loaded successfully. Shape: {df.shape}")
    except Exception as e:
        print(f"  -> ERROR reading CSV file: {e}")
        print("   (Check file path and delimiter setting.)")
        sys.exit(1)
        
    # 4c) Compute the numeric summary
    print("\nRunning descriptive analysis to compute numeric summary (mean, std, min, max, missing)...")
    try:
        # The function prints its own detailed progress
        summary_df = get_numeric_column_summary(df)
        print("  -> Numeric summary computation finished.")
    except Exception as e:
        print(f"  -> ERROR during summary calculation: {e}")
        sys.exit(1)
        
    # 4c) Export the results
    base_output_path = os.path.join(OUTPUT_DIR, "analysis_report")
    print("\nExporting results...")
    try:
        # This function handles both CSV and human-readable text file creation.
        export_summary_report(
            summary_df,
            base_filename=base_output_path,
        )
        print("  -> Export completed successfully.")
    except Exception as e:
        print(f"  -> ERROR during file export: {e}")
        sys.exit(1)
    
    # Final Output Summary
    print("\n=== Analysis Complete ===")
    print("Results saved to the 'demo_output' directory:")
    print(f"- CSV Summary: {base_output_path}_summary.csv")
    print(f"- Text Report: {base_output_path}_report.txt")


if __name__ == "__main__":
    # The main function is executed, passing all command line arguments
    main(sys.argv)