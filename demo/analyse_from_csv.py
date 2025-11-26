###
## cluster_maker: CSV Analysis Demo
## James Foadi - University of Bath
## November 2025
###

from __future__ import annotations

import os
import sys
import pandas as pd

# Import the functions from your package
from cluster_maker.data_analyser import calculate_comprehensive_statistics
from cluster_maker.data_exporter import export_summary_statistics


def main(args: list[str]) -> None:
    """
    Main function to analyse a CSV file and generate comprehensive statistics.
    
    Parameters
    ----------
    args : list of str
        Command line arguments
    """
    print("=== cluster_maker: CSV Analysis Demo ===\n")
    
    # Part 4a: Check command line arguments
    if len(args) != 2:
        print("ERROR: Incorrect number of arguments provided.")
        print("Usage: python demo/analyse_from_csv.py path/to/input.csv")
        print(f"Received {len(args)} argument(s): {args}")
        sys.exit(1)
    
    input_path = args[1]
    print(f"Input CSV file: {input_path}")
    
    # Check if file exists
    if not os.path.exists(input_path):
        print(f"\nERROR: The file '{input_path}' does not exist.")
        print("Please check the file path and try again.")
        sys.exit(1)
    
    # Part 4b: Read the input CSV
    print("\nStep 1: Reading CSV file...")
    try:
        df = pd.read_csv(input_path)
        print(f"✓ Successfully loaded data with {len(df)} rows and {len(df.columns)} columns")
        print(f"  Columns: {list(df.columns)}")
    except Exception as e:
        print(f"✗ ERROR: Failed to read CSV file: {e}")
        sys.exit(1)
    
    # Check if DataFrame has data
    if len(df) == 0:
        print("✗ ERROR: The CSV file is empty.")
        sys.exit(1)
    
    # Part 4c: Compute numeric summary
    print("\nStep 2: Computing comprehensive statistics...")
    try:
        summary_df = calculate_comprehensive_statistics(df)
        numeric_cols_count = len(summary_df)
        print(f"✓ Analyzed {numeric_cols_count} numeric column(s)")
        
        if numeric_cols_count == 0:
            print("  Warning: No numeric columns found in the data")
        else:
            print(f"  Numeric columns: {list(summary_df.index)}")
    except Exception as e:
        print(f"✗ ERROR: Failed to compute statistics: {e}")
        sys.exit(1)
    
    # Ensure output directory exists
    output_dir = "demo_output"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nStep 3: Saving results to '{output_dir}' directory...")
    
    # Generate output filenames based on input filename
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    csv_output = os.path.join(output_dir, f"{base_name}_summary.csv")
    text_output = os.path.join(output_dir, f"{base_name}_summary.txt")
    
    # Part 4c: Export to both CSV and text files
    try:
        export_summary_statistics(summary_df, csv_output, text_output)
        print(f"✓ CSV summary saved to: {csv_output}")
        print(f"✓ Text summary saved to: {text_output}")
    except Exception as e:
        print(f"✗ ERROR: Failed to export results: {e}")
        sys.exit(1)
    
    # Display summary of findings
    print("\nStep 4: Analysis Summary")
    print("-" * 40)
    
    if len(summary_df) == 0:
        print("No numeric columns were found in the input data.")
        print("Please ensure your CSV contains numeric columns for analysis.")
    else:
        print(f"Total numeric columns analyzed: {len(summary_df)}")
        total_missing = summary_df['missing_count'].sum()
        print(f"Total missing values across all numeric columns: {total_missing}")
        
        # Show a preview of key statistics
        print("\nPreview of statistics (first 3 columns):")
        preview_cols = list(summary_df.index)[:3]
        for col in preview_cols:
            stats = summary_df.loc[col]
            print(f"  {col}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, "
                  f"range=[{stats['min']:.2f}, {stats['max']:.2f}], "
                  f"missing={stats['missing_count']}")
    
    print("\n=== Analysis completed successfully! ===")
    print(f"Check the '{output_dir}' directory for detailed results.")


if __name__ == "__main__":
    main(sys.argv)