from __future__ import annotations

import sys
import pandas as pd
import os # To check if the input file exists and for printing absolute paths

# Import required functions from the cluster_maker package
from cluster_maker.data_analyser import analyse_numeric_features
from cluster_maker.data_exporter import export_summary 

OUTPUT_DIR = "demo_output"
BASE_FILENAME = "numeric_analysis_report"

def main(args: list[str]) -> None:
    """
    Main function to orchestrate reading a CSV, analyzing numeric features,
    and exporting the summary.
    """
    
    # 4a) Handle command line arguments and print usage if incorrect
    if len(args) != 2:
        print("ERROR: Incorrect number of arguments provided.")
        print("Usage: python demo/analyse_from_csv.py [input_csv_file]")
        sys.exit(1)

    # The input path is the second argument (args[1])
    input_path = args[1]
    
    print("=== cluster_maker demo: Numeric Feature Analysis ===\n")
    print(f"Input file specified: {input_path}")
    
    # Check if file exists before proceeding
    if not os.path.exists(input_path):
        print(f"ERROR: The input file '{input_path}' does not exist.")
        sys.exit(1)

    # 4b) Read the input CSV into a pandas DataFrame
    print("\n Reading CSV file into DataFrame...")
    try:
        # Assuming the standard delimiter, but adding an informative error for parsing issues
        df = pd.read_csv(input_path) 
        print(f"Data loaded successfully. Rows: {len(df)}, Columns: {len(df.columns)}.")
    except pd.errors.ParserError:
        print("ERROR: Failed to parse CSV file. Check file integrity or delimiter.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during file reading: {e}")
        sys.exit(1)


    # 4c) Use analysis and export functions
    print("\nComputing numeric feature summary...")
    
    # Compute the numeric summary
    summary_df = analyse_numeric_features(df)
    
    if summary_df.empty:
        print("Analysis completed, but no numeric columns were found to report.")
        sys.exit(0) # Exit gracefully
        
    print(f"Analysis complete. Summary generated for {len(summary_df)} numeric features.")
    
    
    print("\n Exporting results...")
    
    # Export the summary to CSV and TXT in the "demo_output" directory
    try:
        export_summary(
            summary_df=summary_df, 
            base_filename=BASE_FILENAME, 
            output_dir=OUTPUT_DIR
        )
        print("All outputs saved successfully.")
    except Exception as e:
        print(f"ERROR during file export: {e}")
        sys.exit(1)
    
    print("\n=== Demo finished successfully ===")


if __name__ == "__main__":
    # The first item in sys.argv is the script name itself, hence we pass sys.argv
    main(sys.argv)