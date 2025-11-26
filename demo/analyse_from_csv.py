###
### cluster_maker : demo for analysing data from CSV
###

from __future__ import annotations

import sys
import os
import pandas as pd

from cluster_maker.data_analyser import get_numeric_summary
from cluster_maker.data_exporter import export_summary_files

def main():
    '''
    1) Check command line arguments
    '''
    if len(sys.argv) != 2:
        print("Error: Incorrect number of arguments.")
        print("Usage: python demo/analyse_from_csv.py path/to/input.csv")
        return
    
    input_path = sys.argv[1]

    if not os.path.isfile(input_path):
        print(f"Error: File '{input_path}' does not exist.")
        return
    
    '''
    2) Load data from CSV
    '''
    print(f"Loading data from CSV File '{input_path}'")
    try:
        data = pd.read_csv(input_path)
    except Exception as e:
        print(f"Error: Failed to read CSV file. {e}")
        return
    
    print("CSV loaded successfully.")

    '''
    3) Generate summary statistics
    '''
    print("Generating summary statistics...")
    summary_df = get_numeric_summary(data)
    print("Summary statistics generated.")

    '''
    4) Ensure output directory exists
    '''
    output_dir = os.path.join(os.path.dirname(__file__), "demo_output")
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        print(f"ERROR: Failed to create output directory '{output_dir}': {e}")
        sys.exit(1)

    csv_output_path = os.path.join(output_dir, "numeric_summary.csv")
    txt_output_path = os.path.join(output_dir, "numeric_summary.txt")

    '''
    5) Export results using data_exporter
    '''
    print(f"Saving results to {output_dir}")
    export_summary_files(summary_df, csv_output_path, txt_output_path)
    print("Results saved successfully.")

if __name__ == "__main__":
    main()