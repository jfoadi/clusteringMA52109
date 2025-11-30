###
## cluster_maker - test_interface_and_export.py
## Test robustness and controlled error handling for high-level interface and exporters.
## November 2025
###

import unittest
import pandas as pd
import os
import shutil
import numpy as np

# Import high-level interface and exporter functions
from cluster_maker import run_clustering # Assumes run_clustering is importable from package root
from cluster_maker.data_exporter import export_to_csv, export_summary 

class TestInterfaceAndExport(unittest.TestCase):
    """
    Tests the error handling and controlled exceptions of the high-level 
    interface (run_clustering) and the exporter functions.
    """
    
    def setUp(self):
        """Set up environment variables, temporary paths, and data."""
        self.temp_dir = "temp_test_dir"
        self.dummy_input_file = os.path.join(self.temp_dir, "dummy_input.csv")
        self.missing_file = os.path.join(self.temp_dir, "non_existent_file.csv")
        self.valid_output_dir = os.path.join(self.temp_dir, "output")
        self.invalid_output_dir_parent = "/non/existent/path/for/test" # Used to test failure on non-existent parent

        # Create temporary directory for all test files
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Create a valid input file for testing (only two columns, x and y, as required by the demo)
        self.valid_df = pd.DataFrame({
            'x': np.random.rand(10),
            'y': np.random.rand(10),
            'irrelevant': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'] # Non-numeric
        })
        self.valid_df.to_csv(self.dummy_input_file, index=False)
        
        # Create a summary DataFrame for exporter testing (Task 3a structure)
        self.summary_df = pd.DataFrame({
            'mean': [10.5, 5.2],
            'std': [2.1, 0.5],
            'min': [8.0, 4.0],
            'max': [12.0, 6.0],
            'missing': [0.0, 1.0]
        }, index=['Feature_A', 'Feature_B'])


    def tearDown(self):
        """Clean up the temporary directory and its contents."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


    # =====================================================================
    # 5a) High-Level Interface Tests (run_clustering)
    # =====================================================================

    def test_run_clustering_file_not_found_raises_controlled_error(self):
        """Checks if run_clustering raises a FileNotFoundError for a missing input file."""
        with self.assertRaisesRegex(FileNotFoundError, "Input file not found"):
            run_clustering(
                input_path=self.missing_file, 
                feature_cols=['x', 'y'], 
                algorithm="kmeans", 
                k=3,
                output_path=os.path.join(self.valid_output_dir, "out.csv")
            )

    def test_run_clustering_missing_feature_column_raises_controlled_error(self):
        """Checks if run_clustering raises a ValueError for missing feature columns."""
        # The valid_df has columns 'x', 'y', 'irrelevant'. Requesting 'z' should fail.
        with self.assertRaisesRegex(ValueError, "Required feature column\(s\) not found"):
            run_clustering(
                input_path=self.dummy_input_file, 
                feature_cols=['x', 'z'], # 'z' is missing
                algorithm="kmeans", 
                k=3,
                output_path=os.path.join(self.valid_output_dir, "out.csv")
            )

    # =====================================================================
    # 5b) Exporter Function Tests (data_exporter.py)
    # =====================================================================
    
    def test_export_summary_creates_files(self):
        """Checks that export_summary creates both CSV and TXT files and the output directory."""
        output_name = "test_summary_report"
        
        # This function should create the directory if it doesn't exist
        export_summary(self.summary_df, output_name, output_dir=self.valid_output_dir)

        csv_path = os.path.join(self.valid_output_dir, f"{output_name}.csv")
        txt_path = os.path.join(self.valid_output_dir, f"{output_name}.txt")

        self.assertTrue(os.path.exists(csv_path), "CSV file was not created by export_summary.")
        self.assertTrue(os.path.exists(txt_path), "Text file was not created by export_summary.")
        self.assertTrue(os.path.isdir(self.valid_output_dir), "Output directory was not created.")

    def test_export_to_csv_creates_file(self):
        """Checks that export_to_csv creates the file with a valid path."""
        output_path = os.path.join(self.valid_output_dir, "test_simple_export.csv")
        export_to_csv(self.valid_df, output_path)
        self.assertTrue(os.path.exists(output_path), "export_to_csv failed to create file.")

    def test_export_to_csv_invalid_path_raises_controlled_error(self):
        """
        Checks that export_to_csv raises a controlled error (e.g., FileNotFoundError) 
        if the output directory is invalid, as this function does not use os.makedirs.
        """
        invalid_file_path = os.path.join(self.invalid_output_dir_parent, "bad_path.csv")
        
        # When writing to a non-existent directory, Python raises FileNotFoundError
        with self.assertRaisesRegex(FileNotFoundError, "No such file or directory"):
            export_to_csv(self.valid_df, invalid_file_path)