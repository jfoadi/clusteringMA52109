###
## cluster_maker - test_interface_and_export.py
## MA52109 Mock Exam - Task 5
###

import unittest
import os
import tempfile
import pandas as pd
import shutil # Used for cleaning up temporary directories

# Import the functions to be tested
from cluster_maker.interface import run_clustering
from cluster_maker.data_exporter import export_to_csv, export_formatted


class TestInterfaceAndExport(unittest.TestCase):
    """Tests high-level interface and export functions for controlled error handling."""

    # ----------------------------------------------------
    # Setup for creating temporary files and data
    # ----------------------------------------------------
    def setUp(self):
        # Create a temporary directory for test inputs/outputs
        self.temp_dir = tempfile.mkdtemp()
        self.valid_output_path = os.path.join(self.temp_dir, "valid_output.csv")
        self.valid_text_path = os.path.join(self.temp_dir, "valid_output.txt")
        
        # Define a path that points to a non-existent directory (for error testing) [cite: 35]
        self.non_existent_dir_path = os.path.join(self.temp_dir, "non_existent_dir", "fail.csv")

        # Create a valid DataFrame and save it to a temporary CSV
        self.valid_df = pd.DataFrame({
            'feature_1': [1.0, 2.0, 3.0, 4.0],
            'feature_2': [10.0, 20.0, 30.0, 40.0],
            'non_feature': ['a', 'b', 'c', 'd'],
        })
        self.valid_input_path = os.path.join(self.temp_dir, "valid_input.csv")
        self.valid_df.to_csv(self.valid_input_path, index=False)
        
        # Create a file with the wrong columns (for testing Task 5a)
        self.bad_feature_input_path = os.path.join(self.temp_dir, "bad_input_cols.csv")
        pd.DataFrame({
            'wrong_col_A': [1, 2, 3],
            'wrong_col_B': [4, 5, 6],
        }).to_csv(self.bad_feature_input_path, index=False)

    def tearDown(self):
        # Clean up the temporary directory and all its contents
        shutil.rmtree(self.temp_dir)
        
    # ----------------------------------------------------
    # a) Testing run_clustering error handling (Interface)
    # ----------------------------------------------------
    
    def test_interface_missing_input_file(self):
        """
        Checks that calling run_clustering with a missing input file 
        raises a controlled FileNotFoundError (no raw Python traceback)[cite: 33].
        """
        missing_path = os.path.join(self.temp_dir, "file_that_does_not_exist.csv")
        # We check for FileNotFoundError [cite: 33]
        with self.assertRaises(FileNotFoundError):
            run_clustering(
                input_path=missing_path, # Missing input file [cite: 30]
                feature_cols=['feature_1', 'feature_2'],
                k=2
            )

    def test_interface_missing_feature_cols(self):
        """
        Checks that calling run_clustering with a file lacking required feature 
        columns results in a controlled KeyError (no raw Python traceback)[cite: 33].
        """
        # The interface calls select_features, which raises KeyError if columns are missing.
        # Check for KeyError or a subclass of ValueError[cite: 33].
        with self.assertRaisesRegex(KeyError, "missing"): 
            run_clustering(
                input_path=self.bad_feature_input_path, # CSV without required feature columns [cite: 31]
                feature_cols=['feature_1', 'feature_2'], # These are not in the CSV
                k=2
            )
            
    # ----------------------------------------------------
    # b) Testing export functions error handling
    # ----------------------------------------------------
    
    def test_export_to_csv_creates_file(self):
        """Check that export_to_csv successfully creates an output file when given valid inputs[cite: 34]."""
        export_to_csv(self.valid_df, self.valid_output_path)
        self.assertTrue(os.path.exists(self.valid_output_path), "CSV file was not created.")
        
    def test_export_formatted_creates_file(self):
        """Check that export_formatted successfully creates an output text file when given valid inputs[cite: 34]."""
        export_formatted(self.valid_df, self.valid_text_path)
        self.assertTrue(os.path.exists(self.valid_text_path), "Formatted text file was not created.")

    def test_export_to_csv_invalid_path_error(self):
        """
        Checks that export_to_csv raises a clean, controlled error (FileNotFoundError) 
        if the output directory does not exist[cite: 35].
        """
        with self.assertRaises(OSError):
            export_to_csv(self.valid_df, self.non_existent_dir_path)

    def test_export_formatted_invalid_path_error(self):
        """
        Checks that export_formatted raises a clean, controlled error (FileNotFoundError) 
        if the output directory does not exist[cite: 35].
        """
        with self.assertRaises(OSError):
            export_formatted(self.valid_df, self.non_existent_dir_path)


if __name__ == "__main__":
    unittest.main()