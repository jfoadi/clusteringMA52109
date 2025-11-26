

import unittest
import pandas as pd
import os
from cluster_maker.interface import run_clustering
from cluster_maker.data_exporter import export_to_csv

class TestInterfaceAndExport(unittest.TestCase):
    def setUp(self):
        # Create a temporary dummy CSV for testing
        self.temp_csv = "test_temp_input.csv"
        pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        }).to_csv(self.temp_csv, index=False)

    def tearDown(self):
        # Clean up temp files after tests run
        if os.path.exists(self.temp_csv):
            os.remove(self.temp_csv)
        if os.path.exists("test_output.csv"):
            os.remove("test_output.csv")

    def test_run_clustering_missing_file(self):
        """
        Task 5a: Check that run_clustering raises FileNotFoundError 
        (or similar) when the input file is missing.
        """
        print("\n[Test] Checking missing file handling...")
        with self.assertRaises(FileNotFoundError):
            run_clustering(
                input_path="ghost_file_that_does_not_exist.csv",
                feature_cols=['col1']
            )

    def test_run_clustering_missing_columns(self):
        """
        Task 5a: Check that run_clustering raises KeyError 
        when requested columns are missing.
        """
        print("\n[Test] Checking missing column handling...")
        with self.assertRaises(KeyError):
            run_clustering(
                input_path=self.temp_csv,
                feature_cols=['non_existent_col']
            )

    def test_export_invalid_path(self):
        """
        Task 5b: Check that exporting raises an error if the path is invalid 
        (e.g., a directory that doesn't exist).
        """
        print("\n[Test] Checking export error handling...")
        df = pd.DataFrame({'a': [1, 2]})
        # Try saving to a folder that definitely doesn't exist
        invalid_path = "non_existent_folder_123/output.csv"
        
        with self.assertRaises(OSError):
            export_to_csv(df, invalid_path)

if __name__ == '__main__':
    unittest.main()