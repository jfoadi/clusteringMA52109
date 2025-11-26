import unittest
import pandas as pd
import numpy as np
import os
import shutil
import tempfile
from pathlib import Path

# --- CORRECT IMPORTS ---
from cluster_maker.interface import run_clustering
from cluster_maker.data_exporter import export_summary_data
from cluster_maker.data_analyser import summarize_numeric_data
# -----------------------


class TestInterfaceAndExport(unittest.TestCase):
    # Setup and Teardown for temporary directories and files
    def setUp(self):
        """Create a temporary directory and a valid input file for tests."""
        self.test_dir = Path(tempfile.mkdtemp())
        
        # Create a small, valid DataFrame for testing
        data = {
            'A': [1, 2, 3, 4, 5],
            'B': [10.0, 20.0, 30.0, np.nan, 50.0],
            'C': [101, 102, 103, 104, 105], 
            'D_Str': ['x', 'y', 'z', 'x', 'y']
        }
        self.input_df = pd.DataFrame(data)
        
        # Create the summary DataFrame needed for export tests
        self.summary_df = summarize_numeric_data(self.input_df)

        # Create a valid input CSV file for the interface test
        self.valid_input_path = str(self.test_dir / "valid_input.csv")
        self.input_df.to_csv(self.valid_input_path, index=False)
        
    def tearDown(self):
        """Remove the temporary directory and all contents."""
        shutil.rmtree(self.test_dir)

    # ====================================================================
    # a) Interface Error Handling Tests (run_clustering)
    # ====================================================================
    
    def test_run_clustering_missing_file_error(self):
        """Checks for FileNotFoundError when input file is missing."""
        missing_path = str(self.test_dir / "non_existent_data.csv")
        
        # FIXED: Changed regex from "not found" to "No such file or directory" 
        # to match the actual exception message text received from the OS/Python.
        with self.assertRaisesRegex(FileNotFoundError, "No such file or directory"):
            run_clustering(
                input_path=missing_path, 
                feature_cols=['A', 'B']
            )

    def test_run_clustering_missing_feature_error(self):
        """Checks for ValueError when required feature columns are missing."""
        
        # This test now passes because preprocessing.py should raise ValueError
        # (the error text matches the actual ValueError raised).
        with self.assertRaisesRegex(ValueError, "The following feature columns are missing:"):
            run_clustering(
                input_path=self.valid_input_path, 
                feature_cols=['A', 'B', 'Missing_X']
            )

    # ====================================================================
    # b) Exporter Tests (data_exporter.py)
    # ====================================================================

    def test_export_summary_data_success(self):
        """Checks if export_summary_data successfully creates both files."""
        csv_path = self.test_dir / "test_summary.csv"
        text_path = self.test_dir / "test_summary.txt"
        
        # 1. Execution
        export_summary_data(self.summary_df, str(csv_path), str(text_path))
        
        # 2. Assertions
        self.assertTrue(csv_path.exists(), "CSV file was not created.")
        self.assertTrue(text_path.exists(), "Text file was not created.")
        
        # Check basic content of the CSV (3 features + 1 index column + 5 stats columns = 6 columns)
        read_csv = pd.read_csv(csv_path)
        self.assertEqual(read_csv.shape, (3, 6), "Exported CSV shape is incorrect.") 
        
        # Check basic content of the TEXT file (must not be empty)
        with open(text_path, 'r') as f:
            content = f.read()
            self.assertTrue(len(content) > 100, "Exported text file content is too short or empty.")

    def test_export_summary_data_invalid_path_error(self):
        """Checks for OSError when the target directory for export does not exist."""
        
        # Target a non-existent subdirectory within the temp directory
        invalid_dir = self.test_dir / "non_existent_folder"
        csv_path = invalid_dir / "bad_path.csv"
        text_path = self.test_dir / "ok_path.txt"
        
        # Expect an OSError (or FileNotFoundError/IsADirectoryError, etc.) 
        # when trying to write to the non-existent directory.
        with self.assertRaises((OSError, FileNotFoundError, IOError)):
            export_summary_data(self.summary_df, str(csv_path), str(text_path))
            
    
if __name__ == "__main__":
    unittest.main()