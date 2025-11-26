import unittest
import pandas as pd
import numpy as np
import os
import shutil
import tempfile
from pathlib import Path

# Assuming these are available from your cluster_maker package
# NOTE: We need the actual functions for testing, but since we don't have run_clustering,
# we'll assume a dummy implementation for the sake of the test structure.
# You must replace DummyRunClustering with your actual run_clustering function.

# --- DUMMY INTERFACE FUNCTION (Needed only for this test file to run standalone) ---
# Replace this with: from cluster_maker.interface import run_clustering
class DummyRunClustering:
    def __init__(self, *args, **kwargs):
        pass # Placeholder class

def dummy_run_clustering(input_path, feature_cols, **kwargs):
    """Simulates the behavior of the high-level interface function."""
    
    # 1. Simulate FileNotFoundError for missing file
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input CSV file not found at: {input_path}")
        
    # Load data (to check for missing columns)
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        # Re-raise exceptions during read if any
        raise ValueError(f"Error reading CSV: {e}")

    # 2. Simulate ValueError for missing features
    missing_features = [col for col in feature_cols if col not in df.columns]
    if missing_features:
        raise ValueError(f"Required feature columns are missing: {missing_features}")
    
    # Simulate successful run
    return {"status": "success"}

# --- END DUMMY IMPLEMENTATION ---

# Import the actual exporter functions
from cluster_maker.data_exporter import export_summary_data
# Import data_analyser to create valid input data
from cluster_maker.data_analyser import summarize_numeric_data


class TestInterfaceAndExport(unittest.TestCase):
    # Setup and Teardown for temporary directories and files
    def setUp(self):
        """Create a temporary directory for test files."""
        self.test_dir = Path(tempfile.mkdtemp())
        
        # Create a small, valid DataFrame for testing (meets summary requirements)
        data = {
            'A': [1, 2, 3, 4, 5],
            'B': [10.0, 20.0, 30.0, np.nan, 50.0],
            'C': [101, 102, 103, 104, 105], 
            'D_Str': ['x', 'y', 'z', 'x', 'y']
        }
        self.input_df = pd.DataFrame(data)
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
        
        with self.assertRaisesRegex(FileNotFoundError, "not found"):
            # Using the dummy function here, replace with run_clustering
            dummy_run_clustering(
                input_path=missing_path, 
                feature_cols=['A', 'B']
            )

    def test_run_clustering_missing_feature_error(self):
        """Checks for ValueError when required feature columns are missing."""
        
        with self.assertRaisesRegex(ValueError, "Required feature columns are missing:"):
            # Using the dummy function here, replace with run_clustering
            dummy_run_clustering(
                input_path=self.valid_input_path, 
                feature_cols=['A', 'B', 'Missing_X'] # 'Missing_X' is the bad column
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
        
        # Check basic content of the CSV
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
        text_path = self.test_dir / "ok_path.txt" # Ensure one path is valid for clarity
        
        # Expect an OSError (or FileNotFoundError/IOError, depending on Python/OS) 
        # when trying to write to the non-existent directory.
        with self.assertRaises((OSError, FileNotFoundError, IOError)):
            export_summary_data(self.summary_df, str(csv_path), str(text_path))
            # The error should occur when to_csv tries to write to the path.

    
if __name__ == "__main__":
    unittest.main()