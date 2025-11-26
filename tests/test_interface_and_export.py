import unittest
import os
import pandas as pd
import numpy as np

# Import the functions we need to test
from cluster_maker.interface import run_clustering
from cluster_maker.data_exporter import export_to_csv, export_formatted

class TestInterfaceAndExport(unittest.TestCase):
    
    def setUp(self):
        """
        Runs before each test. Sets up temporary filenames and data.
        """
        self.temp_input_csv = "test_temp_input.csv"
        self.temp_output_csv = "test_temp_output.csv"
        self.temp_output_txt = "test_temp_output.txt"
        
        # Create a valid dummy DataFrame and save it as CSV for input tests
        self.df = pd.DataFrame({
            'col1': [1.0, 2.0, 3.0],
            'col2': [4.0, 5.0, 6.0],
            'col3': [7.0, 8.0, 9.0]
        })
        self.df.to_csv(self.temp_input_csv, index=False)

    def tearDown(self):
        """
        Runs after each test. Cleans up any files created.
        """
        for f in [self.temp_input_csv, self.temp_output_csv, self.temp_output_txt]:
            if os.path.exists(f):
                os.remove(f)

    # --- Part (a): Testing run_clustering error handling ---

    def test_run_clustering_missing_file(self):
        """
        Test that run_clustering raises FileNotFoundError if input path doesn't exist.
        """
        missing_path = "non_existent_ghost_file.csv"
        
        # We expect a FileNotFoundError when the file is missing
        with self.assertRaises(FileNotFoundError):
            run_clustering(
                input_path=missing_path,
                feature_cols=['col1', 'col2']
            )

    def test_run_clustering_missing_columns(self):
        """
        Test that run_clustering raises KeyError if the CSV is missing required columns.
        """
        # The temp CSV has col1, col2, col3. We request 'col_impossible'.
        # select_features (called inside run_clustering) should raise KeyError.
        with self.assertRaises(KeyError):
            run_clustering(
                input_path=self.temp_input_csv,
                feature_cols=['col1', 'col_impossible']
            )

    # --- Part (b): Testing data_exporter functions ---

    def test_export_to_csv_success(self):
        """
        Test that export_to_csv successfully creates a file.
        """
        export_to_csv(self.df, self.temp_output_csv)
        self.assertTrue(os.path.exists(self.temp_output_csv), "CSV output file was not created.")

    def test_export_to_csv_invalid_path(self):
        """
        Test that export_to_csv raises an error if the directory does not exist.
        """
        # Define a path in a non-existent directory
        bad_path = os.path.join("non_existent_folder", "file.csv")
        
        # Pandas to_csv raises OSError (or FileNotFoundError in newer versions) 
        # when the parent directory doesn't exist.
        with self.assertRaises((OSError, FileNotFoundError)):
            export_to_csv(self.df, bad_path)

    def test_export_formatted_success(self):
        """
        Test that export_formatted successfully creates a text file.
        """
        export_formatted(self.df, self.temp_output_txt)
        self.assertTrue(os.path.exists(self.temp_output_txt), "Text output file was not created.")

    def test_export_formatted_invalid_path(self):
        """
        Test that export_formatted raises an error if the directory does not exist.
        """
        bad_path = os.path.join("non_existent_folder", "file.txt")
        
        # Python's built-in open() raises FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            export_formatted(self.df, bad_path)

if __name__ == '__main__':
    unittest.main()

### 🧪 How to Run the New Tests

# Run this in your terminal from the project root:

# ```bash
# PYTHONPATH=. python3 -m unittest tests/test_interface_and_export.py