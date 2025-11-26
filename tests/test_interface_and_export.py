###
## cluster_maker - test file for interface and export
## James Foadi - University of Bath
## November 2025
###

import unittest
import os
import tempfile
import shutil

import pandas as pd
import numpy as np

from cluster_maker import run_clustering, export_summary_csv, export_summary_text


class TestRunClusteringErrors(unittest.TestCase):
    """Test error handling for run_clustering interface function."""

    def test_missing_input_file(self):
        """Test that run_clustering raises FileNotFoundError for missing input file."""
        missing_file = "this_file_does_not_exist_12345.csv"
        
        with self.assertRaises(FileNotFoundError):
            run_clustering(
                input_path=missing_file,
                feature_cols=["x", "y"],
                k=3,
            )

    def test_missing_feature_columns(self):
        """Test that run_clustering raises appropriate error for missing feature columns."""
        # Create a temporary CSV file without the required columns
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_file = f.name
            f.write("a,b,c\n1,2,3\n4,5,6\n")
        
        try:
            # Try to use non-existent feature columns
            with self.assertRaises((KeyError, ValueError)):
                run_clustering(
                    input_path=temp_file,
                    feature_cols=["x", "y"],  # These columns don't exist
                    k=3,
                )
        finally:
            os.unlink(temp_file)


class TestExportFunctions(unittest.TestCase):
    """Test export functions for valid and invalid inputs."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_summary = pd.DataFrame({
            'mean': [1.0, 2.0, 3.0],
            'std': [0.5, 0.6, 0.7],
            'min': [0.0, 1.0, 2.0],
            'max': [2.0, 3.0, 4.0],
            'missing_count': [0, 1, 0],
        }, index=['col1', 'col2', 'col3'])

    def test_export_summary_csv_valid(self):
        """Test that export_summary_csv creates a CSV file with valid inputs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_file = os.path.join(temp_dir, "summary.csv")
            export_summary_csv(self.test_summary, csv_file)
            
            # Check that file exists
            self.assertTrue(os.path.exists(csv_file))
            
            # Check that file is not empty
            self.assertGreater(os.path.getsize(csv_file), 0)
            
            # Check that file can be read back
            df = pd.read_csv(csv_file, index_col=0)
            self.assertEqual(df.shape[0], 3)

    def test_export_summary_text_valid(self):
        """Test that export_summary_text creates a text file with valid inputs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            text_file = os.path.join(temp_dir, "summary.txt")
            export_summary_text(self.test_summary, text_file)
            
            # Check that file exists
            self.assertTrue(os.path.exists(text_file))
            
            # Check that file is not empty
            self.assertGreater(os.path.getsize(text_file), 0)
            
            # Check file content
            with open(text_file, 'r') as f:
                content = f.read()
                self.assertIn("col1", content)
                self.assertIn("mean", content)

    def test_export_summary_csv_invalid_path(self):
        """Test that export_summary_csv raises error for invalid path."""
        invalid_dir = "/nonexistent_directory_xyz/summary.csv"
        
        with self.assertRaises((OSError, FileNotFoundError)):
            export_summary_csv(self.test_summary, invalid_dir)

    def test_export_summary_text_invalid_path(self):
        """Test that export_summary_text raises error for invalid path."""
        invalid_dir = "/nonexistent_directory_xyz/summary.txt"
        
        with self.assertRaises((OSError, FileNotFoundError)):
            export_summary_text(self.test_summary, invalid_dir)


if __name__ == "__main__":
    unittest.main()
