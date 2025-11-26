###
### cluster_maker - test file
### Tests for interface error handling and data export functionality.
###

import unittest
import os
import tempfile
import pandas as pd

from cluster_maker.data_exporter import export_summary_files
from cluster_maker.interface import run_clustering

class TestInterfaceErrorHandling(unittest.TestCase):
    
    def test_missing_input_file(self):
        """ Calling run_clustering with a missing file must raise a clean error. """
        missing_path = "non_existent_file.csv"
        with self.assertRaises(FileNotFoundError):
            run_clustering(missing_path)
    
    def test_missing_required_column(self):
        """ run_clustering should raise a clean ValueError when the CSV does not contain required numeric columns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_csv = os.path.join(tmpdir, "bad.csv")

            # Create a CSV without numeric columns
            df = pd.DataFrame({
                'wrong1': [1, 2, 3],
                'wrong2': ['x', 'y', 'z']
            })
            df.to_csv(bad_csv, index=False)

            with self.assertRaises(ValueError):
                run_clustering(bad_csv)


class TestExportSummaryFiles(unittest.TestCase):

    def test_export_creates_files(self):
        """Valid input should create both CSV and text output files."""
        with tempfile.TemporaryDirectory() as tmpdir:

            csv_out = os.path.join(tmpdir, "summary.csv")
            txt_out = os.path.join(tmpdir, "summary.txt")

            # Create a sample summary DataFrame
            summary = pd.DataFrame({
                "mean": [1.0, 2.0],
                "std": [0.1, 0.2],
                "max": [1.5, 2.5],
                "min": [0.5, 1.5],
                "n_missing": [0, 0]
            }, index=["A", "B"])

            export_summary_files(summary, csv_out, txt_out)

            # Check that files were created
            self.assertTrue(os.path.isfile(csv_out))
            self.assertTrue(os.path.isfile(txt_out))

    def test_export_invalid_path(self):
        """Exporter must raise an error when given an invalid output path."""

        # Create a sample summary DataFrame
        summary = pd.DataFrame({
            "mean": [1.0],
            "std": [0.1],
            "max": [1.5],
            "min": [0.5],
            "n_missing": [0]
        }, index=["A"])

        # Path to a directory that should NOT exist
        invalid_dir = "/this/path/is/not/real"
        csv_out = os.path.join(invalid_dir, "summary.csv")
        txt_out = os.path.join(invalid_dir, "summary.txt")

        # Expect a controlled error (usually FileNotFoundError or OSError)
        with self.assertRaises(Exception):
            export_summary_files(summary, csv_out, txt_out)

if __name__ == "__main__":
    unittest.main()