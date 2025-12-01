###
## cluster_maker – tests for interface and exporting
## Yas Akilakulasingam - University of Bath
## November 2025
###


## These tests verify robust error handling, clear messages, and
## correct behaviour of high-level interface and exporting functions.
## No tracebacks should occur during these failures.

import unittest
import os
import pandas as pd
from tempfile import TemporaryDirectory

from cluster_maker import run_clustering
from cluster_maker.data_exporter import export_summary


class TestInterfaceAndExport(unittest.TestCase):
    """
    Tests for:
      - the high-level interface function run_clustering();
      - the exporting functions in data_exporter.py.

    These tests check for:
      • controlled, user-friendly errors,
      • no raw Python tracebacks,
      • correct exception types,
      • clear error messages,
      • correct file creation behaviour.
    """

    # ------------------------------------------------------------------
    # PART (a): Tests for the high-level run_clustering interface
    # ------------------------------------------------------------------

    def test_run_clustering_missing_input_file(self):
        """Test that a missing CSV file triggers a clean FileNotFoundError
        with a clear error message and NO raw traceback."""
        
        fake_path = "this_file_does_not_exist.csv"

        with self.assertRaises(FileNotFoundError) as context:
            run_clustering(
                input_path=fake_path,
                feature_cols=["x", "y"],
                algorithm="kmeans",
                k=2,
                standardise=True,
                output_path="dummy.csv",
            )

        # Check the error message is informative
        self.assertIn("no such file", str(context.exception).lower())

    def test_run_clustering_missing_feature_columns(self):
        """If the CSV exists but does not contain the required feature
        columns, run_clustering() must raise a controlled ValueError
        with a clear message."""

        # Create a small valid CSV that is missing the required columns
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [4, 5, 6]
        })

        with TemporaryDirectory() as tmpdir:
            temp_csv = os.path.join(tmpdir, "temp.csv")
            df.to_csv(temp_csv, index=False)

            with self.assertRaises(KeyError) as context:
                run_clustering(
                    input_path=temp_csv,
                    feature_cols=["x", "y"],   # these don’t exist
                    algorithm="kmeans",
                    k=2,
                    standardise=True,
                    output_path=os.path.join(tmpdir, "output.csv"),
                )

            # Check message clarity
            self.assertIn("feature", str(context.exception).lower())
            self.assertIn("missing", str(context.exception).lower())

    # ------------------------------------------------------------------
    # PART (b): Tests for exporting functions
    # ------------------------------------------------------------------

    def test_export_summary_creates_files(self):
        """Test that export_summary() successfully writes the CSV and
        the human-readable text file when given valid inputs."""

        summary_df = pd.DataFrame({
            "mean": [1.0, 2.0],
            "std": [0.1, 0.2],
        })

        with TemporaryDirectory() as tmpdir:
            csv_out = os.path.join(tmpdir, "summary.csv")
            txt_out = os.path.join(tmpdir, "summary.txt")

            export_summary(summary_df, csv_out, txt_out)

            # Both files should now exist
            self.assertTrue(os.path.exists(csv_out))
            self.assertTrue(os.path.exists(txt_out))

    def test_export_summary_invalid_path(self):
        """Test that export_summary() raises a clear error when given an
        invalid or non-existent directory path."""

        summary_df = pd.DataFrame({
            "mean": [1.0],
        })

        # Invalid paths that cannot be written
        bad_csv = "/no_such_directory/summary.csv"
        bad_txt = "/no_such_directory/summary.txt"

        with self.assertRaises(Exception) as context:
            export_summary(summary_df, bad_csv, bad_txt)

        # Error message should be clear
        self.assertIn("directory", str(context.exception).lower())


if __name__ == "__main__":
    unittest.main()