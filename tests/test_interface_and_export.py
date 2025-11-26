###
## cluster_maker - interface and exporter tests
## Georgie Paterson - University of Bath
## November 2025
###

import unittest
import pandas as pd
import numpy as np
import os
import tempfile
import shutil

from cluster_maker.interface import run_clustering
from cluster_maker.data_exporter import export_to_csv, export_summary
from cluster_maker.data_analyser import column_summary


class TestInterfaceAndExport(unittest.TestCase):

    # ---------------------------------------------------------------
    # Part (a): run_clustering error handling
    # ---------------------------------------------------------------
    def test_run_clustering_missing_file(self):
        """Calling run_clustering on a missing file should raise FileNotFoundError."""

        with self.assertRaises(FileNotFoundError):
            run_clustering(
                input_path="this_file_does_not_exist.csv",
                feature_cols=["a", "b"]
            )

    def test_run_clustering_missing_feature_columns(self):
        """Calling run_clustering with missing feature columns should raise KeyError."""

        # Create a temporary CSV file with only columns x, y
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "temp.csv")
            df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
            df.to_csv(csv_path, index=False)

            # run_clustering expects ["a", "b"], which do not exist
            with self.assertRaises(KeyError):
                run_clustering(
                    input_path=csv_path,
                    feature_cols=["a", "b"]
                )

    # ---------------------------------------------------------------
    # Part (b): testing exporting functions
    # ---------------------------------------------------------------
    def test_export_to_csv_creates_file(self):
        """export_to_csv should create a CSV when given a valid path."""

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "output.csv")

            df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
            export_to_csv(df, out_path, delimiter=",", include_index=False)

            self.assertTrue(os.path.exists(out_path))

    def test_export_to_csv_invalid_path(self):
        """export_to_csv should raise an error when path is invalid."""

        df = pd.DataFrame({"a": [1, 2]})

        # Invalid path: directory does not exist
        invalid_path = "/this/directory/does/not/exist/output.csv"

        with self.assertRaises(Exception):
            export_to_csv(df, invalid_path, delimiter=",", include_index=False)

    def test_export_summary_creates_files(self):
        """export_summary should create both CSV and TXT outputs."""

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_out = os.path.join(tmpdir, "summary.csv")
            txt_out = os.path.join(tmpdir, "summary.txt")

            df = pd.DataFrame({"a": [1, 2, 3], "b": [10, np.nan, 30]})
            summary = column_summary(df)

            export_summary(summary, csv_out, txt_out)

            self.assertTrue(os.path.exists(csv_out))
            self.assertTrue(os.path.exists(txt_out))

    def test_export_summary_invalid_path(self):
        """export_summary should raise an error for invalid output paths."""

        df = pd.DataFrame({"a": [1]})
        summary = column_summary(df)

        invalid_csv = "/nope/this/does/not/exist.csv"
        invalid_txt = "/nope/this/does/not/exist.txt"

        with self.assertRaises(Exception):
            export_summary(summary, invalid_csv, invalid_txt)


if __name__ == "__main__":
    unittest.main()
