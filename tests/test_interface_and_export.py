###
## cluster_maker - interface and exporter tests
## MA52109 – Mock Exam
## November 2025
###

import unittest
import os
import pandas as pd
import numpy as np

from cluster_maker.interface import run_clustering
from cluster_maker.data_exporter import export_to_csv, export_summary


class TestInterfaceAndExport(unittest.TestCase):

    def test_run_clustering_missing_file(self):
        """run_clustering should raise FileNotFoundError when file is missing."""
        with self.assertRaises(FileNotFoundError):
            run_clustering(
                input_path="nonexistent.csv",
                feature_cols=["x1", "x2"],
                k=3
            )

    def test_run_clustering_missing_columns(self):
        """run_clustering should raise an error when required columns are missing."""
        # Create a temporary CSV without needed columns
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df.to_csv("temp_missing_cols.csv", index=False)

        with self.assertRaises(KeyError):
            run_clustering(
                input_path="temp_missing_cols.csv",
                feature_cols=["x1", "x2"],  # missing columns
                k=3
            )

        os.remove("temp_missing_cols.csv")

    def test_export_to_csv_valid(self):
        """export_to_csv should create a file when path is valid."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

        export_to_csv(df, "temp_export.csv")
        self.assertTrue(os.path.exists("temp_export.csv"))

        os.remove("temp_export.csv")

    def test_export_to_csv_invalid_path(self):
        """export_to_csv should raise an error for invalid paths."""
        df = pd.DataFrame({"a": [1, 2]})

        with self.assertRaises(Exception):
            export_to_csv(df, "nonexistent_dir/output.csv")

    def test_export_summary_files_created(self):
        """export_summary should create both CSV and TXT summaries."""
        df = pd.DataFrame({
            "a": [1, 2, None],
            "b": [10, 20, 30]
        })

        summary = pd.DataFrame({
            "mean": df.mean(),
            "std": df.std(),
            "min": df.min(),
            "max": df.max(),
            "n_missing": df.isna().sum()
        })

        export_summary(summary, "summary_test.csv", "summary_test.txt")

        self.assertTrue(os.path.exists("summary_test.csv"))
        self.assertTrue(os.path.exists("summary_test.txt"))

        os.remove("summary_test.csv")
        os.remove("summary_test.txt")


if __name__ == "__main__":
    unittest.main()
