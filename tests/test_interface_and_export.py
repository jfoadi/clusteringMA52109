###
## Tests for interface and export functionality
## Task 5 – MA52109 Mock Practical
###

import os
import unittest
import tempfile
import pandas as pd

from cluster_maker.interface import run_clustering
from cluster_maker.data_exporter import export_summary
from cluster_maker.data_analyser import summarise_numeric


class TestInterfaceAndExport(unittest.TestCase):

    # ---------------------------------------------------------------
    # 5(a) Test run_clustering behaviour on invalid input
    # ---------------------------------------------------------------

    def test_run_clustering_missing_file(self):
        """run_clustering must raise FileNotFoundError for missing input."""
        with self.assertRaises(FileNotFoundError):
            run_clustering(
                input_path="non_existent_file.csv",
                feature_cols=["a", "b"],
                algorithm="kmeans",
                k=3,
                standardise=True,
                output_path="dummy.csv",
                random_state=42,
                compute_elbow=False,
            )

    def test_run_clustering_invalid_columns(self):
        """run_clustering must raise ValueError if required columns are missing."""
        # Create a temporary CSV with columns that do NOT match the required ones
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_csv_path = os.path.join(tmpdir, "bad_data.csv")

            df = pd.DataFrame({
                "x": [1, 2, 3],
                "y": [4, 5, 6],
                # missing 'a' and 'b'
            })
            df.to_csv(bad_csv_path, index=False)

            with self.assertRaises(ValueError):
                run_clustering(
                    input_path=bad_csv_path,
                    feature_cols=["a", "b"],   # these are not in the CSV
                    algorithm="kmeans",
                    k=3,
                    standardise=True,
                    output_path=os.path.join(tmpdir, "out.csv"),
                    random_state=42,
                    compute_elbow=False,
                )

    # ---------------------------------------------------------------
    # 5(b) Test export_summary behaviour
    # ---------------------------------------------------------------

    def test_export_summary_creates_files(self):
        """export_summary must create both CSV and text output files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create summary DataFrame manually
            df = pd.DataFrame({
                "a": [1, 2, None],
                "b": [10, 20, 30]
            })
            summary = summarise_numeric(df)

            csv_path = os.path.join(tmpdir, "summary.csv")
            txt_path = os.path.join(tmpdir, "summary.txt")

            export_summary(summary, csv_path, txt_path)

            self.assertTrue(os.path.exists(csv_path))
            self.assertTrue(os.path.exists(txt_path))

    def test_export_summary_invalid_directory(self):
        """export_summary must raise FileNotFoundError for invalid output directory."""
        # Summary DataFrame
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [4, 5, 6]
        })
        summary = summarise_numeric(df)

        # Invalid directory path
        csv_path = "no_such_directory/summary.csv"
        txt_path = "no_such_directory/summary.txt"

        with self.assertRaises(FileNotFoundError):
            export_summary(summary, csv_path, txt_path)


if __name__ == "__main__":
    unittest.main()
