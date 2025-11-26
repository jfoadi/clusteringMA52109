###
## test_interface_and_export.py
## MA52109 – Task 5 (Completed)
## This test file verifies interface error handling and exporter robustness.
###

import os
import unittest
import pandas as pd
from tempfile import TemporaryDirectory

from cluster_maker.interface import run_clustering
from cluster_maker.data_exporter import (
    export_to_csv,
    export_formatted,
    export_numeric_summary,
)
from cluster_maker.data_analyser import numeric_summary


print("\n=== Running Task 5: Interface & Export Tests ===")


class TestInterfaceAndExport(unittest.TestCase):

    # ------------------------------------------------------------
    # 5a – Tests for run_clustering error handling
    # ------------------------------------------------------------

    def test_run_clustering_missing_file(self):
        print("\n[TEST] run_clustering with missing input file")
        with self.assertRaises(FileNotFoundError):
            run_clustering(
                input_path="no_such_file.csv",
                feature_cols=["x", "y"],
                k=3,
            )
        print("[PASS] Correctly raised FileNotFoundError")

    def test_run_clustering_missing_columns(self):
        print("\n[TEST] run_clustering with missing feature columns")

        with TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test.csv")
            df = pd.DataFrame({
                "a": [1, 2, 3],
                "b": [4, 5, 6],
            })
            df.to_csv(path, index=False)

            with self.assertRaises(KeyError):
                run_clustering(
                    input_path=path,
                    feature_cols=["x", "y"],
                    k=2,
                )
        print("[PASS] Correctly raised KeyError for missing columns")

    # ------------------------------------------------------------
    # 5b – Tests for exporter functions
    # ------------------------------------------------------------

    def test_export_to_csv_and_formatted_valid(self):
        print("\n[TEST] export_to_csv and export_formatted with valid inputs")
        with TemporaryDirectory() as tmp:
            csv_out = os.path.join(tmp, "out.csv")
            txt_out = os.path.join(tmp, "out.txt")

            df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

            export_to_csv(df, csv_out)
            self.assertTrue(os.path.exists(csv_out))

            export_formatted(df, txt_out)
            self.assertTrue(os.path.exists(txt_out))

        print("[PASS] CSV and text export files created successfully")

    def test_export_to_csv_invalid_path(self):
        print("\n[TEST] export_to_csv with invalid path")
        df = pd.DataFrame({"x": [1, 2, 3]})
        invalid_path = "invalid_directory/output.csv"

        with self.assertRaises(Exception):
            export_to_csv(df, invalid_path)

        print("[PASS] Correctly raised error on invalid path")

    def test_export_numeric_summary(self):
        print("\n[TEST] export_numeric_summary with valid summary")

        with TemporaryDirectory() as tmp:
            csv_out = os.path.join(tmp, "summary.csv")
            txt_out = os.path.join(tmp, "summary.txt")

            df = pd.DataFrame({
                "a": [1, 2, None],
                "b": [10, 20, 30],
                "label": ["x", "y", "z"],  # non-numeric
            })

            summary = numeric_summary(df)

            export_numeric_summary(summary, csv_out, txt_out)

            self.assertTrue(os.path.exists(csv_out))
            self.assertTrue(os.path.exists(txt_out))

        print("[PASS] Numeric summary exported successfully")


print("\n--- Task 5 test file loaded successfully ---\n")


if __name__ == "__main__":
    print("=== Starting Task 5 test suite ===")
    unittest.main()
    print("\n=== All Task 5 tests completed successfully ===\n")
