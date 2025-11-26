import unittest
import os
import tempfile
import pandas as pd

from cluster_maker.interface import run_clustering
from cluster_maker.data_exporter import export_to_csv, export_formatted


class TestInterfaceAndExport(unittest.TestCase):
    # ------------------------------
    # Tests for high-level interface
    # ------------------------------
    def test_run_clustering_missing_file(self):
        missing_file = "non_existent_file.csv"
        with self.assertRaises(FileNotFoundError):
            run_clustering(missing_file, output_dir=tempfile.gettempdir())

    def test_run_clustering_missing_columns(self):
        # Create a temp CSV with missing required feature columns
        df = pd.DataFrame({"wrong_col1": [1, 2], "wrong_col2": [3, 4]})
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            df.to_csv(tmp.name, index=False)
            temp_file = tmp.name

        try:
            with self.assertRaises(ValueError):
                run_clustering(temp_file, output_dir=tempfile.gettempdir())
        finally:
            os.remove(temp_file)

    # ------------------------------
    # Tests for data_exporter functions
    # ------------------------------
    def test_export_to_csv_creates_file(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "test.csv")
            export_to_csv(df, csv_path)
            self.assertTrue(os.path.isfile(csv_path))

    def test_export_formatted_creates_file(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        with tempfile.TemporaryDirectory() as tmpdir:
            txt_path = os.path.join(tmpdir, "test.txt")
            export_formatted(df, txt_path)
            self.assertTrue(os.path.isfile(txt_path))

    def test_export_to_csv_invalid_path(self):
        df = pd.DataFrame({"a": [1, 2]})
        invalid_path = "/non_existent_dir/file.csv"
        with self.assertRaises(Exception):
            export_to_csv(df, invalid_path)

    def test_export_formatted_invalid_path(self):
        df = pd.DataFrame({"a": [1, 2]})
        invalid_path = "/non_existent_dir/file.txt"
        with self.assertRaises(Exception):
            export_formatted(df, invalid_path)


if __name__ == "__main__":
    unittest.main()
