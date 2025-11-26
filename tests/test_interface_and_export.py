import unittest, os, pandas as pd
from cluster_maker import export_summary
from cluster_maker.interface import run_clustering

class TestInterfaceAndExport(unittest.TestCase):
    def test_run_clustering_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            run_clustering(input_path="nonexistent.csv",
                           feature_cols=["x","y"], algorithm="kmeans", k=3, standardise=True)

    def test_run_clustering_missing_feature_columns(self):
        df = pd.DataFrame({"a":[1,2,3], "b":[4,5,6]})
        tmp_csv = "tmp_missing_cols.csv"
        df.to_csv(tmp_csv, index=False)
        try:
            with self.assertRaises((KeyError, ValueError, TypeError)):
                run_clustering(input_path=tmp_csv,
                               feature_cols=["x","y"], algorithm="kmeans", k=2, standardise=False)
        finally:
            os.remove(tmp_csv)

    def test_export_summary_valid_and_invalid_path(self):
        df = pd.DataFrame({"column":["a"], "mean":[1.0], "std":[0.0],
                           "min":[1.0], "max":[1.0], "missing":[0]})
        valid_dir = "tmp_out"; os.makedirs(valid_dir, exist_ok=True)
        csv_path = os.path.join