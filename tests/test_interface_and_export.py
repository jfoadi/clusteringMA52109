###
## cluster_maker - test interface and export
## James Foadi - University of Bath
## November 2025
###

import unittest
import os
import pandas as pd
from cluster_maker.interface import run_clustering
from cluster_maker.data_exporter import export_to_csv

class TestInterfaceAndExport(unittest.TestCase):
    def test_run_clustering_missing_file(self):
        """Test that run_clustering raises FileNotFoundError for missing input file."""
        with self.assertRaises(FileNotFoundError):
            run_clustering(
                input_path="non_existent_file.csv",
                feature_cols=["x", "y"],
                algorithm="kmeans",
                k=3
            )

    def test_run_clustering_missing_columns(self):
        """Test that run_clustering raises ValueError if columns are missing."""
        # Create a temporary CSV file
        temp_csv = "temp_test_data.csv"
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df.to_csv(temp_csv, index=False)

        try:
            with self.assertRaises(ValueError):
                run_clustering(
                    input_path=temp_csv,
                    feature_cols=["x", "y"], # These columns don't exist
                    algorithm="kmeans",
                    k=3
                )
        finally:
            if os.path.exists(temp_csv):
                os.remove(temp_csv)

    def test_export_to_csv_invalid_path(self):
        """Test that export_to_csv raises OSError (or similar) for invalid paths."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        
        # Try to write to a directory that doesn't exist
        invalid_path = "non_existent_dir/output.csv"
        
        with self.assertRaises(OSError):
            export_to_csv(df, invalid_path)

if __name__ == "__main__":
    unittest.main()
