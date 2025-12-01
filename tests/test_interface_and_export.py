## 5) Create a NEW test file in the "tests" directory called
## "test_interface_and_export.py". In this file, using only the standard
## "unittest" module and the allowed libraries, write tests that check that:
##      a) Calling the high-level interface function (for example
##      "run_clustering" in "interface.py") with:
##          - a missing input file; or
##          - a CSV file that does not contain the required feature
##            columns;
##         results in a clean, controlled error, which means:
##          - no raw Python traceback;
##          - a raised exception of an appropriate type (e.g. ValueError
##            or FileNotFoundError) with a clear error message.
##      b) The exporting functions in "data_exporter.py":
##          - create output files when given valid inputs;
##          - raise a clear, controlled error if given an invalid path
##            (for example, a directory that does not exist).
##
## Hint: You may create small temporary DataFrames inside the tests instead
## of reading from real CSV files.

import unittest
import os
import pandas as pd
from cluster_maker.interface import run_clustering
from cluster_maker.data_exporter import export_clusters

class TestInterfaceAndExport(unittest.TestCase):
    def test_run_clustering_missing_file(self):
        """Test that run_clustering raises FileNotFoundError for missing input file."""
        with self.assertRaises(FileNotFoundError) as context:
            run_clustering("non_existent_file.csv", features=['feature1', 'feature2'])
        self.assertIn("No such file or directory", str(context.exception))

    def test_run_clustering_invalid_columns(self):
        """Test that run_clustering raises ValueError for missing required feature columns."""
        temp_file = "temp_invalid_columns.csv"
        df = pd.DataFrame({'wrong_feature1': [1, 2], 'wrong_feature2': [3, 4]})
        df.to_csv(temp_file, index=False)

        try:
            with self.assertRaises(ValueError) as context:
                run_clustering(temp_file, features=['feature1', 'feature2'])
            self.assertIn("Feature columns not found in the data", str(context.exception))
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def test_export_clusters_valid_input(self):
        """Test that export_clusters creates output files with valid inputs."""
        # Create a temporary DataFrame
        df = pd.DataFrame({'cluster': [0, 1], 'data': [10, 20]})

        output_file = "temp_clusters.csv"
        try:
            export_clusters(df, output_file)
            self.assertTrue(os.path.exists(output_file))
        finally:
            if os.path.exists(output_file):
                os.remove(output_file)

    def test_export_clusters_invalid_path(self):
        """Test that export_clusters raises FileNotFoundError for invalid output path."""
        df = pd.DataFrame({'cluster': [0, 1], 'data': [10, 20]})
        invalid_path = "/non_existent_directory/clusters.csv"

        with self.assertRaises(FileNotFoundError) as context:
            export_clusters(df, invalid_path)
        self.assertIn("No such file or directory", str(context.exception))

if __name__ == '__main__':
    unittest.main()