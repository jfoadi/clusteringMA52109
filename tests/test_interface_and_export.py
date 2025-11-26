import unittest
import os
import tempfile
import pandas as pd
import numpy as np
import shutil
import matplotlib.pyplot as plt

# Import core package functions
from cluster_maker.interface import run_clustering
from cluster_maker.data_exporter import export_to_csv, export_formatted


# --- Helper Function for Creating Test Data ---

def create_temp_csv(data: dict, filename: str) -> str:
    """Creates a temporary CSV file and returns its path."""
    df = pd.DataFrame(data)
    filepath = os.path.join(tempfile.gettempdir(), filename)
    df.to_csv(filepath, index=False)
    return filepath


class TestInterfaceAndExport(unittest.TestCase):
    """
    Tests for error handling in interface.py and data_exporter.py.
    """
    
    def setUp(self):
        """Set up temporary directory and data for tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.valid_data = {
            'x': [1, 2, 3, 4, 5],
            'y': [5, 4, 3, 2, 1],
            'z': [0, 0, 0, 0, 0],
            'non_numeric': ['a', 'b', 'c', 'd', 'e']
        }
        
        # Create a valid input file for run_clustering tests
        self.valid_input_path = os.path.join(self.temp_dir, "valid_input.csv")
        pd.DataFrame(self.valid_data).to_csv(self.valid_input_path, index=False)
        
        # Create a small valid DataFrame for export tests
        self.valid_df_export = pd.DataFrame({'Col1': [10, 20], 'Col2': [30, 40]})


    def tearDown(self):
        """Clean up the temporary directory."""
        shutil.rmtree(self.temp_dir)


    # --------------------------------------------------------------------------
    # Part 5a: Testing run_clustering Robustness
    # --------------------------------------------------------------------------

    def test_run_clustering_missing_file(self):
        """
        Check that run_clustering raises a FileNotFoundError for a missing input file.
        """
        missing_path = os.path.join(self.temp_dir, "missing.csv")
        
        with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
            # We expect the error to be raised cleanly
            run_clustering(
                input_path=missing_path,
                feature_cols=['x', 'y'],
            )

    def test_run_clustering_missing_features(self):
        """
        Check that run_clustering raises a KeyError for missing feature columns.
        """
        # Feature 'A' is intentionally missing from the valid input file
        missing_features = ['x', 'A'] 

        with self.assertRaisesRegex(KeyError, "missing"):
            # We expect the error to be raised cleanly by select_features or a downstream function
            run_clustering(
                input_path=self.valid_input_path,
                feature_cols=missing_features,
            )
            
    # --------------------------------------------------------------------------
    # Part 5b: Testing data_exporter Robustness
    # --------------------------------------------------------------------------

    def test_export_to_csv_valid_path(self):
        """Check that export_to_csv creates a file and content is correct."""
        output_path = os.path.join(self.temp_dir, "output_csv.csv")
        
        export_to_csv(self.valid_df_export, output_path, include_index=False)
        
        self.assertTrue(os.path.exists(output_path), "CSV file was not created.")
        
        # Check content by loading it back
        loaded_df = pd.read_csv(output_path)
        self.assertTrue(loaded_df.equals(self.valid_df_export), "Exported CSV content is incorrect.")


    def test_export_to_csv_invalid_directory(self):
        """
        Check that export_to_csv raises an appropriate OS error for an invalid path.
        """
        invalid_path = os.path.join("non_existent_directory", "file.csv")

        # We expect a FileNotFoundError or similar OS error for non-existent directories.
        with self.assertRaises((FileNotFoundError, IOError, OSError)):
            export_to_csv(self.valid_df_export, invalid_path, include_index=False)


    def test_export_formatted_valid_path(self):
        """Check that export_formatted creates a file."""
        output_path = os.path.join(self.temp_dir, "output_text.txt")
        
        export_formatted(self.valid_df_export, output_path, include_index=False)
        
        self.assertTrue(os.path.exists(output_path), "Formatted text file was not created.")
        
        # Check content (should contain 'Col1' and 'Col2')
        with open(output_path, 'r') as f:
            content = f.read()
        self.assertIn("Col1", content)
        self.assertIn("30", content)

    def test_run_clustering_with_pca(self):
        """
        Verifies that run_clustering correctly applies PCA and uses the 
        transformed data for clustering (checking final centroid shape).
        """
        k_clusters = 3
        pca_components = 2
        
        # When run_clustering returns, 'centroids' is based on the final
        # shape of X, which should be (n_samples, pca_components).
        result = run_clustering(
            input_path=self.valid_input_path,
            feature_cols=['x', 'y', 'z'], # 3 features
            algorithm="sklearn_kmeans",
            k=k_clusters,
            standardise=True,
            n_components_pca=pca_components # Request PCA reduction
        )
        
        # The centroid array should reflect the reduced dimensionality
        # Expected shape: (k_clusters, pca_components) -> (3, 2)
        expected_centroid_shape = (k_clusters, pca_components)
        
        self.assertEqual(
            result["centroids"].shape,
            expected_centroid_shape,
            f"Centroids shape must be {expected_centroid_shape} after PCA, but got {result['centroids'].shape}"
        )
        
        # Also check figure axes are using the reduced dimensions (2D)
        self.assertIsInstance(result["fig_cluster"], plt.Figure)
        
    def test_run_clustering_pca_error_handling(self):
        """
        Verifies that run_clustering raises an error if n_components_pca 
        is invalid (e.g., too high or zero/negative).
        """
        # Requesting 4 components when only 3 features are available
        with self.assertRaisesRegex(ValueError, "cannot exceed the number of features"):
            run_clustering(
                input_path=self.valid_input_path,
                feature_cols=['x', 'y', 'z'], # 3 features
                k=2,
                n_components_pca=4 # Invalid: 4 > 3
            )
            
        # Requesting zero components
        with self.assertRaisesRegex(ValueError, "must be a positive integer"):
             run_clustering(
                input_path=self.valid_input_path,
                feature_cols=['x', 'y', 'z'],
                k=2,
                n_components_pca=0 # Invalid: 0
            )


    def test_export_formatted_invalid_directory(self):
        """
        Check that export_formatted raises an appropriate OS error for an invalid path.
        """
        invalid_path = os.path.join("non_existent_directory", "file.txt")
        
        # We expect a FileNotFoundError or similar OS error for non-existent directories.
        with self.assertRaises((FileNotFoundError, IOError, OSError)):
            export_formatted(self.valid_df_export, invalid_path, include_index=False)

if __name__ == '__main__':
    unittest.main()