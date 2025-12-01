###
## cluster_maker – tests for PCA preprocessing
## Yas Akilakulasingam - University of Bath
## November 2025
###


# These tests validate the correctness and numerical stability of the
# PCA-based preprocessing extension. They check that dimensionality
# reduction behaves as expected, explained variance ratios are valid,
# and meaningful errors are raised for invalid inputs.


import unittest
import numpy as np
import os
import pandas as pd
from cluster_maker.preprocessing import apply_pca

class TestPCA(unittest.TestCase):

    def test_pca_reduces_dimensions(self):
        
        X = np.array([
            [1.0, 2.0, 3.0],
            [1.1, 2.1, 3.1],
            [0.9, 1.9, 2.9]
        ])
        # PCA should return the correct number of components and produce valid explained-variance 
        # ratios (between 0 and 1, summing to <= 1)
        
        X_pca, var = apply_pca(X, n_components=2)

        # Correct dimensionality
        self.assertEqual(X_pca.shape[1], 2)

        # Valid variance values
        self.assertTrue(np.all(var >= 0))
        self.assertTrue(np.all(var <= 1))

        # Sum must be <= 1, not necessarily close to 1
        self.assertLessEqual(var.sum(), 1.0)

        # No NaNs
        self.assertFalse(np.isnan(var).any())

    def test_invalid_components(self):
        X = np.random.normal(size=(10, 4))

        with self.assertRaises(ValueError):
            apply_pca(X, n_components=0)

        with self.assertRaises(ValueError):
            apply_pca(X, n_components=10)
            
    def test_pca_in_interface(self):
        """
        Check that PCA integrates correctly with run_clustering and that
        the output dictionary contains the expected PCA variance metrics.
        """

        # --- Create small numeric DataFrame ---
        df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": [1.1, 2.2, 3.1, 4.3],
            "z": [0.5, 0.7, 0.9, 1.2],  # third numeric column
        })

        # Save to temporary CSV
        temp_csv = "temp_test_pca.csv"
        df.to_csv(temp_csv, index=False)

        # --- Run clustering with PCA enabled ---
        from cluster_maker.interface import run_clustering

        result = run_clustering(
            input_path=temp_csv,
            feature_cols=["x", "y", "z"],
            algorithm="kmeans",
            k=2,
            standardise=True,
            use_pca=True,
            pca_components=2,
            compute_elbow=False,
            random_state=0,
        )

        # --- Assertions ---
        # PCA variance should be added inside metrics
        self.assertIn("pca_variance", result["metrics"])
        var = result["metrics"]["pca_variance"]

        # Should have length 2 (we asked for 2 PCA components)
        self.assertEqual(len(var), 2)

        # All entries should be between 0 and 1
        self.assertTrue(np.all(var >= 0))
        self.assertTrue(np.all(var <= 1))

        # PCA output should be meaningful (not NaNs)
        self.assertFalse(np.isnan(var).any())

        # Cleanup
        os.remove(temp_csv)       
            
            

if __name__ == "__main__":
    unittest.main()