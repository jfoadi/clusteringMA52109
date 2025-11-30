###
## cluster_maker - test_pca_extension.py
## Test correctness and error handling of the PCA extension (Task 6).
## November 2025
###

import unittest
import numpy as np

# Import the new function
from cluster_maker.preprocessing import apply_pca 

class TestPCAExtension(unittest.TestCase):

    def setUp(self):
        """
        Create a high-dimensional dataset (5D) for testing PCA.
        Data is designed to be highly correlated along some dimensions.
        """
        np.random.seed(42)
        # Create 100 samples of 5 features (X1 to X5)
        self.n_samples = 100
        self.n_features = 5
        
        # Highly correlated data: X1, X2, X3 are related, X4, X5 are noise/less related
        X1 = np.random.normal(0, 1, self.n_samples)
        X2 = X1 * 0.5 + np.random.normal(0, 0.1, self.n_samples)
        X3 = X1 * 1.5 + np.random.normal(0, 0.2, self.n_samples)
        X4 = np.random.normal(5, 1, self.n_samples)
        X5 = np.random.normal(10, 2, self.n_samples)
        
        self.X_original = np.column_stack([X1, X2, X3, X4, X5])


    def test_apply_pca_shape_reduction(self):
        """Verify that the output shape matches the requested number of components."""
        n_components_target = 3
        X_reduced = apply_pca(self.X_original, n_components=n_components_target)
        
        self.assertEqual(X_reduced.shape, (self.n_samples, n_components_target),
                         "PCA output shape is incorrect after reduction.")

    def test_apply_pca_correctness(self):
        """
        Verify that PCA performs a transformation (i.e., data is changed) 
        and the transformed variance is non-zero.
        """
        n_components_target = 3
        X_reduced = apply_pca(self.X_original, n_components=n_components_target)

        # Check that the reduced data is not equal to the original features 
        # (A simple check that a transformation occurred)
        self.assertFalse(np.allclose(X_reduced[:, 0], self.X_original[:, 0]),
                         "First PC should not be equal to the first original feature.")

        # Check that the variance is distributed (first component has highest variance)
        variances = np.var(X_reduced, axis=0)
        self.assertTrue(variances[0] > variances[1], 
                        "First principal component should have the highest variance.")
        self.assertTrue(all(v > 1e-6 for v in variances), 
                        "All component variances should be non-zero.")


    def test_apply_pca_invalid_n_components_error_handling(self):
        """Verify that controlled ValueErrors are raised for invalid n_components."""
        # Case 1: n_components > n_features
        with self.assertRaisesRegex(ValueError, r"n_components \(\d+\) must be .* <= number of features \(5\)."):
            apply_pca(self.X_original, n_components=self.n_features + 1)
        
        # Case 2: n_components <= 0
        with self.assertRaisesRegex(ValueError, r"n_components \(\d+\) must be > 0"):
            apply_pca(self.X_original, n_components=0)


if __name__ == "__main__":
    unittest.main()