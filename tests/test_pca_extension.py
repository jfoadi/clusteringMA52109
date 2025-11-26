###
## cluster_maker - test file
## Test for PCA extension in cluster_maker
###

import unittest
import numpy as np

from cluster_maker.pca_extension import pca_extension

class TestPCAExtension(unittest.TestCase):
    def test_pca_extension_captures_latent_factors(self):
        """
        Construct synthetic data with 2 latent factors and 5 observed features.
        X = Z @ W.T + noise where Z has shape (n_samples, 2) and W (5,2).
        With n_components=2 PCA should explain almost all variance.
        """
        rng = np.random.default_rng(42)
        n_samples = 500
        latent = rng.normal(size=(n_samples, 2)) # 2 latent factors

        # Weight matrix to map latent factors to observed features
        W = np.array([[1.0, 0.5],
                      [0.9, -0.4],
                      [-0.8, 0.6],
                      [0.7, 0.9],
                      [-0.5, -1.0]]) # shape (5,2)
        
        noise = rng.normal(scale=0.1, size=(n_samples, 5))

        # Observed data
        X = latent @ W.T + noise  # shape (n_samples, 5)

        # Apply PCA extension
        X_pca, pca = pca_extension(X, n_components=2, random_state=42)

        # Check shape
        self.assertEqual(X_pca.shape, (n_samples, 2))

        # The two components should capture almost all variance
        explained = np.sum(pca.explained_variance_ratio_)
        self.assertGreaterEqual(explained, 0.99)

        # 3) Reconstructing from the components should be close to the original
        X_reconstructed = pca.inverse_transform(X_pca)
        max_abs_diff = np.max(np.abs(X - X_reconstructed))
        self.assertLessEqual(max_abs_diff, 0.1)

if __name__ == "__main__":
    unittest.main()
