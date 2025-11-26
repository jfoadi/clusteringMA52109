###
## cluster_maker – PCA tests
###

import unittest
import numpy as np
from cluster_maker.pca_tools import apply_pca


class TestPCA(unittest.TestCase):

    def test_pca_reduces_dimensions(self):
        X = np.random.randn(50, 5)  # 50 samples, 5 features
        X_reduced = apply_pca(X, n_components=2)

        # Correct reduced shape
        self.assertEqual(X_reduced.shape, (50, 2))

    def test_pca_invalid_n_components(self):
        X = np.random.randn(10, 3)
        with self.assertRaises(ValueError):
            apply_pca(X, n_components=0)

    def test_pca_type_error(self):
        with self.assertRaises(TypeError):
            apply_pca("not an array", n_components=2)


if __name__ == "__main__":
    unittest.main()
