import unittest
import pandas as pd
import numpy as np
from cluster_maker.preprocessing import pca_preprocess

class TestPCAPreprocessing(unittest.TestCase):
    def test_pca_basic(self):
        # Sample DataFrame with numeric columns
        df = pd.DataFrame({
            "x": np.arange(10),
            "y": np.arange(10, 20),
            "z": np.arange(20, 30),
            "label": ["A"]*10  # non-numeric column
        })
        result = pca_preprocess(df, n_components=2)
        self.assertEqual(result.shape[1], 2)  # only 2 PCs kept
        self.assertTrue(np.all(np.isfinite(result.values)))  # no NaNs or infs

    def test_pca_fraction_variance(self):
        df = pd.DataFrame(np.random.rand(50, 5))
        result = pca_preprocess(df, n_components=0.90)
        # Number of components should be <= original features
        self.assertLessEqual(result.shape[1], 5)
        self.assertTrue(np.all(np.isfinite(result.values)))

if __name__ == "__main__":
    unittest.main()
