###
## cluster_maker – tests for cluster quality diagnostics
## Yas Akilakulasingam - University of Bath
## November 2025
###
#
# These tests ensure the Davies–Bouldin diagnostic behaves correctly
# under ideal conditions, noisy conditions, and invalid inputs.
#

import unittest
import numpy as np

from cluster_maker import compute_davies_bouldin
from cluster_maker import run_clustering


class TestQuality(unittest.TestCase):

    def test_dbi_valid_clusters(self):
        """DBI should be low for well-separated clusters."""
        rng = np.random.RandomState(0)
        X1 = rng.normal([0,0], 0.05, size=(30,2))
        X2 = rng.normal([5,5], 0.05, size=(30,2))
        X = np.vstack([X1, X2])
        labels = np.array([0]*30 + [1]*30)

        score = compute_davies_bouldin(X, labels)
        self.assertLess(score, 0.5)

    def test_dbi_higher_when_clusters_overlap(self):
        """DBI should worsen when clusters mix together."""
        rng = np.random.RandomState(0)
        X = rng.normal([0,0], 1.2, size=(100,2))
        labels = np.random.randint(0,2,size=100)

        score = compute_davies_bouldin(X, labels)
        self.assertGreaterEqual(score, 0.5)

    def test_dbi_invalid_types(self):
        with self.assertRaises(TypeError):
            compute_davies_bouldin("not array", np.array([0,1]))

    def test_dbi_mismatched_shapes(self):
        X = np.random.randn(10,2)
        labels = np.array([0,1,0])   # too short
        with self.assertRaises(ValueError):
            compute_davies_bouldin(X, labels)

    def test_dbi_with_interface(self):
        """run_clustering should compute DBI when compute_quality=True."""
        # simple 2-feature DataFrame
        import pandas as pd
        df = pd.DataFrame({
            "x": np.random.randn(40),
            "y": np.random.randn(40),
            "u": np.random.randn(40),
        })
        df.to_csv("temp.csv", index=False)

        result = run_clustering(
            input_path="temp.csv",
            feature_cols=["x","y"],
            k=2,
            compute_quality=True,
            standardise=True
        )

        self.assertIn("davies_bouldin", result["metrics"])


if __name__ == "__main__":
    unittest.main()