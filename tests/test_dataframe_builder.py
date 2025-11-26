###
## cluster_maker - test file
## James Foadi - University of Bath
## November 2025
###

import unittest

import numpy as np
import pandas as pd

from cluster_maker.dataframe_builder import define_dataframe_structure, simulate_data


class TestDataFrameBuilder(unittest.TestCase):
    def test_define_dataframe_structure_basic(self):
        column_specs = [
            {"name": "x", "reps": [0.0, 1.0, 2.0]},
            {"name": "y", "reps": [10.0, 11.0, 12.0]},
        ]
        seed_df = define_dataframe_structure(column_specs)
        self.assertEqual(seed_df.shape, (3, 2))
        self.assertListEqual(list(seed_df.columns), ["x", "y"])
        self.assertTrue(np.allclose(seed_df["x"].values, [0.0, 1.0, 2.0]))

    def test_simulate_data_shape(self):
        column_specs = [
            {"name": "x", "reps": [0.0, 5.0]},
            {"name": "y", "reps": [2.0, 4.0]},
        ]
        seed_df = define_dataframe_structure(column_specs)
        data = simulate_data(seed_df, n_points=100, random_state=1)
        self.assertEqual(data.shape[0], 100)
        self.assertIn("true_cluster", data.columns)

    def test_calculate_numeric_summary(self):
        from cluster_maker.data_analyser import calculate_numeric_summary

        # Create a DataFrame with mixed types and missing values
        df = pd.DataFrame({
            "a": [1.0, 2.0, 3.0, np.nan],
            "b": [10, 20, 30, 40],
            "c": ["x", "y", "z", "w"]
        })

        summary = calculate_numeric_summary(df)

        # Check shape: 2 numeric columns ("a", "b"), 5 stats (mean, std, min, max, missing_count)
        self.assertEqual(summary.shape, (2, 5))
        
        # Check index (should be column names)
        self.assertListEqual(sorted(summary.index.tolist()), ["a", "b"])

        # Check specific values
        self.assertEqual(summary.loc["a", "missing_count"], 1)
        self.assertEqual(summary.loc["b", "missing_count"], 0)
        self.assertAlmostEqual(summary.loc["b", "mean"], 25.0)

    def test_apply_pca(self):
        from cluster_maker.preprocessing import apply_pca
        
        # Create a DataFrame with 3 correlated features
        # x = t, y = 2t, z = 3t
        t = np.linspace(0, 10, 20)
        df = pd.DataFrame({
            "x": t,
            "y": 2 * t,
            "z": 3 * t
        })

        # Apply PCA with 2 components
        pca_df = apply_pca(df, n_components=2)

        # Check shape
        self.assertEqual(pca_df.shape, (20, 2))
        self.assertListEqual(list(pca_df.columns), ["PC1", "PC2"])

        # Since data is perfectly linear, PC1 should explain most variance
        # and PC2 should be close to 0 (or at least much smaller variance)
        self.assertTrue(pca_df["PC1"].var() > pca_df["PC2"].var())


if __name__ == "__main__":
    unittest.main()