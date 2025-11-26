###
## cluster_maker - test file
## James Foadi - University of Bath
## November 2025
###

import unittest

import numpy as np
import pandas as pd

from cluster_maker.dataframe_builder import define_dataframe_structure, simulate_data
from cluster_maker.data_analyser import calculate_descriptive_statistics # NEW IMPORT


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

    # --- NEW TEST FUNCTION FOR TASK 3(c) ---
    def test_descriptive_statistics_mixed_dataframe(self):
        # Small DataFrame with 3 numeric columns, 1 non-numeric, and 1 missing value
        df = pd.DataFrame({
            "num1": [1, 2, 3, 4, np.nan],  # numeric with missing value
            "num2": [10, 20, 30, 40, 50],  # numeric
            "num3": [5.5, 6.5, 7.5, 8.5, 9.5],  # numeric
            "cat": ["a", "b", "c", "d", "e"]  # non-numeric
        })

        summary_df = calculate_descriptive_statistics(df)

        # Check that the result contains only numeric columns
        self.assertListEqual(list(summary_df.columns), ["num1", "num2", "num3"])

        # Check that count is correct for column with missing value
        self.assertEqual(summary_df.loc["count", "num1"], 4.0)

        # Check that mean is calculated correctly ignoring NaN
        self.assertAlmostEqual(summary_df.loc["mean", "num1"], 2.5)

        # Check that non-numeric column is ignored
        self.assertNotIn("cat", summary_df.columns)
if __name__ == "__main__":
    unittest.main()