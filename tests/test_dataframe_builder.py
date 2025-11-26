###
## cluster_maker - test file
## James Foadi - University of Bath
## November 2025
###

import unittest

import numpy as np
import pandas as pd

from cluster_maker.dataframe_builder import define_dataframe_structure, simulate_data
from cluster_maker.data_analyser import calculate_extended_statistics

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
        
    def test_calculate_extended_statistics_mixed(self):
        """
        Test extended statistics on a DataFrame with:
         - 3 numeric columns
         - 1 non-numeric column
         - Missing values
        """
        data = pd.DataFrame({
            "A": [1.0, 2.0, 3.0, 4.0],      # Numeric
            "B": [10, 20, np.nan, 40],      # Numeric with NaN
            "C": [0.5, 0.5, 0.5, 0.5],      # Numeric constant
            "D": ["cat", "dog", "fish", "bird"] # Non-numeric
        })

        stats = calculate_extended_statistics(data)

        # Check shape: Should have 3 rows (A, B, C) and 5 cols (mean, std, min, max, n_missing)
        self.assertEqual(stats.shape, (3, 5))
        
        # Check that the non-numeric column 'D' was ignored
        self.assertNotIn("D", stats.index)
        self.assertIn("A", stats.index)

        # Check specific values for column B (which has a NaN)
        # Mean of 10, 20, 40 is 70/3 = 23.333...
        expected_mean_b = (10 + 20 + 40) / 3
        self.assertAlmostEqual(stats.loc["B", "mean"], expected_mean_b, places=4)
        
        # Check missing count for B
        self.assertEqual(stats.loc["B", "n_missing"], 1.0)

if __name__ == "__main__":
    unittest.main()

