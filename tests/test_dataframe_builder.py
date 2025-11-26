###
## cluster_maker - test file
## James Foadi - University of Bath
## November 2025
###

import unittest

import numpy as np
import pandas as pd
from cluster_maker.data_analyser import calculate_summary_stats

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
    
    def test_calculate_summary_stats(self):
        data = pd.DataFrame({
            "A": [1, 2, 3, np.nan],
            "B": [4, 5, 6, 7],
            "C": [np.nan, np.nan, np.nan, np.nan]
        })
        stats_df = calculate_summary_stats(data)
        self.assertIn("mean", stats_df.columns)
        self.assertIn("std", stats_df.columns)
        self.assertIn("missing_count", stats_df.columns)
        self.assertAlmostEqual(stats_df.loc["A", "mean"], 2.0)
        self.assertEqual(stats_df.loc["C", "missing_count"], 4)
        


if __name__ == "__main__":
    unittest.main()