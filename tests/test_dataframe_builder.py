###
## cluster_maker - test file
## James Foadi - University of Bath
## November 2025
###

import unittest

import numpy as np
import pandas as pd

from cluster_maker.dataframe_builder import define_dataframe_structure, simulate_data
from cluster_maker.data_analyser import column_statistics


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

##      c) Create at least one NEW test function in the existing test file
##         to check that the new analysis function in (a) works correctly
##         on a small DataFrame with:
##          - at least 3 numeric columns,
##          - at least 1 non-numeric column,
##          - at least 1 missing value.

    def test_column_statistics(self):
        df = pd.DataFrame({
            "A": [1, 2, 3, None],
            "B": [4.0, None, 6.0, 7.0],
            "C": [None, 6.0, None, None],
            "D": ["x", "y", "z", "w"]
        })
        stats_df = column_statistics(df)
        self.assertIn("A", stats_df.index)
        self.assertIn("B", stats_df.index)
        self.assertIn("C", stats_df.index) 
        self.assertNotIn("D", stats_df.index)
        self.assertAlmostEqual(stats_df.loc["A", "mean"], 2.0)
        self.assertEqual(stats_df.loc["A", "missing_count"], 1)
        self.assertEqual(stats_df.loc["B", "missing_count"], 1)
        self.assertEqual(stats_df.loc["C", "missing_count"], 3)

if __name__ == "__main__":
    unittest.main()