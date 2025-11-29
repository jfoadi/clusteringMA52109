###
## cluster_maker - test file
## James Foadi - University of Bath
## November 2025
###

import unittest

import numpy as np
import pandas as pd

from cluster_maker.dataframe_builder import define_dataframe_structure, simulate_data
from cluster_maker.data_analyser import numeric_statistics


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

    def test_numeric_statistics_mixed_columns(self):
        # DataFrame with 3 numeric columns, 1 non-numeric column, and a missing value
        df = pd.DataFrame({
            "a": [1.0, 2.0, None],
            "b": [3.0, 4.0, 5.0],
            "c": [0.5, 0.5, 0.5],
            "d": ["x", "y", "z"],
        })

        stats = numeric_statistics(df)

        # Numeric columns only
        self.assertCountEqual(list(stats.index), ["a", "b", "c"])

        # Required summary columns
        for col in ["mean", "sd", "min", "max", "missing_values"]:
            self.assertIn(col, stats.columns)

        # missing value count for 'a' is 1
        self.assertEqual(int(stats.loc["a", "missing_values"]), 1)

        # mean for b is 4.0
        self.assertAlmostEqual(float(stats.loc["b", "mean"]), 4.0)

        # std for c is zero
        self.assertAlmostEqual(float(stats.loc["c", "sd"]), 0.0)


if __name__ == "__main__":
    unittest.main()