###
## cluster_maker - test file
## James Foadi - University of Bath
## November 2025
###

import unittest

import numpy as np
import pandas as pd

from cluster_maker.dataframe_builder import define_dataframe_structure, simulate_data
from cluster_maker.data_analyser import summarise_numeric


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
        
    def test_summarise_numeric_mixed_dataframe(self):
        # DataFrame with:
        # - 3 numeric columns
        # - 1 non-numeric column
        # - at least 1 missing value
        df = pd.DataFrame(
            {
                "a": [1.0, 2.0, None, 4.0],   # one missing
                "b": [10, 20, 30, 40],        # no missing
                "c": [0.5, 0.5, 0.5, 0.5],    # zero std
                "label": ["x", "y", "z", "w"] # non-numeric
            }
        )

        summary = summarise_numeric(df)

        # Only numeric columns should be included
        self.assertListEqual(list(summary.index), ["a", "b", "c"])

        # Check missing-value counts
        self.assertEqual(summary.loc["a", "n_missing"], 1)
        self.assertEqual(summary.loc["b", "n_missing"], 0)

        # Check a simple mean calculation for column 'a'
        expected_mean_a = (1.0 + 2.0 + 4.0) / 3.0
        self.assertAlmostEqual(summary.loc["a", "mean"], expected_mean_a)

        # Non-numeric column should not be in the summary
        self.assertNotIn("label", summary.index)



if __name__ == "__main__":
    unittest.main()