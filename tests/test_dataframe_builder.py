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
        
    def test_summarise_numeric_columns(self):
        # --- Prepare test DataFrame ---
        df = pd.DataFrame({
            "a": [1, 2, 3, None],     # numeric with missing
            "b": [10, 20, 30, 40],    # numeric
            "c": [5.5, None, 6.5, 7], # numeric with missing
            "text": ["x", "y", "z", "w"]  # non-numeric
        })

        # --- Run summary function ---
        from cluster_maker.data_analyser import summarise_numeric_columns
        summary = summarise_numeric_columns(df)

        # --- Assertions ---
        # Should only contain numeric columns
        self.assertListEqual(list(summary.index), ["a", "b", "c"])

        # Correct missing values
        self.assertEqual(summary.loc["a", "missing_values"], 1)
        self.assertEqual(summary.loc["c", "missing_values"], 1)

        # Mean calculation sanity check for one column
        self.assertAlmostEqual(summary.loc["b", "mean"], 25.0)

        # Confirm non-numeric column is excluded
        self.assertNotIn("text", summary.index)


if __name__ == "__main__":
    unittest.main()