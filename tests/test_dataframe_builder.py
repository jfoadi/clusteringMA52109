###
## cluster_maker - test file
## James Foadi - University of Bath
## November 2025
###

import unittest

import numpy as np
import pandas as pd

from cluster_maker.dataframe_builder import define_dataframe_structure, simulate_data
from cluster_maker.data_analyser import summarize_numeric

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

class TestSummarizeNumeric(unittest.TestCase):              ### CHANGE: new test class
    def test_summarize_numeric_with_mixed_df(self):
        df = pd.DataFrame({
            "a": [1, 2, 3, None],       # numeric with missing
            "b": [10.0, 20.0, 30.0, 40.0],  # numeric
            "c": [5, 5, 5, 5],          # numeric constant
            "text": ["x", "y", "z", "w"]  # non-numeric
        })
        summary = summarize_numeric(df)
        self.assertEqual(set(summary["column"].tolist()), {"a", "b", "c"})
        missing_a = summary.loc[summary["column"] == "a", "missing"].item()
        self.assertEqual(missing_a, 1)

if __name__ == "__main__":
    unittest.main()