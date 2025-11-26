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



class TestDataAnalyser(unittest.TestCase):
    def test_get_numeric_summary(self):
        from cluster_maker.data_analyser import get_numeric_summary

        data = pd.DataFrame({
            "A": [1.0, 2.0, 3.0, np.nan, 5.0],
            "B": [50, 40, 30, 20, 10],
            "C": [2.2, 3.3, 4.4, 5.5, 6.6],
            "label": ["a", "b", "c", "d", "e"]
        })
        summary = get_numeric_summary(data)

        # Expected numeric columns only
        expected_columns = ["A", "B", "C"]
        self.assertListEqual(list(summary.index), expected_columns)

        # Check presence of required summary metrics
        for metric in ["mean", "std", "min", "max", "n_missing"]:
            self.assertIn(metric, summary.columns)
        
        # Check missing count is correct
        self.assertEqual(summary.loc["A", "n_missing"], 1)
        self.assertEqual(summary.loc["B", "n_missing"], 0)
        self.assertEqual(summary.loc["C", "n_missing"], 0)

        # Check a known mean value
        # Mean value of B is (50+40+30+20+10)/5 = 30
        self.assertAlmostEqual(summary.loc["B", "mean"], 30.0)

if __name__ == "__main__":
    unittest.main()